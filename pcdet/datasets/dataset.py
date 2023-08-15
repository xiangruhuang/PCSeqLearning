from collections import defaultdict
from pathlib import Path

import numpy as np
import torch_cluster
import torch
import pickle
import torch.utils.data as torch_data

from ..utils import common_utils, polar_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        if self.dataset_cfg is None:
            return

        self.training = training
        self.box_classes = dataset_cfg.get("BOX_CLASSES", None)
        segmentation_cfg = self.dataset_cfg.get("SEGMENTATION_CFG", {})
        self.segmentation_cfg = segmentation_cfg
        self.num_seg_classes = segmentation_cfg.get("NUM_SEG_CLASSES", None)
        self.use_only_samples_with_seg_labels = self.segmentation_cfg.get("USE_ONLY_SAMPLES_WITH_SEG_LABELS", False)
        self.load_seg = self.segmentation_cfg.get("LOAD_SEG", False)
        self.evaluation_list = dataset_cfg.get('EVALUATION_LIST', [])
        self.mix3d_cfg = dataset_cfg.get("MIX3D", None)
        self.filter_box_by_num_points = dataset_cfg.get("FILTER_BOX_BY_NUM_POINTS", None)

        self.max_num_points = self.dataset_cfg.get("MAX_NUM_POINTS", 200000)

        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        
        self.num_point_features = dataset_cfg.get("NUM_POINT_FEATURES", 0)
        self.drop_points_by_lidar_index = dataset_cfg.get("DROP_POINTS_BY_LIDAR_INDEX", None)
        if self.dataset_cfg.get("POINT_FEATURE_ENCODING", None) is not None:
            #if self.dataset_cfg.get("POINT_CLOUD_RANGE", None) is not None:
            #self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
            self.point_feature_encoder = PointFeatureEncoder(
                self.dataset_cfg.POINT_FEATURE_ENCODING,
            )
            self.num_point_features = self.point_feature_encoder.num_point_features
        else:
            self.point_feature_encoder = None

        self.point_cloud_range = self.dataset_cfg.get("POINT_CLOUD_RANGE", None)
        if isinstance(self.point_cloud_range, list):
            self.point_cloud_range = np.array(self.point_cloud_range, dtype=np.float32)
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.box_classes, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.num_point_features
        )

        self.num_point_features += self.data_processor.num_extra_point_features

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None
        
        self.runtime_cfg = dict(
            point_cloud_range=self.point_cloud_range,
            num_point_features=self.num_point_features,
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            box_classes=self.box_classes,
            num_seg_classes=self.num_seg_classes,
        )
        if self.logger is not None:
            self.logger.info(f"{self.__class__} Dataset in {self.mode} mode.")

    def fade_augmentor(self):
        self.data_augmentor.fade()

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        print(f'getting state {d}')
        del d['logger']
        return d

    def __setstate__(self, d):
        print(f'setting state {d}')
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, box_classes, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            box_classes:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_box_attr' in data_dict['object_wise'], 'gt_box_attr should be provided for training'
            gt_boxes_mask = np.array([(n in self.box_classes) or (len(n) == 0) for n in data_dict['object_wise']['gt_box_cls_label']], dtype=np.bool_)

            data_dict['object_wise'] = common_utils.filter_dict(
                                           data_dict['object_wise'], gt_boxes_mask
                                       )
            data_dict = self.data_augmentor.forward(data_dict)

        if data_dict['object_wise'].get('gt_box_attr', None) is not None:
            gt_box_attr = data_dict['object_wise']['gt_box_attr']
            gt_box_cls_label = data_dict['object_wise']['gt_box_cls_label']
            if gt_box_cls_label.dtype != np.int32:
                gt_box_cls_label = np.array([self.box_classes.index(n) + 1 if len(n) > 0 else 0 for n in gt_box_cls_label ], dtype=np.int32)
            #if 'gt_box_track_label' in data_dict['object_wise']:
            #    data_dict['object_wise'].pop('gt_box_track_label')
            data_dict['object_wise']['gt_box_cls_label'] = gt_box_cls_label
            gt_boxes = np.concatenate((gt_box_attr, gt_box_cls_label.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['object_wise']['gt_boxes'] = gt_boxes

        if (data_dict['point_wise'].get('point_feat', None) is not None) and (self.point_feature_encoder is not None):
            data_dict['point_wise'] = self.point_feature_encoder.forward(data_dict['point_wise'])

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False, num_mix3d_samples=1):

        data_dict = defaultdict(lambda: defaultdict(list))
        for cur_sample in batch_list:
            for key0, val0 in cur_sample.items():
                for key, val in val0.items():
                    data_dict[key0][key].append(val)
        batch_size = len(batch_list)
        ret = {}
        
        for key0, val0 in data_dict.items():
            for key, val in val0.items():
                try:
                    if not isinstance(val, list):
                        ret[key] = val
                    elif key in [
                               'voxel_point_xyz', 'voxel_num_points', 'voxel_point_feat',
                               'voxel_spherical_h', 'voxel_spherical_w',
                               'point_sweep', 'voxel_sweep', 'point_box_id', 'in_any_box', 'point_velo',
                               'point_feat', 'sinw', 'spherical_h', 'spherical_w', 'point_template_xyz',
                               'segmentation_label', 'voxel_segmentation_label', 'is_foreground',
                               'point_ground_mask', 'instance_label',
                               'voxel_is_foreground', 'point_sweep', 'point_polar_angle', 'point_azimuth', 'point_embedding',
                               'point_rimage_w', 'point_rimage_h', 'point_curvature', 'curvy', 'point_segment_id', 'point_in_large_segment',
                               'plane_eigvecs', 'plane_eigvals', 'plane_fitness', 'plane_l1_proj_min', 'plane_l1_proj_max', 'plane_degree',
                               'plane_mean_error', 'point_plane_id', 'plane_normal'
                              ]:
                        ret[key] = np.concatenate(val, axis=0)
                    elif key in ['point_xyz', 'voxel_coords', 'plane_xyz']:
                        coors = []

                        for i, coor in enumerate(val):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(coor_pad)
                        if key == 'point_xyz':
                            key = 'point_bxyz'
                        if key == 'plane_xyz':
                            key = 'plane_bxyz'
                        ret[key] = np.concatenate(coors, axis=0)
                    elif key in ['gt_boxes', 'gt_box_attr', 'gt_box_cls_label', 'difficulty', 'num_points_in_gt', 'augmented', 'obj_sweep', 'gt_box_corners_3d', 'gt_box_velo']:
                        if key in ['gt_box_cls_label', 'difficulty', 'num_points_in_gt', 'obj_sweep']:
                            val = [v.reshape(-1, 1) for v in val]
                            dtype = np.int32
                        elif key in ['augmented']:
                            val = [v.reshape(-1, 1) for v in val]
                            dtype = bool
                        else:
                            dtype = np.float32
                        max_gt = max([len(x) for x in val])
                        batch_box_data = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=dtype)
                        for k in range(batch_size):
                            batch_box_data[k, :val[k].__len__(), :] = val[k]
                        ret[key] = batch_box_data
                    elif key in ['gt_boxes2d']:
                        max_boxes = 0
                        max_boxes = max([len(x) for x in val])
                        batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                        for k in range(batch_size):
                            if val[k].size > 0:
                                batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                        ret[key] = batch_boxes2d
                    elif key in ["images", "depth_maps"]:
                        # Get largest image size (H, W)
                        max_h = 0
                        max_w = 0
                        for image in val:
                            max_h = max(max_h, image.shape[0])
                            max_w = max(max_w, image.shape[1])

                        # Change size of images
                        images = []
                        for image in val:
                            pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                            pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                            pad_width = (pad_h, pad_w)
                            # Pad with nan, to be replaced later in the pipeline.
                            pad_value = np.nan

                            if key == "images":
                                pad_width = (pad_h, pad_w, (0, 0))
                            elif key == "depth_maps":
                                pad_width = (pad_h, pad_w)

                            image_pad = np.pad(image,
                                               pad_width=pad_width,
                                               mode='constant',
                                               constant_values=pad_value)

                            images.append(image_pad)
                        ret[key] = np.stack(images, axis=0)
                    elif key in ['num_points_in_box', 'obj_ids',
                                 'voxel_instance_label']:
                        if key == 'obj_ids':
                            ret['obj_ids'] = val
                        continue
                    elif key in ['template_xyz', 'template_embedding']:
                        ret[key] = np.array(val[0], dtype=np.float32)
                    else:
                        ret[key] = np.stack(val, axis=0)
                except:
                    print('Error in collate_batch: key=%s' % key)
                    raise TypeError

        ret['batch_size'] = batch_size
        return ret
