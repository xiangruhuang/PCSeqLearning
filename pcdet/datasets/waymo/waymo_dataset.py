# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from sklearn.neighbors import NearestNeighbors as NN

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, polar_utils
from ..dataset import DatasetTemplate
from pcdet.utils.box_utils import boxes_to_corners_3d

class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        
        if dataset_cfg.get('ONE_SEQUENCE', None) is not None:
            _seq_id = self.dataset_cfg.ONE_SEQUENCE+'.tfrecord'
            if _seq_id not in self.sample_sequence_list:
                raise ValueError(f"{_seq_id} not in Sequence List")
            self.sample_sequence_list = [_seq_id]

        if dataset_cfg.get("EXTERNAL_SEQUENCE_LIST", None) is not None:
            with open(dataset_cfg.EXTERNAL_SEQUENCE_LIST, 'r') as fin:
                sequence_list = [l.strip()+'.tfrecord' for l in fin.readlines()]
            sample_list = []
            for s in sequence_list:
                if s in self.sample_sequence_list:
                    sample_list.append(s)
                else:
                    raise ValueError(f"{s} not in Sequence List")
            self.sample_sequence_list = sample_list

        self.num_sweeps = self.dataset_cfg.get('NUM_SWEEPS', 1)
        self.sweep_dir = self.dataset_cfg.get("SWEEP_DIR", -1)
        self.max_num_points *= self.num_sweeps
        self._merge_all_iters_to_one_epoch = dataset_cfg.get("MERGE_ALL_ITERS_TO_ONE_EPOCH", False)
        self.more_cls5 = self.segmentation_cfg.get('MORE_CLS5', False)
        self.use_spherical_resampling = self.dataset_cfg.get("SPHERICAL_RESAMPLING", False)
        self.ignore_index = dataset_cfg.get("IGNORE_INDEX", [0])
        self.with_time_feat = dataset_cfg.get("WITH_TIME_FEAT", False)
        self.extend_offsets = dataset_cfg.get("EXTEND_OFFSETS", [])
        self.sync_moving_points = dataset_cfg.get("SYNC_MOVING_POINTS", False)
        self.remove_ground = dataset_cfg.get("REMOVE_GROUND", False)

        self.drop_points_by_seg_len = dataset_cfg.get("DROP_POINTS_BY_SEG_LEN", False)
        filter_foreground = dataset_cfg.get("FILTER_FOREGROUND", None)
        self.filter_foreground = (filter_foreground is not None)
        if filter_foreground is not None:
            self.pred_label_path = filter_foreground["PRED_LABEL_PATH"]
            self.filter_class = filter_foreground["FILTER_CLASS"]

        self.infos = []
        self.include_waymo_data(self.mode)
        if self.num_sweeps > 1:
            logger.info(f"Sequence Dataset: {self.num_sweeps} sweeps")

        if len(self.extend_offsets) > 0:
            new_infos = []
            for info in self.infos:
                g_sample_idx = info['point_cloud']['sample_idx']
                sequence_id = info['point_cloud']['lidar_sequence']
                for offset in self.extend_offsets:
                    if (sequence_id, g_sample_idx-offset) in self.info_pool:
                        new_infos.append(self.info_pool[(sequence_id, g_sample_idx-offset)])
                    if (sequence_id, g_sample_idx-offset) in self.info_pool:
                        new_infos.append(self.info_pool[(sequence_id, g_sample_idx-offset)])
                new_infos.append(info)
            self.infos = new_infos

        if 'REPEAT' in dataset_cfg and self.training:
            repeat = dataset_cfg.get("REPEAT", 1)
            new_infos = []
            for i in range(repeat):
                new_infos += self.infos
            self.infos = new_infos
            logger.info(f"Repeating data by {repeat} times")
            
        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        if self.logger is not None:
            self.logger.info(f"{self.__class__} Dataset switched to {self.mode} mode, split={self.split}")
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_waymo_data(self.mode)

    def spherical_resampling(self, point_wise_dict, config={}):
        max_h = 64
        max_w = 2650
        point_xyz = point_wise_dict['point_xyz']
        point_feat = point_wise_dict['point_feat']
        point_rimage_h = point_wise_dict.pop('point_rimage_h')
        offset = 0
        new_point_wise_dict = dict(
            point_xyz = [],
            point_feat = [],
        )
        for h in range(max_h):
            mask_h = np.where(point_rimage_h == h)[0]
            num_points = mask_h.shape[0]
            if num_points == 0:
                continue
            point_xyz_h = point_xyz[mask_h]
            point_feat_h = point_feat[mask_h]
            new_point_wise_dict['point_xyz'].append(point_xyz_h)
            new_point_wise_dict['point_feat'].append(point_feat_h)
            if num_points < 10:
                continue
            r, polar, azimuth = polar_utils.cartesian2spherical_np(point_xyz_h)
            prange = np.linalg.norm(point_xyz_h, ord=2, axis=-1)
            tree = NN(n_neighbors=10).fit(point_xyz_h)
            dists, e1 = tree.kneighbors(point_xyz_h)
            e0 = np.arange(num_points)[:, np.newaxis]
            azimuth_diff = azimuth[e0] - azimuth[e1]
            azimuth_diff[azimuth_diff < 1e-6] = 1e10
            nn_index = azimuth_diff.argmin(axis=-1)
            e0 = e0[:, 0]
            dists = dists[(e0, nn_index)]
            e1 = e1[(e0, nn_index)]
            
            mask = dists < 0.3
            e0, e1, dists = e0[mask], e1[mask], dists[mask]

            num_samples_per_edge = np.ceil((dists+1e-6) / 0.1) + 1
            max_sample_per_edge = int(num_samples_per_edge.max())
        
            for sample_idx in range(max_sample_per_edge):
                edge_mask = sample_idx <= num_samples_per_edge - 1
                ratio = (sample_idx / (num_samples_per_edge-1))
                edge_mask = edge_mask & (ratio > 1e-6) & (ratio < 1 - 1e-6)
                if edge_mask.any():
                    ratio = ratio[edge_mask, np.newaxis]
                    new_xyz = point_xyz_h[e0[edge_mask]] * ratio + point_xyz_h[e1[edge_mask]] * (1.0-ratio)
                    new_feat = point_feat_h[e0[edge_mask]] * ratio + point_feat_h[e1[edge_mask]] * (1.0-ratio)
                    new_point_wise_dict['point_xyz'].append(new_xyz)
                    new_point_wise_dict['point_feat'].append(new_feat)

        for key in new_point_wise_dict.keys():
            new_point_wise_dict[key] = np.concatenate(new_point_wise_dict[key], axis=0)
            if new_point_wise_dict[key].dtype == np.float64:
                new_point_wise_dict[key] = new_point_wise_dict[key].astype(np.float32)
        
        tree = NN(n_neighbors=1).fit(point_xyz)
        dists, indices = tree.kneighbors(new_point_wise_dict['point_xyz'])
        indices = indices[:, 0]

        for key in point_wise_dict.keys():
            if key not in new_point_wise_dict:
                new_point_wise_dict[key] = point_wise_dict[key][indices]

        return new_point_wise_dict

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        if self.mode == 'train':
            sampled_sequence_offset = self.dataset_cfg.get("SAMPLED_SEQUENCE_OFFSET_TRAIN", 0)
        else:
            sampled_sequence_offset = self.dataset_cfg.get("SAMPLED_SEQUENCE_OFFSET_TEST", 0)

        for k in range(sampled_sequence_offset, len(self.sample_sequence_list), self.dataset_cfg.SAMPLED_SEQUENCE_INTERVAL[mode]):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))
        
        self.info_pool = {}
        for index, info in enumerate(self.infos):
            pc_info = info['point_cloud']
            sequence_id = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            self.info_pool[(sequence_id, sample_idx)] = info
        
        sample_idx_range = self.dataset_cfg.get("SAMPLE_IDX_RANGE", None)
        if sample_idx_range is not None:
            new_infos = []
            min_idx, max_idx = sample_idx_range[mode]
            for info in waymo_infos:
                sample_idx = info['point_cloud']['sample_idx']
                if min_idx <= sample_idx < max_idx:
                    new_infos.append(info)
            self.logger.info(f"Sample Index Filtering [{min_idx}, {max_idx}) for Waymo Dataset: {len(waymo_infos)} -> {len(new_infos)}")
            self.infos = new_infos

        if self.use_only_samples_with_seg_labels:
            new_infos = [info for info in self.infos if info['annos'].get('seg_label_path', None) is not None]
            new_infos = [info for info in new_infos if '_propseg.npy' not in info['annos'].get('seg_label_path', None)]
            new_infos = [info for info in new_infos if (info['point_cloud']['sample_idx'] >= self.num_sweeps - 1)]
            self.logger.info(f'Dropping samples without segmentation labels {len(self.infos)} -> {len(new_infos)}')
            self.infos = new_infos
        
        if self.more_cls5 and self.training:
            with open('../data/waymo/cls5.txt', 'r') as fin:
                frame_ids = [line.strip() for line in fin.readlines()]
            new_infos = [info for info in self.infos if info['frame_id'] in frame_ids]

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
        
        if self.more_cls5 and self.training:
            self.logger.info(f'repeating {len(new_infos)} scenes for cls 5')
            self.infos += new_infos

    def load_data_to_shared_memory(self):
        self.logger.info(f'Loading training data to shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                points = self.get_lidar(sequence_name, sample_idx)
                common_utils.sa_create(f"shm://{sa_key}", points)
            
            sa_key = f'{sequence_name}___seglabel___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                seg_labels = self.get_seg_label(sequence_name, sample_idx)
                common_utils.sa_create(f"shm://{sa_key}", seg_labels)

        dist.barrier()
        self.logger.info('Training data has been saved to shared memory')

    def clean_shared_memory(self):
        self.logger.info(f'Clean training data from shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")
            
            sa_key = f'{sequence_name}___seglabel___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('Training data has been deleted from shared memory')

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 8): [x, y, z, intensity, elongation, range, rimage_w, rimage_h]

        points_all = point_features[:, [0,1,2,3,4,5,6,7]] # [x, y, z, intensity, elongation, range, rimage_w, rimage_h]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        points_all[:, 5] /= 75.
        points_all[:, 7] *= 64
        points_all[:, 6] *= 2650
        return points_all
    
    def get_ground_mask(self, sequence_name, sample_idx):
        height_file = self.dataset_cfg.POINT_HEIGHT_PATH / sequence_name / ('%04d_point_height.npy' % sample_idx)
        #mask_file = self.data_path / sequence_name / ('%04d_point_height.npy' % sample_idx)
        ground_mask = np.load(height_file) < self.dataset_cfg.TRUNCATE_GROUND_HEIGHT

        return ground_mask

    def get_seg_label(self, sequence_name, sample_idx):
        seg_file = str(self.data_path / sequence_name / ('%04d_seg.npy' % sample_idx))
        if not os.path.exists(seg_file):
            seg_file = seg_file.replace('_seg.npy', '_propseg.npy')
        seg_labels = np.load(seg_file)  # (N, 2): [instance_label, segmentation_label]

        return seg_labels

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def load_data(self, info):
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        
        if self.use_shared_memory: # and ((self.shared_memory_file_limit < 0) or (index < self.shared_memory_file_limit)):
            assert False
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
            if self.use_only_samples_with_seg_labels:
                sa_key = f'{sequence_name}___seglabel___{sample_idx}'
                seg_labels = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)
            if self.load_seg:
                seg_labels = self.get_seg_label(sequence_name, sample_idx)
            
        points = points.astype(np.float32)
        point_wise_dict = dict(
            point_xyz=points[:, :3],
            point_feat=points[:, 3:-2],
            point_rimage_h=points[:,-1].astype(np.int64),
        )
        if self.load_seg:
            point_wise_dict['segmentation_label'] = seg_labels[:, 1]
            point_wise_dict['instance_label'] = seg_labels[:, 0]

        if self.drop_points_by_lidar_index is not None:
            num_points_of_each_lidar = info['num_points_of_each_lidar']
            offset = 0
            lidar_point_index_list = []

            for i, num_points in enumerate(num_points_of_each_lidar):
                if i not in self.drop_points_by_lidar_index:
                    lidar_point_index = np.arange(offset, offset+num_points)
                    lidar_point_index_list.append(lidar_point_index)
            lidar_point_indices = np.concatenate(lidar_point_index_list, axis=0)
            point_wise_dict = common_utils.filter_dict(point_wise_dict, lidar_point_indices)
        
        if self.drop_points_by_seg_len:
            seg_len = np.arange(point_wise_dict['segmentation_label'].shape[0])
            point_wise_dict = common_utils.filter_dict(point_wise_dict, seg_len)
        
        scene_wise_dict = dict(
            frame_id=info['frame_id'],
            pose=info['pose'].reshape(4, 4),
        )
        
        if 'top_lidar_pose' in info['metadata']:
            top_lidar_pose = info['metadata']['top_lidar_pose'][4].reshape(4, 4)
            top_lidar_origin = top_lidar_pose[:3, 3]
            scene_wise_dict['top_lidar_origin'] = top_lidar_origin

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')
            annos = common_utils.drop_info_with_name(annos, name='Sign')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]
                annos['obj_ids'] = annos['obj_ids'][mask]
                if 'transform' in annos:
                    annos['transform'] = annos['transform'][mask]

            if self.training and self.dataset_cfg.get("FILTER_BOX_BY_NUM_POINTS", None) is not None:
                filter_box_by_num_points = self.dataset_cfg.get("FILTER_BOX_BY_NUM_POINTS", None)
                min_num_points = filter_box_by_num_points.get("MIN_NUM_POINTS", 0)
                mask = (annos['num_points_in_gt'] >= min_num_points)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                removed_boxes = gt_boxes_lidar[~mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]
                annos['obj_ids'] = annos['obj_ids'][mask]
                if 'transform' in annos:
                    annos['transform'] = annos['transform'][mask]

            if self.training and (self.dataset_cfg.get("RANDOM_DROP_BOX_RATIO", None) is not None):
                ratio = self.dataset_cfg.get("RANDOM_DROP_BOX_RATIO", None)
                rands = np.array([(sum([ord(c) for c in obj_id]) % 100) for obj_id in annos['obj_ids']]).astype(np.int32)
                mask = (rands / 100.0) >= ratio
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]
                annos['obj_ids'] = annos['obj_ids'][mask]
                if 'transform' in annos:
                    annos['transform'] = annos['transform'][mask]
                
            point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(point_wise_dict['point_xyz'][:, :3],
                                                                    gt_boxes_lidar).sum(0)
            point_wise_dict['is_foreground'] = point_masks > 0
            object_wise_dict = dict(
                gt_box_cls_label=annos['name'].astype(str),
                gt_box_attr=gt_boxes_lidar,
                augmented=np.zeros(annos['name'].shape[0], dtype=bool),
                obj_ids=annos['obj_ids'],
                num_points_in_gt=annos['num_points_in_gt'],
            )
        else:
            object_wise_dict = {}

        if self.use_spherical_resampling:
            point_wise_dict = self.spherical_resampling(point_wise_dict)
        else:
            point_wise_dict.pop('point_rimage_h')

        if self.filter_foreground:
            pred_label = np.load(f'{self.pred_label_path}/{sequence_name}/{sample_idx:03d}_pred.npy').astype(np.int64)
            gt_label = point_wise_dict['segmentation_label'].astype(np.int64)
            mask = np.ones(pred_label.shape[0], dtype=bool)
            for cls in self.filter_class:
                mask[pred_label == cls] = 0
            gt_mask = np.ones(gt_label.shape[0], dtype=bool)
            for cls in self.filter_class:
                gt_mask[gt_label == cls] = 0
            point_wise_dict = common_utils.filter_dict(point_wise_dict, mask)
        if self.remove_ground:
            ground_mask = self.get_ground_mask(sequence_name, sample_idx)
            #point_wise_dict['point_ground_mask'] = ground_mask
            point_wise_dict = common_utils.filter_dict(point_wise_dict, ~ground_mask)
        
        input_dict=dict(
            point_wise=point_wise_dict,
            scene_wise=scene_wise_dict,
            object_wise=object_wise_dict,
        )

        return input_dict

    def __getitem__(self, index, sweeping=False, mix3d=False):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        input_dict = self.load_data(info)
        cur_sample_idx = info['point_cloud']['sample_idx']
        lidar_sequence = info['point_cloud']['lidar_sequence']
        data_dicts = [input_dict]

        if self.num_sweeps > 1:
            transform = {}
            global_T = []
            obj_id_to_box = {}
            box_corners = boxes_to_corners_3d(input_dict['object_wise']['gt_box_attr'])
            for obj_idx, obj_id in enumerate(input_dict['object_wise']['obj_ids']):
                transform[obj_id] = np.eye(4).astype(np.float64)
                global_T.append(np.eye(4).astype(np.float64))
                obj_id_to_box[obj_id] = obj_idx
            T0_inv = np.linalg.inv(input_dict['scene_wise']['pose'])
            if len(global_T) > 0:
                input_dict['object_wise']['global_T'] = np.stack(global_T, axis=0)
            else:
                input_dict['object_wise']['global_T'] = np.zeros((0, 4, 4), dtype=np.float32)

            for cur_index in range(cur_sample_idx + self.sweep_dir, cur_sample_idx + self.sweep_dir * self.num_sweeps, self.sweep_dir):
                if (lidar_sequence, cur_index) not in self.info_pool:
                    continue
                prev_info = self.info_pool[(lidar_sequence, cur_index)]
                data_dict = self.load_data(prev_info)
                
                T_this = T0_inv @ data_dict['scene_wise']['pose']
                corners = boxes_to_corners_3d(data_dict['object_wise']['gt_box_attr'])
                corners = (corners @ T_this[:3, :3].T + T_this[:3, 3]).reshape(-1, 8, 3)
                object_wise = data_dict['object_wise']
                global_T = []
                for obj_idx, obj_id in enumerate(object_wise['obj_ids']):
                    if obj_id not in obj_id_to_box:
                        # no valid transformation from frame (cur_index+1) to frame (cur_sample_idx)
                        # OR no valid transformation from frame (cur_index) to frame (cur_index+1)
                        # points should be removed
                        T_t = np.eye(4).astype(np.float64)
                        T_t[:3, 3] = 1e4
                        global_T.append(T_t)
                    else:
                        # fit ICP
                        last_box_corner = corners[obj_idx]
                        box_corner = box_corners[obj_id_to_box[obj_id]]
                        b0 = box_corner.mean(0)
                        l0 = last_box_corner.mean(0)
                        q = box_corner - b0
                        p = last_box_corner - l0
                        M = p.T @ q
                        U, S, VT = np.linalg.svd(M)
                        V = VT.T
                        sign = np.linalg.det(V @ U.T)
                        R = V @ np.diag([1, 1, sign]) @ U.T
                        t = b0 - R @ l0
                        T_t = np.eye(4).astype(np.float64)
                        T_t[:3, :3] = R
                        T_t[:3, 3] = t
                        global_T.append(T_t)
                if len(global_T) > 0:
                    global_T = np.stack(global_T, axis=0)
                else:
                    global_T = np.zeros((0, 4, 4), dtype=np.float32)
                data_dict['object_wise']['global_T'] = global_T
                
                if self.sweep_dir == -1:
                    data_dicts = [data_dict] + data_dicts
                else:
                    data_dicts = data_dicts + [data_dict]
        T0 = data_dicts[-1]['scene_wise']['pose'].reshape(4, 4)
        T0_inv = np.linalg.inv(T0)
        points_list = []
        ## transform all points into this coordinate system
        max_num_objects = 0
        num_sweeps = len(data_dicts)
        #print(f"Loading Sequence {input_dict['scene_wise']['frame_id']}, get {num_sweeps} sweeps")
        for sweep, data_dict in enumerate(data_dicts):
            T1 = data_dict['scene_wise']['pose'].reshape(4, 4)
            T = T0_inv @ T1
            
            # compute box-point association first
            if self.sync_moving_points and ('global_T' in data_dict['object_wise']):
                boxes3d = data_dict['object_wise']['gt_box_attr']
                point_xyz = data_dict['point_wise']['point_xyz']
                if boxes3d.shape[0] > 0:
                    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(point_xyz, boxes3d) # [nbox, npoint]
                    in_any_box = point_masks.any(0) # [npoint]
                    point_box_id = point_masks.astype(np.int32).argmax(0)
            
            # apply transformation
            points = data_dict['point_wise']['point_xyz']
            points[:, :3] = points[:, :3] @ T[:3, :3].T + T[:3, 3]
            data_dict['point_wise']['point_xyz'] = points
            
            if self.sync_moving_points and ('global_T' in data_dict['object_wise']):
                boxes3d = data_dict['object_wise']['gt_box_attr']
                if boxes3d.shape[0] > 0:
                    box_global_T = data_dict['object_wise']['global_T']
                    point_xyz = data_dict['point_wise']['point_xyz']
                    
                    point_trans = np.zeros((point_xyz.shape[0], 4, 4), dtype=np.float64)
                    point_trans[in_any_box] = box_global_T[point_box_id[in_any_box]]
                    point_trans[~in_any_box] = np.eye(4).astype(np.float64)
                    
                    transformed_point_xyz = (point_trans[:, :3, :3] @ point_xyz[:, :, np.newaxis]).squeeze(-1) + point_trans[:, :3, 3]
                    data_dict['point_wise']['point_xyz'] = transformed_point_xyz
                    #data_dict['point_wise']['point_velo'] = transformed_point_xyz - point_xyz
                    valid_mask = (transformed_point_xyz < 1e3).all(-1)
                    data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], valid_mask)
                    
                data_dict['object_wise'].pop('global_T')
            elif ('global_T' in data_dict['object_wise']):
                data_dict['object_wise'].pop('global_T')

            # attach sweep index as the first channel similar to batch index
            num_points = data_dict['point_wise']['point_xyz'].shape[0]
            _sample_idx = int(data_dict['scene_wise']['frame_id'][-3:])
            point_sweep = np.zeros((num_points, 1), dtype=np.int32) + _sample_idx
            data_dict['point_wise']['point_sweep'] = point_sweep
            if (self.num_sweeps > 1) and (self.with_time_feat):
                data_dict['point_wise']['point_feat'] = np.concatenate(
                        [point_sweep.reshape(-1, 1) / (num_sweeps-1),
                         data_dict['point_wise']['point_feat']],
                        axis=-1
                    ).astype(np.float32)

            if 'top_lidar_origin' in data_dict['scene_wise']:
                origin = data_dict['scene_wise'].pop('top_lidar_origin')
                origin = origin @ T[:3, :3].T + T[:3, 3]
                data_dict['scene_wise']['top_lidar_origin'] = origin

            boxes = data_dict['object_wise']['gt_box_attr']
            box_corners = boxes_to_corners_3d(boxes)
            box_corners = (box_corners @ T[:3, :3].T + T[:3, 3]).reshape(-1, 24)
            boxes[:, :3] = boxes[:, :3] @ T[:3, :3].T + T[:3, 3]

            # compute heading
            theta = boxes[:, -1]
            heading = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1) # [B, 3]
            heading = heading @ T[:3, :3].T
            heading[:, :2] = heading[:, :2] / np.linalg.norm(heading[:, :2], ord=2, axis=-1)[:, np.newaxis]
            theta = np.arctan2(heading[:, 1], heading[:, 0])
            boxes[:, -1] = theta

            data_dict['object_wise']['gt_box_attr'] = boxes
            data_dict['object_wise']['gt_box_corners_3d'] = box_corners

            # insert sweep id
            num_objects = data_dict['object_wise']['gt_box_attr'].shape[0]
            if num_objects > max_num_objects:
                max_num_objects = num_objects
        
        input_dict = dict(
            point_wise=dict(common_utils.concat_dicts([dd['point_wise'] for dd in data_dicts])),
            object_wise=dict(common_utils.stack_dicts([dd['object_wise'] for dd in data_dicts],
                                                 pad_to_size=max_num_objects)),
            scene_wise=dict(common_utils.stack_dicts([dd['scene_wise'] for dd in data_dicts])),
        )
        for key, val in input_dict['object_wise'].items():
            input_dict['object_wise'][key] = val.reshape(num_sweeps*max_num_objects, *val.shape[2:])
        #if self.training:
        #    input_dict['object_wise']['gt_box_track_label'] = np.unique(input_dict['object_wise']['obj_ids'], return_inverse=True)[1]
        #    print(input_dict['object_wise']['gt_box_track_label'])
            
        data_dict = self.prepare_data(data_dict=input_dict)
        if (self.mix3d_cfg is not None) and (not mix3d) and self.training:
            prob = self.mix3d_cfg.get("PROB", 1.0)
            if np.random.rand() < prob:
                num_samples = self.__len__()
                rand_idx = np.random.randint(0, num_samples)
                data_dict2 = self.__getitem__(rand_idx, mix3d=True)
                data_dict['point_wise'] = common_utils.concat_dicts([data_dict['point_wise'], data_dict2['point_wise']])
                data_dict['object_wise'] = common_utils.concat_dicts([data_dict['object_wise'], data_dict2['object_wise']])

        data_dict['scene_wise']['num_sweeps'] = num_sweeps

        return data_dict

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def generate_single_sample_dict(cur_dict, output_path=None):
            frame_id = str(cur_dict['scene_wise']['frame_id'].reshape(-1)[-1])
            sequence_id, sample_idx = frame_id[:-4], int(frame_id[-3:])

            pred_dict = dict(
                object_wise=dict(),
                point_wise=dict(),
                scene_wise=dict(
                    frame_id=frame_id,
                    sequence_id=sequence_id,
                    sample_idx=sample_idx,
                ),
            )

            if 'pred_box_attr' in cur_dict['object_wise']:
                object_wise_dict = cur_dict['object_wise']

                pred_scores = object_wise_dict['pred_box_scores'].cpu().numpy()
                pred_boxes = object_wise_dict['pred_box_attr'].cpu().numpy()
                pred_labels = object_wise_dict['pred_box_cls_label'].cpu().numpy()
                
                if pred_scores.shape[0] > 0:
                    pred_dict['object_wise'].update(
                        dict(
                            box_scores=pred_scores,
                            box_attr=pred_boxes,
                            box_name=np.array(class_names)[pred_labels - 1],
                        ))

            if 'pred_segmentation_label' in cur_dict['point_wise']:
                point_wise_dict = cur_dict['point_wise']
                
                # propagate segmentation from predicted points to gt points
                path = f'../data/waymo/waymo_processed_data_v0_5_0/{sequence_id}/{sample_idx:04d}_seg.npy'
                if not os.path.exists(path):
                    path = f'../data/waymo/waymo_processed_data_v0_5_0/{sequence_id}/{sample_idx:04d}_propseg.npy'
                segmentation_label = np.load(path)[:, 1]

                point_xyz = np.load(f'../data/waymo/waymo_processed_data_v0_5_0/{sequence_id}/{sample_idx:04d}.npy')
                point_xyz = point_xyz[:segmentation_label.shape[0], :3]
                tree = NN(n_neighbors=1).fit(point_wise_dict['point_xyz'].detach().cpu().numpy())
                dists, indices = tree.kneighbors(point_xyz)
                pred_segmentation_label = point_wise_dict['pred_segmentation_label'].detach().cpu().numpy()[indices[:, 0]]
                if self.filter_foreground:
                    pred_label = np.load(f'{self.pred_label_path}/{sequence_id}/{sample_idx:03d}_pred.npy').astype(np.int64)
                    mask = np.zeros(pred_label.shape[0], dtype=bool)
                    for cls in self.filter_class:
                        mask[pred_label == cls] = 1
                    pred_segmentation_label[mask] = pred_label[mask]
                    
                for ignore_index in self.ignore_index:
                    pred_segmentation_label[segmentation_label == ignore_index] = ignore_index

                point_wise_dict['pred_segmentation_label'] = torch.from_numpy(pred_segmentation_label)
                
                # compute statistics
                ups = torch.zeros(23, dtype=torch.long)
                downs = torch.zeros(23, dtype=torch.long)
                for i in range(23):
                    ups[i] = ((segmentation_label == i) & (pred_segmentation_label == i)).sum()
                    downs[i] = ((segmentation_label == i) | (pred_segmentation_label == i)).sum()

                if output_path is not None:
                    pred_labels = point_wise_dict['pred_segmentation_label'].detach().to(torch.uint8).cpu()
                    os.makedirs(output_path / sequence_id, exist_ok=True)
                    path = str(output_path / sequence_id / f"{sample_idx:03d}_pred.npy")
                    np.save(path, pred_labels)

                pred_dict['scene_wise'].update(
                    ups=ups.detach().cpu(),
                    downs=downs.detach().cpu(),
                )

            return pred_dict

        annos = []

        if (output_path is not None) and (not os.path.exists(output_path)):
            os.makedirs(output_path, exist_ok=True)
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict, output_path=output_path)
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, pred_dicts, box_class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=box_class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        #eval_det_annos = copy.deepcopy(det_annos)
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        #if kwargs['eval_metric'] == 'kitti':
        #    ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        #elif kwargs['eval_metric'] == 'waymo':
        #    ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        #else:
        #    raise NotImplementedError

        #return ap_result_str, ap_dict
        
        if 'box' in self.evaluation_list:
            eval_gt_annos = []
            for i in range(self.__len__()):
                annos = self.infos[i]['annos']
                box_label = annos['name']
                box_difficulty = annos['difficulty']
                box_attr = annos['gt_boxes_lidar']
                box_npoints = annos['num_points_in_gt']
                #box_attr, box_label, box_difficulty, box_npoints = self.get_box3d(i)
                eval_gt_annos.append(
                    dict(
                        difficulty=box_difficulty,
                        num_points_in_gt=box_npoints,
                        name=box_label,
                        gt_boxes_lidar=box_attr
                    )
                )
            eval_det_annos = copy.deepcopy(pred_dicts)
            #eval_det_annos = translate_names(eval_det_annos)
            #eval_gt_annos = translate_names(eval_gt_annos)
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
            return ap_result_str, ap_dict
        elif 'seg' in self.evaluation_list:
            total_ups, total_downs = None, None
            for pred_dict in pred_dicts:
                ups, downs = pred_dict['scene_wise']['ups'], pred_dict['scene_wise']['downs']
                if total_ups is None:
                    total_ups = ups.clone()
                    total_downs = downs.clone()
                else:
                    total_ups += ups
                    total_downs += downs
            seg_result_str = '\n'
            iou_dict = {}
            ious = []
            for cls in range(total_ups.shape[0]):
                iou = total_ups[cls]/np.clip(total_downs[cls], 1, None)
                seg_result_str += f'IoU for class {cls} {iou:.4f} \n'
                iou_dict[f'IoU_{cls}'] = iou
                ious.append(iou)
            ious = np.array(ious).reshape(-1)[1:]
            iou_dict['mIoU'] = ious.mean()
            iou_dict['IoU_FG'] = total_ups[1:14].sum() / np.clip(total_downs[1:14].sum(), 1, None)
            iou_dict['IoU_BG'] = total_ups[14:].sum() / np.clip(total_downs[14:].sum(), 1, None)
            seg_result_str += f'mIoU={ious.mean():.4f} \n'
            seg_result_str += f"IoU_FG={iou_dict['IoU_FG']:.4f} \n"
            seg_result_str += f"IoU_BG={iou_dict['IoU_BG']:.4f} \n"
            return seg_result_str, iou_dict
        else:
            raise NotImplementedError

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):
        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            if k % 4 != 0 and len(names) > 0:
                mask = (names == 'Vehicle')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            if k % 2 != 0 and len(names) > 0:
                mask = (names == 'Pedestrian')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # it will be used if you choose to use shared memory for gt sampling
        stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        np.save(db_data_save_path, stacked_gt_points)
    
    def propagate_segmentation_labels(self, waymo_infos, save_path, num_workers=multiprocessing.cpu_count()):
        from functools import partial
        from . import waymo_utils
        print('---------------Propagating Segmentation Labels------------------------')

        propagate_single_sequence = partial(
            waymo_utils.propagate_segmentation_labels,
            waymo_infos=waymo_infos,
            save_path=save_path
        )

        sequence_ids = list(set([info['point_cloud']['lidar_sequence'] for info in waymo_infos]))

        #propagate_single_sequence(sequence_id = sequence_ids[1])
        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(propagate_single_sequence,
                                              sequence_ids),
                                       total=len(sequence_ids)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

def create_waymo_infos(dataset_cfg, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=min(16, multiprocessing.cpu_count())):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split = dataset_cfg.DATA_SPLIT['train']
    val_split = dataset_cfg.DATA_SPLIT['test']
    #train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split=train_split, sampled_interval=1,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
    )
    print('---------------Data preparation Done---------------')


def propagate_segmentation_labels(dataset_cfg, data_path, save_path,
                                  raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                                  workers=min(16, multiprocessing.cpu_count())):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split = dataset_cfg.DATA_SPLIT['train']
    val_split = dataset_cfg.DATA_SPLIT['test']
    #train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    waymo_infos_train = dataset.propagate_segmentation_labels(
        waymo_infos_train,
        save_path=save_path / processed_data_tag,
        num_workers=workers,
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    waymo_infos_val = dataset.propagate_segmentation_labels(
        waymo_infos_val,
        save_path=save_path / processed_data_tag,
        num_workers=workers,
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split=train_split, sampled_interval=1,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
    )
    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=args.processed_data_tag
        )
    
    if args.func == 'propagate_segmentation_labels':
        import yaml
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        propagate_segmentation_labels(
            dataset_cfg=dataset_cfg,
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=args.processed_data_tag
        )
