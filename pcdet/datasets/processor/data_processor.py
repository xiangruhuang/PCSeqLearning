from functools import partial

import numpy as np
import torch
from skimage import transform
from sklearn.neighbors import NearestNeighbors as NN
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from torch_cluster import fps, grid_cluster
from torch_scatter import scatter

from ...utils import box_utils, common_utils, polar_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from collections import defaultdict

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2
        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxel_points'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            tv_points = tv.from_numpy(np.ascontiguousarray(points))
            voxel_output = self._voxel_generator.point_to_voxel(tv_points)
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self._voxel_generator = None

        self.num_extra_point_features = 0
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            if cur_cfg.NAME == "attach_spherical_feature":
                self.num_extra_point_features += 3
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        points = data_dict['point_wise']['point_xyz']
        mask = common_utils.mask_points_by_range(points, self.point_cloud_range)
        data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], mask)

        if config.REMOVE_OUTSIDE_BOXES and self.training:
            gt_boxes = data_dict['object_wise']['gt_box_attr']
            mask = box_utils.mask_boxes_outside_range_numpy(
                gt_boxes, self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['object_wise'] = common_utils.filter_dict(data_dict['object_wise'], mask)

        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['point_wise']['point_xyz']
            shuffle_idx = np.random.permutation(points.shape[0])
            data_dict['point_wise'] = common_utils.filter_dict(
                                          data_dict['point_wise'],
                                          shuffle_idx
                                      )

        return data_dict
    
    def limit_num_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.limit_num_points, config=config)

        max_num_points = config["MAX_NUM_POINTS"]

        points = data_dict['point_wise']['point_xyz']
        if points.shape[0] > max_num_points and self.training:
            subsampler = config.get("SUBSAMPLE_METHOD", 'UNIFORM')
            if subsampler == 'UNIFORM':
                shuffle_idx = np.random.permutation(points.shape[0])[:max_num_points]
                data_dict['point_wise'] = common_utils.filter_dict(
                                              data_dict['point_wise'],
                                              shuffle_idx
                                          )
            elif subsampler == 'FPS':
                points = torch.from_numpy(points)
                ratio = max_num_points / points.shape[0]
                fps_index = fps(points, ratio=ratio)
                data_dict['point_wise'] = common_utils.filter_dict(
                                              data_dict['point_wise'],
                                              fps_index
                                          )
            elif subsampler == 'GRID':
                points = torch.from_numpy(points)
                grid_size = torch.tensor(config["GRID_SIZE"]).float()
                cluster = grid_cluster(points, grid_size)
                unique, inv = torch.unique(cluster, return_inverse=True)
                subindices = scatter(torch.arange(points.shape[0]), inv, dim=0, dim_size=unique.shape[0], reduce='max')
                data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], subindices.numpy())

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        point_cloud_range = np.array(config.POINT_CLOUD_RANGE)
        if data_dict is None:
            grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if not config.get('DRY', False):
            pointvecs = []
            num_points = data_dict['point_wise']['point_xyz'].shape[0]
            for key in ['point_xyz', 'point_feat']:
                pointvecs.append(data_dict['point_wise'][key])
            pointvecs = [p.reshape(num_points, -1) for p in pointvecs]
            dims = [p.shape[-1] for p in pointvecs]
            dtypes = [p.dtype for p in pointvecs]
            num_point_features = sum(dims)

            if self._voxel_generator is None:
                self._voxel_generator = VoxelGeneratorWrapper(
                    vsize_xyz=config.VOXEL_SIZE,
                    coors_range_xyz=point_cloud_range,
                    num_point_features=num_point_features,
                    max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                )

            pointvecs = np.concatenate(pointvecs, axis=-1)

            point_feat_dim = pointvecs.shape[1]
            voxel_output = self._voxel_generator.generate(pointvecs)
            voxels, coordinates, num_points_in_voxel = voxel_output
            voxel_wise = dict(
                voxel_coords = coordinates,
                voxel_num_points = num_points_in_voxel
            )
            offset = 0
            for key, dim, dtype in zip(['point_xyz', 'point_feat'], dims, dtypes):
                voxel_wise['voxel_'+key] = voxels[..., offset:offset+dim].astype(dtype)
                offset += dim
                
            data_dict['voxel_wise'] = voxel_wise

        return data_dict

    #def sample_points(self, data_dict=None, config=None):
    #    if data_dict is None:
    #        return partial(self.sample_points, config=config)

    #    num_points = config.NUM_POINTS[self.mode]
    #    if num_points == -1:
    #        return data_dict

    #    points = data_dict['point_wise']['points']
    #    if num_points < len(points):
    #        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
    #        pts_near_flag = pts_depth < 40.0
    #        far_idxs_choice = np.where(pts_near_flag == 0)[0]
    #        near_idxs = np.where(pts_near_flag == 1)[0]
    #        choice = []
    #        if num_points > len(far_idxs_choice):
    #            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
    #            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
    #                if len(far_idxs_choice) > 0 else near_idxs_choice
    #        else: 
    #            choice = np.arange(0, len(points), dtype=np.int32)
    #            choice = np.random.choice(choice, num_points, replace=False)
    #        np.random.shuffle(choice)
    #    else:
    #        choice = np.arange(0, len(points), dtype=np.int32)
    #        if num_points > len(points):
    #            extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
    #            choice = np.concatenate((choice, extra_choice), axis=0)
    #        np.random.shuffle(choice)
    #    data_dict['point_wise']['points'] = points[choice]
    #    return data_dict

    #def calculate_grid_size(self, data_dict=None, config=None):
    #    if data_dict is None:
    #        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
    #        self.grid_size = np.round(grid_size).astype(np.int64)
    #        self.voxel_size = config.VOXEL_SIZE
    #        return partial(self.calculate_grid_size, config=config)
    #    return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def propagate_box_label_to_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.propagate_box_label_to_points, config=config)
        points = data_dict['points'][:, :3]
        seg_label_map = config['SEG_LABEL_MAP']
        labels = np.array([seg_label_map[n] for n in data_dict['gt_names']]).astype(np.int64)
        boxes = np.copy(data_dict['gt_boxes'])
        boxes = boxes[:, :7]
        boxes[:, 3:6] *= 0.95
        seg_inst_labels = data_dict['seg_inst_labels']
        inst_labels = seg_inst_labels.max() + 1 + np.arange(boxes.shape[0])
        seg_cls_labels = data_dict['seg_cls_labels']

        mask = roiaware_pool3d_utils.points_in_boxes_cpu(points, boxes)
        in_box_points = mask.any(0)
        if in_box_points.any():
            box_indices = mask[:, in_box_points].argmax(0)
            seg_cls_labels[in_box_points] = labels[box_indices]
            data_dict['seg_cls_labels'] = seg_cls_labels
        
            seg_inst_labels[in_box_points] = inst_labels[box_indices]
            data_dict['seg_inst_labels'] = seg_inst_labels

        return data_dict

    def attach_spherical_feature(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.attach_spherical_feature, config=config)
        if (config is not None) and config.get("USE_LIDAR_TOP_ORIGIN", False):
            origin = data_dict['scene_wise']['top_lidar_origin']
        else:
            origin = np.zeros(3)
        xyz = data_dict['point_wise']['point_xyz'] - origin
        
        r, polar, azimuth = polar_utils.cartesian2spherical_np(xyz)
        #polar_feat[:, 0] = polar_feat[:, 0] / 78.
        azimuth_sin_cos = np.stack([np.sin(azimuth), np.cos(azimuth)], axis=-1).astype(np.float32)
        data_dict['point_wise']['point_feat'] = np.concatenate(
                                                [data_dict['point_wise']['point_feat'],
                                                 (polar.reshape(-1, 1)-1.276)/0.375,
                                                 azimuth_sin_cos.reshape(-1, 2)], axis=-1).astype(np.float32)
        data_dict['point_wise']['point_polar_angle'] = polar.reshape(-1, 1)
        data_dict['point_wise']['point_azimuth'] = azimuth.reshape(-1, 1)
        
        return data_dict
    
    def shift_to_top_lidar_origin(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shift_to_top_lidar_origin, config=config)
        points = data_dict['point_wise']['point_xyz']
        origin = data_dict['scene_wise']['top_lidar_origin']
        
        points[:, :3] = points[:, :3] - origin
        data_dict['point_wise']['point_xyz'] = points
        data_dict['scene_wise']['top_lidar_origin'] = np.zeros_like(origin)
        
        return data_dict

    def point_centering(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.point_centering, config=config)
        points = data_dict['point_wise']['points']
        
        origin = points[:, :3].mean(0)
        if config is not None and config.get("Z_SHIFT_MIN", False):
            origin[2] = points[:, 2].min()
        points[:, :3] = points[:, :3] - origin
        data_dict['point_wise']['points'] = points
        
        return data_dict
        
    def process_point_feature(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.process_point_feature, config=config)

        points = data_dict['point_wise']['point_feat']
        points = points[:, [0,1]]
        points[:, 0] = np.clip(points[:, 1], 0, 1)
        #points[:, [0, 1]] = points[:, [0, 1]]
        points[:, [0, 1]] = (points[:, [0, 1]] - [0.1382, 0.082]) / [0.1371, 0.1727]

        data_dict['point_wise']['point_feat'] = points
        return data_dict

    def sync_box_motion(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sync_box_motion, config=config)

        import ipdb; ipdb.set_trace()

        return data_dict

    
    def extract_ground_plane_classes(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.extract_ground_plane_classes, config=config)

        seg_cls_labels = data_dict['point_wise']['seg_cls_labels']
        mask = seg_cls_labels == -10
        classes = config['CLASSES']
        for cls in classes:
            mask = mask | (seg_cls_labels == cls)
        data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], mask)

        return data_dict

    def estimate_velocity(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.estimate_velocity, config=config)

        keep_mask = np.ones(data_dict['object_wise']['obj_ids'].shape[0], dtype=bool)
        obj_ids = data_dict['object_wise']['obj_ids']
        obj_ids = obj_ids
        if 'obj_sweep' not in data_dict['object_wise']:
            raise ValueError("Not in multi-frame setting")
        obj_sweeps = data_dict['object_wise']['obj_sweep']

        unique_obj_ids = np.unique(obj_ids)
        min_sweep = defaultdict(lambda : -1)
        for i, (obj_id, obj_sweep) in enumerate(zip(obj_ids, obj_sweeps)):
            if (min_sweep[obj_id] == -1) or (min_sweep[obj_id] > obj_sweep):
                min_sweep[obj_id] = obj_sweep
        
        for obj_id, val in min_sweep.items():
            if val == 0:
                # good trace
                pass
            else:
                # to be removed
                keep_mask[obj_ids == obj_id] = False
        data_dict['object_wise'] = common_utils.filter_dict(data_dict['object_wise'], keep_mask)
        data_dict['object_wise'].pop('obj_ids')

        return data_dict

    def build_spherical_graph(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_spherical_graph, config=config)

        #xyz = data_dict['point_wise']['point_xyz']
        #r, inclination, azimuth = polar_utils.cartesian2spherical(xyz)
        #
        #import ipdb; ipdb.set_trace()

        return data_dict

    def _merge_points_into_depth_frame(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.merge_points_into_depth_frame, config=config)

        max_h = config.get("MAX_H", 64)
        max_w = config.get("MAX_W", 2650)
        
        point_rimage_h = data_dict['point_wise']['point_rimage_h']
        point_rimage_w = data_dict['point_wise']['point_rimage_w']

        point_key = point_rimage_h * max_w + point_rimage_w

        _, indices = np.unique(point_key, return_index=True)

        data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], indices)

        return data_dict

    def lidar_line_segment(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.lidar_line_segment, config=config)

        import ipdb; ipdb.set_trace()
        data_dict = self._merge_points_into_depth_frame(data_dict, config)

        max_h = config.get("MAX_H", 64)
        max_w = config.get("MAX_W", 2650)
        curvature_th = config.get("CURVATURE_TH", 0.01)
        point_xyz = data_dict['point_wise']['point_xyz']
        num_points = point_xyz.shape[0]
        point_rimage_h = data_dict['point_wise']['point_rimage_h']
        point_rimage_w = data_dict['point_wise']['point_rimage_w']
        point_curvature = np.zeros((num_points, 1))
        
        for h in range(max_h):
            mask = np.where(point_rimage_h == h)[0] # [L]
            points = point_xyz[mask] # [L, 3]
            tree = NN(n_neighbors=10).fit(points) 
            dists, indices = tree.kneighbors(points) # [L, 10], [L, 10]
            grouped_points = points[indices, :] # [L, 10, 3]
            diff = grouped_points - points[:, np.newaxis, :] # [L, 10, 3]
            cov = (diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :]).sum(axis=1) # ([L, 10, 3, 1] @ [L, 10, 1, 3]).sum(1) = [L, 3, 3]
            eigvals, eigvecs = np.linalg.eigh(cov)
            curvature = eigvals[:, 1:2]
            #curvature_mask = curvature > curvature_th
            point_curvature[mask] = curvature

        data_dict['point_wise']['curvy'] = (point_curvature > 0.01).astype(np.int64).reshape(-1)
        data_dict['point_wise']['point_curvature'] = point_curvature

        return data_dict

    def lidar_line_segment_v2(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.lidar_line_segment_v2, config=config)

        max_h = config.get("MAX_H", 64)
        max_w = config.get("MAX_W", 2650)
        dist_th = config.get("DIST_TH", 0.05)
        point_xyz = data_dict['point_wise']['point_xyz']
        point_rimage_h = data_dict['point_wise']['point_rimage_h']
        point_rimage_w = data_dict['point_wise']['point_rimage_w']
        point_segment_id = np.zeros(point_xyz.shape[0], dtype=np.int64)
        offset = 0
        for h in range(max_h):
            mask_h = np.where(point_rimage_h == h)[0]
            num_points = mask_h.shape[0]
            if num_points == 0:
                continue
            point_xyz_h = point_xyz[mask_h]
            prange = np.linalg.norm(point_xyz_h, ord=2, axis=-1)
            tree = NN(n_neighbors=10).fit(point_xyz_h)
            dists, indices = tree.kneighbors(point_xyz_h)
            e0 = np.arange(num_points).repeat(10)
            prange = prange.repeat(10)
            e1 = indices.reshape(-1)
            mask = dists.reshape(-1) / (prange + 1e-6) < dist_th
            e0, e1 = e0[mask], e1[mask]
        
            graph = csr_matrix((np.ones_like(e0), (e0, e1)), shape=(num_points, num_points))
            n_components, labels = connected_components(graph, directed=False)
            point_segment_id[mask_h] = offset + labels
            offset += n_components

        data_dict['point_wise']['point_segment_id'] = point_segment_id
        _, counts = np.unique(point_segment_id, return_counts=True)

        #data_dict['point_wise']['point_segment_size'] = counts[point_segment_id]
        data_dict['point_wise']['point_in_large_segment'] = counts[point_segment_id] > 30
        print('size of large segment point cloud', (counts[point_segment_id] > 10).sum())
        print('size of large segment', (counts > 10).sum())
        print('size of segment', counts.shape)

        return data_dict

    def remove_seg_class(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.remove_seg_class, config=config)
        removed_classes = config.get("REMOVED_CLASSES", [])
        point_wise_dict = data_dict['point_wise']
        mask = np.zeros(point_wise_dict['point_xyz'].shape[0], dtype=bool)
        seg_labels = point_wise_dict['segmentation_label']
        for cls in removed_classes:
            mask[seg_labels == cls] = True
        
        point_wise_dict = common_utils.filter_dict(point_wise_dict, ~mask)
        data_dict['point_wise'] = point_wise_dict
        
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
