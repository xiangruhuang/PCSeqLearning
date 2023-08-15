import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict

from pcdet.utils import common_utils, box_utils
from pcdet.models.model_utils import graph_utils
from .registration_module_template import RegistrationTemplate
from pcdet.models.model_utils.grid_sampling import GridSampling3D
from torch_scatter import scatter

class SimpleReg(RegistrationTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg, runtime_cfg, dataset)

        self.module_list = self.build_networks()
        self.pillar_size = model_cfg.get("PILLAR_SIZE", [1, 1])
        self.fake_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.forward_dict = EasyDict()
        self.subsample = model_cfg.get("SUBSAMPLE", False)
        if self.subsample:
            self.grid_sampler = GridSampling3D(grid_size=[0.08, 0.08, 0.08])
            #self.grid_sampler = GridSampling3D(grid_size=[0.2, 0.2, 0.2])

    def process_sequence(self, seq_dict):
        if self.preprocessors:
            for preprocessor_module in self.preprocessors:
                seq_dict = preprocessor_module(seq_dict)
                
                if self.visualizer and preprocessor_module.model_cfg.get("VISUALIZE", False):
                    self.visualizer(seq_dict)


    def format_boxes(self, seq_dict):
        """

        Args:
            
        Returns:
            seq_boxes (dictionary): per-box attributes
                seq_boxes.attr ([B, 7], float32): each box's attributes
                seq_boxes.cls_label ([B, 7], float32): each box's class label {1, 2, 3}
                seq_boxes.trace_id ([B], int32): each box's trace id
                seq_boxes.frame ([B], int32): each box's frame id
        """
        num_frames = seq_dict['point_sweep'].max().long().item() - seq_dict['point_sweep'].min().long().item() + 1

        # extract GT boxes attributes
        seq_box_attr = seq_dict['gt_box_attr'].reshape(-1, 7)
        seq_box_cls_label = seq_dict['gt_box_cls_label'].reshape(-1)
        if seq_box_attr.shape[0] % num_frames != 0:
            import ipdb; ipdb.set_trace()
            pass
        assert seq_box_attr.shape[0] % num_frames == 0, "Weird"
        seq_box_frame_id = torch.repeat_interleave(
                             torch.arange(0, num_frames),
                             seq_box_cls_label.numel() // num_frames,
                             dim=-1
                           ).to(seq_box_cls_label)
        seq_boxes = EasyDict(dict(
                        gt_box_attr=seq_box_attr,
                        gt_box_cls_label=seq_box_cls_label,
                        gt_box_frame=seq_box_frame_id,
                        augmented=seq_dict['augmented'].reshape(-1),
                        num_points_in_gt=seq_dict['num_points_in_gt'].reshape(-1),
                        gt_boxes=seq_dict['gt_boxes'],
                        gt_box_corners_3d=seq_dict['gt_box_corners_3d'],
                    ))

        # remove empty boxes
        non_empty_mask = seq_boxes.gt_box_attr[:, 3:6].norm(p=2, dim=-1) > 1e-5
        seq_boxes = EasyDict(common_utils.filter_dict(seq_boxes, non_empty_mask))
        obj_ids = np.array([seq_dict['obj_ids'][i] for i in non_empty_mask.nonzero()]).astype(str)
        
        seq_box_track_label = np.unique(obj_ids.reshape(-1), return_inverse=True)[1]
        seq_box_track_label = torch.from_numpy(seq_box_track_label).to(seq_box_cls_label).long()
        seq_boxes.gt_box_track_label = seq_box_track_label
        seq_dict['obj_ids'] = obj_ids

        velo = torch.zeros_like(seq_boxes.gt_box_attr[:, 0])
        for trace_id in seq_boxes.gt_box_track_label.unique().tolist():
            trace_mask = (seq_boxes.gt_box_track_label == trace_id).reshape(-1)
            trace_frame = seq_boxes.gt_box_frame[trace_mask]
            sorted_idx = torch.argsort(trace_frame)
            trace_attr = seq_boxes.gt_box_attr[trace_mask][sorted_idx]
            trace_corners = box_utils.boxes_to_corners_3d(trace_attr)

            trace_velo = torch.zeros_like(trace_attr[:, 0])
            if trace_velo.numel() > 1:
                trace_velo[1:] = (trace_corners[1:] - trace_corners[:-1]).norm(p=2, dim=-1).mean(dim=-1)
                trace_velo[0] = trace_velo[1]

            velo[trace_mask.nonzero()[:, 0][sorted_idx]] = trace_velo
        seq_boxes.gt_box_velo = velo
        seq_boxes.moving = velo > 5e-2
            
        for key in seq_boxes.keys():
            seq_dict[key] = seq_boxes[key]

        return seq_dict

    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']
        for b in range(batch_size):
            seq_dict = EasyDict(dict())

            batch_mask = (batch_dict['point_bxyz'][:, 0] == b).reshape(-1)
            for key in ['point_bxyz', 'point_feat', 'segmentation_label',
                        'instance_label', 'is_foreground', 'point_sweep',
                        ]:
                if key in batch_dict:
                    seq_dict[key] = batch_dict[key][batch_mask]
            seq_dict['point_fxyz'] = torch.cat([seq_dict['point_sweep'].reshape(-1, 1),
                                                seq_dict['point_bxyz'][:, 1:]],
                                                dim=-1)
            seq_dict.pop('point_bxyz')

            if self.subsample:
                pointcloud_torch = seq_dict['point_fxyz']
                _, inv = self.grid_sampler(pointcloud_torch, return_inverse=True)
                random_idx = scatter(torch.arange(pointcloud_torch.shape[0]).cuda(),
                                     inv, dim_size=inv.max().long()+1, dim=0,
                                     reduce='max')
                print(f'num points={random_idx.shape[0]}')
                for key in ['point_fxyz', 'point_feat', 'segmentation_label',
                            'instance_label', 'is_foreground', 'point_sweep',
                            ]:
                    if key in seq_dict:
                        seq_dict[key] = seq_dict[key][random_idx]
            
            for key in ['gt_box_cls_label', 'gt_box_attr', 'augmented',
                        'num_points_in_gt', 'gt_boxes', 'obj_ids', 
                        'frame_id', 'pose', 'top_lidar_origin', 'num_sweeps',
                        'gt_box_corners_3d', 'gt_box_velo',
                        ]:
                if key in batch_dict:
                    seq_dict[key] = batch_dict[key][b]

            seq_dict = self.format_boxes(seq_dict)
            
            sequence_id = seq_dict['frame_id'][0][:-4]
            if not os.path.exists(f'{self.model_cfg.SAVE_DIR}/{sequence_id}/all.pth'):
                print(f'Working on {sequence_id}')
                self.process_sequence(seq_dict)
            else:
                print(f'Skipping {sequence_id}')



        if self.training:
            ret_dict = dict(
                loss=torch.zeros(1, device=batch_dict['point_bxyz'].device, requires_grad=True),
            )
            return ret_dict, {}, {}
        else:
            return {}, None
