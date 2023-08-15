import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_utils, loss_utils
from .point_head_template import PointHeadTemplate


class PrimitiveHead(nn.Module):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.num_class = num_class
    
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        primitives = batch_dict['primitives']
        mu = primitives[:, :4]
        cov = primitives[:, 4:13].view(-1, 3, 3)
        fitness = primitives[:, -1]
        seg_labels = batch_dict['seg_labels']
        seg_labels = seg_labels[:, 0] * self.num_class + seg_labels[:, 1]
        e_point, e_prim = batch_dict['edges'] # (point, primitive)
        points = batch_dict['points']
        edge_weight = batch_dict['edge_weight']
        e_point, e_prim = edges
        import ipdb; ipdb.set_trace()

        return batch_dict
