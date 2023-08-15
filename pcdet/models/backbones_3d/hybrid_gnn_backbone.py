import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

from .pointnet2_utils import PointNetSetAbstractionCN2Nor, PointNetFeaturePropagationCN2


class HybridGNN(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(HybridGNN, self).__init__()
        import ipdb; ipdb.set_trace()

    def forward(self, batch_dict):
        import ipdb; ipdb.set_trace()

        return batch_dict
