import torch
from torch import nn
import numpy as np

from torch_scatter import scatter
from torch_cluster import grid_cluster
from easydict import EasyDict

from pcdet.ops.pointops.functions.pointops import (
    furthestsampling,
    sectorized_fps,
)
from pcdet.ops.voxel.voxel_modules import VoxelAggregation
from .graph_utils import VoxelGraph
from .misc_utils import bxyz_to_xyz_index_offset
from pcdet.utils import common_utils

class PartitionTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, ref, runtime_dict=None):
        raise NotImplementedError

class GridPartitioner(PartitionTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(GridPartitioner, self).__init__(
                                         runtime_cfg=runtime_cfg,
                                         model_cfg=model_cfg
                                     )
        grid_size = model_cfg.get("GRID_SIZE", None)
        self._grid_size = grid_size
        if isinstance(grid_size, list):
            grid_size = torch.tensor([1]+grid_size).float()
        else:
            grid_size = torch.tensor([1]+[grid_size for i in range(3)]).float()
        assert grid_size.shape[0] == 4, "Expecting 4D grid size." 
        self.register_buffer("grid_size", grid_size)
        
        point_cloud_range = model_cfg.get("POINT_CLOUD_RANGE", None)
        if point_cloud_range is None:
            self.point_cloud_range = None
        else:
            point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
            self.register_buffer("point_cloud_range", point_cloud_range)
        
    def forward(self, ref, runtime_dict):
        point_bxyz = ref.bxyz
        
        if self.point_cloud_range is not None:
            start = self.point_cloud_range.new_zeros(4)
            end = self.point_cloud_range.new_zeros(4)
            start[1:4] = self.point_cloud_range[:3]
            end[1:4] = self.point_cloud_range[3:]
            start[0] = point_bxyz[:, 0].min() - 0.5
            end[0] = point_bxyz[:, 0].max() + 0.5
        else:
            start = point_bxyz.min(0)[0]
            start[0] -= 0.5
            end = point_bxyz.max(0)[0]
            end[0] += 0.5

        cluster = grid_cluster(point_bxyz, self.grid_size, start=start, end=end)
        unique, inv = torch.unique(cluster, sorted=True, return_inverse=True)

        ref.bcenter = ref.bxyz.clone()
        ref.bcenter[:, 1:] = torch.div(ref.bxyz[:, 1:] - start[1:], self.grid_size[1:], rounding_mode='trunc') * self.grid_size[1:] + self.grid_size[1:] / 2 + start[1:]
        ref.partition_id = inv

        return ref

    def extra_repr(self):
        return f"grid_size={self._grid_size}, point_cloud_range={list(self.point_cloud_range)}"


PARTITIONERS = {
    'GridPartitioner': GridPartitioner,
}
