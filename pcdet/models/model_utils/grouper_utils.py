from torch import nn
import torch
from torch_cluster import grid_cluster
import numpy as np
from torch_scatter import scatter
from pcdet.ops.hybrid_geop.hybrid_geop_modules import PrimitiveFitting
from .primitive_utils import pca_fitting
from .partition_utils import PARTITIONERS

class GrouperTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super(GrouperTemplate, self).__init__()
        self.model_cfg = model_cfg

    def forward(self, point_bxyz):
        raise NotImplementedError


class VoxelGrouper(GrouperTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        """
        Args in model_cfg:
            grid_size (int or list of three int)
        """
        super(VoxelGrouper, self).__init__(runtime_cfg, model_cfg)

        grid_size = model_cfg.get("GRID_SIZE", None)
        self._grid_size = grid_size
        if isinstance(grid_size, list):
            grid_size = torch.tensor([1]+grid_size).float()
        else:
            grid_size = torch.tensor([1]+[grid_size for i in range(3)]).float()
        assert grid_size.shape[0] == 4, "Expecting 4D grid size." 

        partition_cfg = model_cfg.get("PARTITION_CFG", None)
        partitioner = model_cfg.get("PARTITIONER", None)
        self.partitioner = PARTITIONERS[partitioner](
                               runtime_cfg,
                               partition_cfg
                           )

    def forward(self, points):
        """Partition into groups via voxelization
        Args:
            point_bxyz [N, 4] points. (first dimension is batch index)
        Returns:
            group_id [N] indicate the group id of each point
        """
        ref = EasyDict(dict(
                  bxyz=points.bxyz,
              ))

        ref = self.partitioner(ref, {})

        return ref.group_ids

    def __repr__(self):
        return "VoxelGrouper(grid size={})".format(self._grid_size)


class PrimitiveGrouper(GrouperTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        """
        Args in model_cfg:
            grid_size (int or list of three int)
        """
        super(PrimitiveGrouper, self).__init__(runtime_cfg, model_cfg)

        grid_size = model_cfg.get("GRID_SIZE", None)
        self._grid_size = grid_size
        self.pt = PrimitiveFitting(grid_size, max_num_points=800000)

    def forward(self, points):
        """Partition into groups via ransac
        Args:
            point_bxyz [N, 4] points. (first dimension is batch index)
        Returns:
            group_id [N] indicate the group id of each point
        """
        #start = point_bxyz.min(0)[0]
        #start[0] -= 0.5
        #end = point_bxyz.max(0)[0]
        #end[0] += 0.5
        import ipdb; ipdb.set_trace()
        pca_fitting(points, e_plane, cfg)

        center, covariance, group_ids = self.pt(point_bxyz)
        #cluster = grid_cluster(point_bxyz, self.grid_size, start=start, end=end)
        #_, group_ids = torch.unique(cluster, sorted=True, return_inverse=True)

        return group_ids

    def __repr__(self):
        return "VoxelGrouper(grid size={})".format(self._grid_size)

class ClusterGrouper(GrouperTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        """
        Args in model_cfg:
            grid_size (int or list of three int)
        """
        super(ClusterGrouper, self).__init__(runtime_cfg, model_cfg)

        grid_size = model_cfg.get("GRID_SIZE", None)
        self._grid_size = grid_size
        self.pt = PrimitiveFitting(grid_size, max_num_points=800000)

    def forward(self, point_bxyz):
        """Partition into groups via voxelization
        Args:
            point_bxyz [N, 4] points. (first dimension is batch index)
        Returns:
            group_id [N] indicate the group id of each point
        """
        #start = point_bxyz.min(0)[0]
        #start[0] -= 0.5
        #end = point_bxyz.max(0)[0]
        #end[0] += 0.5

        center, covariance, group_ids = self.pt(point_bxyz)
        #cluster = grid_cluster(point_bxyz, self.grid_size, start=start, end=end)
        #_, group_ids = torch.unique(cluster, sorted=True, return_inverse=True)

        return group_ids

    def __repr__(self):
        return "VoxelGrouper(grid size={})".format(self._grid_size)


GROUPERS = dict(
    VoxelGrouper=VoxelGrouper,
    ClusterGrouper=ClusterGrouper,
)
