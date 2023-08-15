from torch import nn
import torch
from torch_cluster import grid_cluster
import numpy as np
from torch_scatter import scatter

class GridSampling3D(nn.Module):
    def __init__(self, grid_size):
        """
        Args:
            grid_size (int or list of three int)
        """
        super(GridSampling3D, self).__init__()
        self._grid_size = grid_size
        if isinstance(grid_size, list):
            grid_size = torch.tensor([1]+grid_size).float()
        else:
            grid_size = torch.tensor([1]+[grid_size for i in range(3)]).float()
        assert grid_size.shape[0] == 4, "Expecting 4D grid size." 
        self.register_buffer("grid_size", grid_size)

    def forward(self, points, return_inverse=False):
        """
        Args:
            points [N, 4] first dimension is batch index

        Returns:
            sampled_grids [M, 4]
            inv [N] point to voxel mapping

        """
        start = points.min(0)[0]
        start[0] -= 0.5
        end = points.max(0)[0]
        end[0] += 0.5

        cluster = grid_cluster(points, self.grid_size, start=start, end=end)
        unique, inv = torch.unique(cluster, sorted=True, return_inverse=True)
        #perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        #perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
        num_grids = unique.shape[0]
        sampled_grids = scatter(points, inv, dim=0, dim_size=num_grids, reduce='mean')
        if return_inverse:
            return sampled_grids, inv
        else:
            return sampled_grids

    def extra_repr(self):
        return "grid size {}".format(self._grid_size)
