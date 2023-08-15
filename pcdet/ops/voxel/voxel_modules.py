import torch
from torch import nn
import numpy as np
from torch_scatter import scatter
from pcdet.utils import common_utils

class VoxelAggregation(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        voxel_size = model_cfg.get("VOXEL_SIZE", None)
        voxel_size = [1] + voxel_size
        self.register_buffer("voxel_size",
                             torch.tensor(voxel_size).float(),
                             persistent=False)
        point_cloud_range = model_cfg.get("POINT_CLOUD_RANGE", None)
        if point_cloud_range is not None:
            point_cloud_range = [0] + point_cloud_range[:3] + [1] + point_cloud_range[3:]
            self.register_buffer("point_cloud_range",
                                 torch.tensor(point_cloud_range).float(),
                                 persistent=False)
        else:
            self.point_cloud_range = None

    def forward(self, point_wise_mean_dict, point_wise_median_dict=None):
        """
        Args:
            point_bxyz [N, 4]
            point_feat [N, C]
            **point_wise_attributes

        Returns:
            voxel_wise_dict: dictionary of per voxel attributes of shape [V, ?]
            voxel_index: the voxel index of each point [N]
            num_voxels: V
        """
        
        point_bxyz = point_wise_mean_dict['point_bxyz']
        #point_feat = point_wise_mean_dict['point_feat']
        batch_size = point_bxyz[:, 0].max().long().item() + 1

        if self.point_cloud_range is None:
            pc_range_min = point_bxyz.min(0)[0]
            voxel_coords = torch.floor((point_bxyz-pc_range_min) / self.voxel_size).long()
            voxel_coords = voxel_coords - voxel_coords.min(0)[0]
            dims = voxel_coords.max(0)[0] + 1
            out_of_boundary_mask = torch.zeros(voxel_coords.shape[0], dtype=torch.bool, device=voxel_coords.device)
        else:
            pc_range_min = self.point_cloud_range[:4].clone()
            pc_range_max = self.point_cloud_range[4:].clone()
            pc_range_min[0] = 0
            pc_range_max[0] = batch_size-1e-5
            voxel_coords = torch.floor((point_bxyz-pc_range_min) / self.voxel_size).long()
            dims = torch.ceil((pc_range_max - pc_range_min) / self.voxel_size).long() + 1
            out_of_boundary_mask = (voxel_coords >= dims)[:, 1:4].any(-1) | (voxel_coords < 0)[:, 1:4].any(-1)
            voxel_coords = voxel_coords[~out_of_boundary_mask]
            point_bxyz = point_bxyz[~out_of_boundary_mask]
            point_wise_mean_dict = common_utils.filter_dict(point_wise_mean_dict, ~out_of_boundary_mask)
        point_coords = voxel_coords
            
        assert (voxel_coords >= 0).all(), f"VoxelGraph: min={voxel_coords.min(0)[0]}"
        assert (voxel_coords < dims).all(), f"VoxelGraph: min={voxel_coords.max(0)[0]}, dims={dims}"

        merge_coords = torch.zeros_like(voxel_coords[:, 0])
        for i in range(dims.shape[0]):
            merge_coords = merge_coords * dims[i] + voxel_coords[:, i]

        unq_coor1d, voxel_index, voxel_count = \
                torch.unique(
                    merge_coords, return_inverse=True, return_counts=True
                )

        num_voxels = unq_coor1d.shape[0]
        unq_coords = unq_coor1d.new_zeros(num_voxels, 4)
        for i in range(dims.shape[0]-1, -1, -1):
            unq_coords[:, i] = unq_coor1d % dims[i]
            unq_coor1d = torch.div(unq_coor1d, dims[i], rounding_mode='trunc').long()
        voxel_center = (unq_coords * self.voxel_size + pc_range_min)[:, 1:4]
        voxel_center += self.voxel_size[1:4] / 2

        voxel_wise_mean_dict = {}
        for key in point_wise_mean_dict.keys():
            voxel_key = 'voxel_'+key.split('point_')[-1]
            voxel_wise_mean_dict[voxel_key] = scatter(point_wise_mean_dict[key], voxel_index, dim_size=num_voxels, dim=0, reduce='mean')
        voxel_bxyz = voxel_wise_mean_dict['voxel_bxyz']

        if point_wise_median_dict is not None:
            degree = scatter(torch.ones_like(voxel_index), voxel_index, dim=0, dim_size=num_voxels, reduce='sum')
            offset = degree.cumsum(dim=0) - degree
            median_offset = offset + torch.div(degree, 2, rounding_mode='floor')
            ret_median_dict = {}
            for key, val in point_wise_median_dict.items():
                val = val[~out_of_boundary_mask]
                max_val, min_val = val.max(), val.min()
                tval = (val - min_val) + voxel_index * max_val
                sorted_vals, indices = torch.sort(tval)
                voxel_median_val = sorted_vals[median_offset] - torch.arange(num_voxels).to(val) * max_val
                ret_median_dict['voxel_'+key] = voxel_median_val

        point_xyz = point_bxyz[:, 1:4].contiguous()
        voxel_xyz = voxel_bxyz[:, 1:4].contiguous()
        voxel_wise_dict = dict(
            voxel_batch_index=voxel_bxyz[:, 0].round().long(),
            voxel_xyz=voxel_xyz,
            voxel_center=voxel_center,
            voxel_bcenter=torch.cat([voxel_bxyz[:, 0:1], voxel_center], dim=-1),
            voxel_bcoords=unq_coords,
        )
        voxel_wise_dict.update(voxel_wise_mean_dict)

        if point_wise_median_dict is not None:
            voxel_wise_dict.update(ret_median_dict)
        diff_min = (point_xyz - voxel_center[voxel_index] + self.voxel_size[1:4] / 2)
        diff_max = (point_xyz - voxel_center[voxel_index] - self.voxel_size[1:4] / 2)
        try:
            assert (diff_min > -1e-4).all()
            assert (diff_max < 1e-4).all()
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(e)

        point_wise_dict = dict(
            voxel_index=voxel_index,
            point_bcoords=point_coords,
        )
        return voxel_wise_dict, point_wise_dict, num_voxels, out_of_boundary_mask

    def extra_repr(self):
        return f"voxel_size={list(self.voxel_size.detach().cpu().numpy())[1:]}"

