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
from pcdet.models.model_utils.partition_utils import PARTITIONERS
from pcdet.models.model_utils.primitive_utils import pca_fitting

class SamplerTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.required_attributes = model_cfg.get("REQUIRED_ATTRIBUTES", ['bxyz'])
        if not isinstance(self.required_attributes, list):
            self.required_attributes = [self.required_attributes]

    def sample(self, point_bxyz):
        raise NotImplementedError

    def forward(self, point_bxyz):
        result_dict = self.sample(point_bxyz)
        results = []
        for attr in self.required_attributes:
            if attr not in result_dict:
                raise ValueError(f"{self}: Required attribute {attr} not in sample results")
            results.append(result_dict[attr])
        if len(results) == 1:
            return results[0]
        else:
            return results


class SamplerV2Template(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, ref, runtime_dict=None):
        raise NotImplementedError


class VoxelCenterSampler(SamplerV2Template):
    def __init__(self, runtime_cfg, model_cfg):
        super(VoxelCenterSampler, self).__init__(
                                     runtime_cfg=runtime_cfg,
                                     model_cfg=model_cfg,
                                 )
        voxel_size = model_cfg.get("VOXEL_SIZE", None)
        self._voxel_size = voxel_size
        if isinstance(voxel_size, list):
            voxel_size = torch.tensor([1]+voxel_size).float()
        else:
            voxel_size = torch.tensor([1]+[voxel_size for i in range(3)]).float()
        assert voxel_size.shape[0] == 4, "Expecting 4D voxel size." 
        self.register_buffer("voxel_size", voxel_size)

        stride = model_cfg.get("STRIDE", 1)
        if not isinstance(stride, list):
            stride = [stride for i in range(3)]
        stride = torch.tensor(stride, dtype=torch.int64)
        self.register_buffer('stride', stride)
        
        self.z_padding = model_cfg.get("Z_PADDING", 1)
        downsample_times = model_cfg.get("DOWNSAMPLE_TIMES", 1)
        if not isinstance(downsample_times, list):
            downsample_times = [downsample_times for i in range(3)]
        self.downsample_times = downsample_times
        downsample_times = torch.tensor(downsample_times, dtype=torch.float32)
        model_cfg_cp = {}
        model_cfg_cp.update(model_cfg)
        model_cfg_cp['VOXEL_SIZE'] = [voxel_size[1+i] / downsample_times[i] for i in range(3)]
        self.voxel_aggr = VoxelAggregation(model_cfg=model_cfg_cp, runtime_cfg=runtime_cfg)
        
    @torch.no_grad()
    def forward(self, ref, runtime_dict=None):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            voxel_center: [V, 4] sampled centers of voxels
        """

        point_bxyz = ref.bcenter

        point_bxyz_list = []
        for dx in range(-self.stride[2]+1, self.stride[2]):
            for dy in range(-self.stride[2]+1, self.stride[2]):
                for dz in range(-self.stride[2]+1, self.stride[2]):
                    dr = torch.tensor([dx / self.stride[0], dy / self.stride[1], dz / self.stride[2]]).to(self.voxel_size)
                    point_bxyz_this = point_bxyz.clone()
                    point_bxyz_this[:, 1:4] += dr * self.voxel_size[1:]
                    point_bxyz_list.append(point_bxyz_this)
        point_bxyz = torch.cat(point_bxyz_list, dim=0)
            
        point_wise_mean_dict = dict(
            point_bxyz=point_bxyz,
        )

        voxel_wise_dict, point_wise_dict, num_voxels, _ = self.voxel_aggr(point_wise_mean_dict)
        voxel_index = point_wise_dict['voxel_index']

        vc = voxel_wise_dict['voxel_bcoords']
        if self.z_padding == -1:
            mask = (vc[:, 3] % self.downsample_times[2] == 0)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)
        else:
            mask = (vc[:, 3] % self.downsample_times[2] == self.z_padding)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)

        voxel_wise_dict = common_utils.filter_dict(voxel_wise_dict, mask)

        query = EasyDict(dict(
                    bcenter=voxel_wise_dict['voxel_bcenter'],
                    bcoords=voxel_wise_dict['voxel_bcoords'],
                    bxyz=voxel_wise_dict['voxel_bxyz'],
                ))

        return query
        

class VolumeSampler(SamplerV2Template):
    def __init__(self, runtime_cfg, model_cfg):
        super(VolumeSampler, self).__init__(
                                       runtime_cfg=runtime_cfg,
                                       model_cfg=model_cfg,
                                   )
        voxel_size = model_cfg.get("VOXEL_SIZE", None)
        self._voxel_size = voxel_size
        if isinstance(voxel_size, list):
            voxel_size = torch.tensor([1]+voxel_size).float()
        else:
            voxel_size = torch.tensor([1]+[voxel_size for i in range(3)]).float()
        assert voxel_size.shape[0] == 4, "Expecting 4D voxel size." 
        self.register_buffer("voxel_size", voxel_size)
        self.key = model_cfg.get("KEY", 'bcenter')
        self.output_key = model_cfg.get("OUTPUT_KEY", 'bcenter')
        self.from_runtime = model_cfg.get("FROM_RUNTIME", None)

        stride = model_cfg.get("STRIDE", 1)
        if not isinstance(stride, list):
            stride = [stride for i in range(3)]
        stride = torch.tensor(stride, dtype=torch.int64)
        self.register_buffer('stride', stride)
        
        self.z_padding = model_cfg.get("Z_PADDING", 1)
        downsample_times = model_cfg.get("DOWNSAMPLE_TIMES", 1)
        if not isinstance(downsample_times, list):
            downsample_times = [downsample_times for i in range(3)]
        self.downsample_times = downsample_times
        downsample_times = torch.tensor(downsample_times, dtype=torch.float32)

        model_cfg_cp = {}
        model_cfg_cp.update(model_cfg)
        model_cfg_cp['VOXEL_SIZE'] = [voxel_size[1+i] / downsample_times[i] for i in range(3)]
        self.voxel_aggr = VoxelAggregation(model_cfg=model_cfg_cp, runtime_cfg=runtime_cfg)
        
    @torch.no_grad()
    def forward(self, ref, runtime_dict=None):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            voxelwise attributes
        """
        if self.from_runtime is not None:
            ref = runtime_dict[self.from_runtime]

        point_bcenter = ref[self.key]

        point_bcenter_list = []
        for dx in range(-self.stride[2]+1, self.stride[2]):
            for dy in range(-self.stride[2]+1, self.stride[2]):
                for dz in range(-self.stride[2]+1, self.stride[2]):
                    dr = torch.tensor([dx / self.stride[0], dy / self.stride[1], dz / self.stride[2]]).to(self.voxel_size)
                    point_bcenter_this = point_bcenter.clone()
                    point_bcenter_this[:, 1:4] += dr * self.voxel_size[1:]
                    point_bcenter_list.append(point_bcenter_this)
        point_bcenter = torch.cat(point_bcenter_list, dim=0)
        
        point_wise_mean_dict = dict(
            point_bxyz=point_bcenter,
        )

        voxel_wise_dict, point_wise_dict, num_voxels, out_of_boundary_mask = \
                self.voxel_aggr(point_wise_mean_dict)
        voxel_index = point_wise_dict['voxel_index']

        vc = voxel_wise_dict['voxel_bcoords']
        if self.z_padding == -1:
            mask = (vc[:, 3] % self.downsample_times[2] == 0)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)
        else:
            mask = (vc[:, 3] % self.downsample_times[2] == self.z_padding)
            for i in range(2):
                mask &= (vc[:, i+1] % self.downsample_times[i] == 0)

        voxel_wise_dict = common_utils.filter_dict(voxel_wise_dict, mask)
        num_voxels = mask.sum().long().item()

        query = EasyDict(dict(
                    bcoords=voxel_wise_dict['voxel_bcoords'],
                    bcenter=voxel_wise_dict['voxel_bcenter'],
                    bxyz=voxel_wise_dict['voxel_bxyz'],
                ))
             
        return query

    def extra_repr(self):
        return f"stride={self.stride}"


class GridSampler(SamplerTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(GridSampler, self).__init__(
                               runtime_cfg=runtime_cfg,
                               model_cfg=model_cfg,
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

        self.from_base = model_cfg.get("FROM_BASE", False)
        
    def forward(self, points, runtime_dict=None):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            new_bxyz: [M, 4] sampled points, M roughly equals (N // self.stride)
        """
        point_bxyz = points.bxyz
        if self.from_base:
            point_bxyz = runtime_dict['base_bxyz']

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
        #perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        #perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
        num_grids = unique.shape[0]
        sampled_bxyz = scatter(point_bxyz, inv, dim=0, dim_size=num_grids, reduce='mean')
        ret = EasyDict(dict(bxyz=sampled_bxyz))

        coords = torch.div(ret.bxyz[:, 1:] - start[1:], self.grid_size[1:], rounding_mode='trunc')
        ret.bcoords = torch.cat([ret.bxyz[:, :1], coords], dim=-1).round().long()
        center = (coords * self.grid_size[1:]) + start[1:] + self.grid_size[1:] / 2
        ret.bcenter = torch.cat([ret.bxyz[:, :1], center], dim=-1)

        if 'bcenter' in points.keys():
            points.voxel_id = inv
            grid_weight = scatter(torch.ones_like(inv), inv, dim=0, dim_size=num_grids, reduce='sum')
            points.weight = 1.0 / grid_weight[inv]

        return ret


    def extra_repr(self):
        return f"stride={self._grid_size}, point_cloud_range={self.point_cloud_range}"

        
class FPSSampler(SamplerV2Template):
    def __init__(self, runtime_cfg, model_cfg):
        super(FPSSampler, self).__init__(
                                    runtime_cfg=runtime_cfg,
                                    model_cfg=model_cfg,
                                )
        self.stride = model_cfg.get("STRIDE", 1)
        self.num_sectors = model_cfg.get("NUM_SECTORS", 1)
        self.from_base = model_cfg.get("FROM_BASE", False)
        
    def forward(self, point, runtime_dict=None):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            new_bxyz: [M, 4] sampled points, M roughly equals (N // self.stride)
        """
        if self.stride == 1:
            return point

        if self.from_base:
            point = runtime_dict['base']
        point_bxyz = point.bxyz

        point_xyz, point_indices, offset = bxyz_to_xyz_index_offset(point_bxyz)

        # sample
        new_offset = [(offset[0].item() + self.stride - 1) // self.stride]
        sample_idx = (offset[0].item() + self.stride - 1) // self.stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item() + self.stride - 1) // self.stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if (self.num_sectors > 1) and (point_xyz.shape[0] > 100):
            fps_idx = sectorized_fps(point_xyz, offset, new_offset, self.num_sectors) # [M]
        else:
            fps_idx = furthestsampling(point_xyz, offset, new_offset) # [M]
        fps_idx = point_indices[fps_idx.long()]

        ret_point = common_utils.filter_dict(point, fps_idx, ignore_keys=['name'])
        return EasyDict(ret_point)

    def extra_repr(self):
        return f"stride={self.stride}"


class HybridSampler(SamplerV2Template):
    def __init__(self, runtime_cfg, model_cfg):
        super(HybridSampler, self).__init__(
                                    runtime_cfg=runtime_cfg,
                                    model_cfg=model_cfg,
                                )

        partition_cfg = EasyDict(dict(
                            TYPE=model_cfg["PARTITIONER_TYPE"],
                            GRID_SIZE=model_cfg["PARTITION_GRID_SIZE"],
                            POINT_CLOUD_RANGE=model_cfg["POINT_CLOUD_RANGE"],
                        ))
        self.partitioner = PARTITIONERS[partition_cfg['TYPE']](
                               runtime_cfg=None,
                               model_cfg=partition_cfg,
                           )

        self.pca_cfg = EasyDict(model_cfg)
        
    def forward(self, ref, runtime_dict=None):
        """
        Args:
            point_bxyz [N, 4]: (b,x,y,z), first dimension is batch index
        Returns:
            new_bxyz: [M, 4] sampled points, M roughly equals (N // self.stride)
        """

        ref = self.partitioner(ref, runtime_dict)
        points, planes = pca_fitting(ref, ref.partition_id, self.pca_cfg)
        points.update(ref)

        return points, planes

    def extra_repr(self):
        return f"{self.pca_cfg}"

def build_sampler(sampler_cfg, runtime_cfg=None):
    sampler = sampler_cfg["TYPE"]
    return SAMPLERS[sampler](runtime_cfg=runtime_cfg, model_cfg=sampler_cfg)

SAMPLERS = {
    'FPSSampler': FPSSampler,
    'GridSampler': GridSampler,
    'VoxelCenterSampler': VoxelCenterSampler,
    'VolumeSampler': VolumeSampler,
    'HybridSampler': HybridSampler,
}
