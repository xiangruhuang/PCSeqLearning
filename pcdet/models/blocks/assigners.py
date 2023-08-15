import torch
from torch import nn
import numpy as np

import torch_cluster


def compute_conv3d_positions(voxel_size):
    vx, vy, vz = voxel_size
    kernel_pos = []
    for dx in [-vx, 0, vx]:
        for dy in [-vy, 0, vy]:
            for dz in [-vz, 0, vz]:
                kernel_pos.append([dx, dy, dz])
    kernel_pos = np.array(kernel_pos)
    kernel_pos = torch.from_numpy(kernel_pos).float()

    return kernel_pos

def compute_ball_positions(num_kernel_points, radius=0.9):
    """Find K kernel positions evenly distributed inside a unit 3D ball, 
    computed via Farthest Point Sampling.

    Args:
        num_kernel_points: integer, denoted as K.
    Returns:
        kernel_pos [K, 3] kernel positions
    """
    X = Y = Z = torch.linspace(-1, 1, 100)
    
    candidate_points = torch.stack(torch.meshgrid(X, Y, Z), dim=-1).reshape(-1, 3)
    candidate_mask = candidate_points.norm(p=2, dim=-1) <= radius
    candidate_points = candidate_points[candidate_mask]

    ratio = (num_kernel_points + 1) / candidate_points.shape[0]
    kernel_pos_index = torch_cluster.fps(candidate_points, None, ratio,
                                         random_start=False)[:num_kernel_points]

    kernel_pos = candidate_points[kernel_pos_index]

    return kernel_pos


class GridVolumeAssigner(nn.Module):
    def __init__(self, assigner_cfg):
        super().__init__()

    @torch.no_grad()
    def forward(self, ref, query, e_ref, e_query):
        assert query.get("volume_mask", None) is not None

        relative_bcoords = ref.bcoords[e_ref] - query.bcoords[e_query]
        assert (relative_bcoords.shape[-1] == 4) and (relative_bcoords.dtype==torch.long)

        relative_coord = relative_bcoords[:, 1:4]
        kernel_index = torch.zeros(relative_coord.shape[0], dtype=torch.long,
                                   device=relative_coord.device)
        for i in [2, 1, 0]:
            sign = relative_coord[:, i].sign()
            offset = sign + 1
            kernel_index = kernel_index * 3 + offset
        kernel_index = 2 * kernel_index + query.volume_mask[e_query].long()
            
        return kernel_index


class GridAssigner(nn.Module):
    def __init__(self, assigner_cfg):
        super().__init__()

    @torch.no_grad()
    def forward(self, ref, query, e_ref, e_query):
        relative_bcoords = ref.bcoords[e_ref] - query.bcoords[e_query]
        assert (relative_bcoords.shape[-1] == 4) and (relative_bcoords.dtype==torch.long)

        relative_coord = relative_bcoords[:, 1:4]
        kernel_index = torch.zeros(relative_coord.shape[0], dtype=torch.long,
                                   device=relative_coord.device)
        for i in [2, 1, 0]:
            sign = relative_coord[:, i].sign()
            offset = sign + 1
            kernel_index = kernel_index * 3 + offset
            
        return kernel_index


class Grid3x3Assigner(nn.Module):
    def __init__(self, assigner_cfg):
        super().__init__()
        half_voxel_size = torch.tensor(assigner_cfg["VOXEL_SIZE"]) / 2.0
        self.register_buffer('half_voxel_size', half_voxel_size, persistent=False)
        self.relative_key = assigner_cfg.get("RELATIVE_KEY", 'bxyz') 

    @torch.no_grad()
    def forward(self, ref, query, e_ref, e_query):
        relative = ref[self.relative_key][e_ref] - query[self.relative_key][e_query]
        assert (relative.shape[-1] == 4) and (relative.dtype == torch.float32)

        relative = relative[:, 1:4]
        kernel_index = torch.zeros(relative.shape[0], dtype=torch.long,
                                   device=relative.device)
        for i in [2, 1, 0]:
            is_zero = (relative[:, i] < self.half_voxel_size[i]) & (relative[:, i] > -self.half_voxel_size[i])
            sign = relative[:, i].sign()
            sign[is_zero] = 0 # in range [-1, 0, 1]
            offset = sign + 1
            kernel_index = kernel_index * 3 + offset
            
        return kernel_index


class GeometricAssigner(nn.Module):
    def __init__(self, assigner_cfg):
        super().__init__()
        self.num_kernels = assigner_cfg.get("NUM_KERNELS", 27)
        self.voxel_size = assigner_cfg.get("VOXEL_SIZE", None)
        kernel_pos = compute_conv3d_positions(self.voxel_size)

        self.register_buffer("kernel_pos", kernel_pos)

    @torch.no_grad()
    def forward(self, ref, query, e_ref, e_query):
        relative_bxyz = ref.bxyz[e_ref] - query.bxyz[e_query]
        assert (relative_bxyz.shape[-1] == 4) and (relative_bxyz.dtype==torch.float32)

        relative_xyz = relative_bxyz[:, 1:4]

        dist = (relative_xyz[:, None, :] - self.kernel_pos[None, :, :]).norm(p=2, dim=-1)

        return dist.argmin(dim=1).long()

    def extra_repr(self):
        return f"voxel_size={self.voxel_size}, num_kernels={self.num_kernels}"

ASSIGNERS = dict(
    GeometricAssigner=GeometricAssigner,
    GridAssigner=GridAssigner,
    Grid3x3Assigner=Grid3x3Assigner,
    GridVolumeAssigner=GridVolumeAssigner,
)

def build_assigner(assigner_cfg):
    assigner = assigner_cfg["TYPE"]
    return ASSIGNERS[assigner](assigner_cfg)
