from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn
from .pointops import (
    furthestsampling,
    sectorized_fps,
    knnquery
)
from pcdet.utils.polar_utils import xyz2sphere

def sample_and_group(stride, num_neighbors, point_xyz, point_feat, offset,
                     return_idx=False, return_polar=False, num_sectors=1):
    """Make a M-subset of N-points and query a fixed number of nearest neighbors.
    
    Input:
        stride: sampling rate = 1/stride
        num_neighbors: number of neighbors to sample
        point_xyz: input point position data, [N, 3]
        point_feat: input point feature, [N, C]
    Return:
        new_xyz: sampled points position data, [M, num_neighbors, 3]
        new_feat: sampled points feature, [M, num_neighbors, 3+D]
        new_offset: for multiple batches
        fps_idx: (optional) trace-back indices [M]
    """
    # sample
    if stride > 1:
        new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if num_sectors > 1:
            fps_idx = sectorized_fps(point_xyz, offset, new_offset, num_sectors) # [M]
        else:
            fps_idx = furthestsampling(point_xyz, offset, new_offset) # [M]
        new_xyz = point_xyz[fps_idx.long(), :]  # [M, 3]
    else:
        new_xyz = point_xyz
        new_offset = offset

    # group
    N, M = point_xyz.shape[0], new_xyz.shape[0]
    group_idx, _ = knnquery(num_neighbors, point_xyz, new_xyz, offset, new_offset)  # [M, num_neighbors]
    group_xyz = point_xyz[group_idx.view(-1).long(), :].view(M, num_neighbors, 3)  # [M, num_neighbors, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(1)
    if return_polar:
        group_polar = xyz2sphere(group_xyz_norm)
        group_xyz_norm = torch.cat([group_xyz_norm, group_polar], dim=-1)

    if point_feat is not None:
        C = point_feat.shape[1]
        group_feat = point_feat[group_idx.view(-1).long(), :].view(M, num_neighbors, C)
        new_feat = torch.cat([group_xyz_norm, group_feat], dim=-1)  # [M, num_neighbors, 3/6+C]
    else:
        new_feat = group_xyz_norm

    if return_idx:
        return new_xyz, new_feat, new_offset, fps_idx, group_idx
    else:
        return new_xyz, new_feat, new_offset

