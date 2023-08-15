import torch
import torch.nn as nn

from .pointops import (
    furthestsampling,
    sectorized_fps,
    knnquery
)

def knn_graph(point_bxyz, stride, k, num_sectors=1):
    """Make a M-subset of N-points and query a fixed number of nearest neighbors.
    
    Input:
        point_bxyz: input point position data, [N, 4], first dimension is batch index,
                    assuming points are sorted by batch index.
        stride: sampling rate = 1/stride
        k: number of neighbors to sample
        num_sectors: acceleration factor for KNN, should have almost no effect for values smaller than 6.
    Return:
        fps_idx [M]: point indices downsampled.
        edge_indices [2, E]: (idx_of_downsampled_points, idx_of_original_points)
    """
    num_points = []
    batch_size = point_bxyz[:, 0].max().round().long().item() + 1
    for i in range(batch_size):
        num_points.append(point_bxyz[:, 0].round().long() == i)
    num_points = torch.cat(num_points)
    _, indices = torch.sort(point_bxyz[:, 0])
    offset = num_points.cumsum()
    
    point_xyz = point_bxyz[indices, 1:4].contiguous()

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
        fps_idx = indices[fps_idx]
    else:
        new_xyz = point_xyz
        new_offset = offset

    # building a bipartite graph from [N] to [M]
    N, M = point_xyz.shape[0], new_xyz.shape[0]
    src_idx, _ = knnquery(k, point_xyz, new_xyz, offset, new_offset)  # [M, k] -> [N]
    src_idx = indices[src_idx]
    tgt_idx = torch.arange(k)[None, :].expand(0, M).to(src_idx) # [M, k] -> [M]
    edge_index = torch.stack([tgt_idx, src_idx], dim=0)

    return fps_idx, edge_index
