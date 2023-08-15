import torch

@torch.no_grad()
def bxyz_to_xyz_index_offset(point_bxyz):
    num_points = []
    batch_size = point_bxyz[:, 0].max().round().long().item() + 1
    for i in range(batch_size):
        num_points.append((point_bxyz[:, 0].round().long() == i).int().sum())
    num_points = torch.stack(num_points, dim=0).reshape(-1).int()
    _, indices = torch.sort(point_bxyz[:, 0])
    offset = num_points.cumsum(dim=0).int()
    point_xyz = point_bxyz[indices, 1:4].contiguous()
    return point_xyz, indices.long(), offset

