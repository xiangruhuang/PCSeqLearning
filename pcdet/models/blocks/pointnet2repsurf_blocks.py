import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from ...utils.polar_utils import xyz2sphere, xyz2cylind, xyz2sphere_aug
from ...ops.pointops.functions import pointops

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def batch_index_to_offset(batch_index):
    if batch_index.dtype != torch.int64:
        batch_index = batch_index.round().long()
    batch_size = batch_index.max().item() + 1
    num_points = []
    for i in range(batch_size):
        num_points.append((batch_index == i).sum().int())
    num_points = torch.tensor(num_points).int().to(batch_index.device)
    offset = num_points.cumsum(dim=0).int()

    return offset

def pc_normalize(pc, norm='instance'):
    """
    Batch Norm to Instance Norm
    Normalize Point Clouds | Pytorch Version | Range: [-1, 1]

    :param pc: [B, 3/6, N]
    :param norm: 'instance', 'batch'
    :return: [B, 3/6, N]

    """
    points = pc[:, :3, :]
    centroid = torch.mean(points, dim=2, keepdim=True)
    points = points - centroid
    if norm == 'instance':
        m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)), dim=1)[0]
        pc[:, :3, :] = points / m.view(-1, 1, 1)
    else:
        m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
        pc[:, :3, :] = points / m
    return pc

def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]

    FLOPs:
        (1 + 2 + 1) * N + (1 + 2 + 1) * M + N * M * (3 + 2)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def euclidean_distance(src, dst):
    """
    Calculate Euclidean distance

    :param src: [B, N, C]
    :param dst: [B, M, C]
    :return: [B, N, M]
    """
    return torch.norm(src.unsqueeze(-2) - dst.unsqueeze(-3), p=2, dim=-1)

def sample_and_group(stride, nsample, xyz, points, offset,
                     return_idx=False, return_polar=False, num_sectors=1):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, C]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    # sample
    if stride > 1:
        new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if num_sectors > 1:
            fps_idx = pointops.sectorized_fps(xyz, offset, new_offset, num_sectors)  # [M]
        else:
            fps_idx = pointops.furthestsampling(xyz, offset, new_offset)  # [M]
        new_xyz = xyz[fps_idx.long(), :]  # [M, 3]
    else:
        new_xyz = xyz
        new_offset = offset

    # group
    N, M = xyz.shape[0], new_xyz.shape[0],
    group_idx, _ = pointops.knnquery(nsample, xyz, new_xyz, offset, new_offset)  # [M, nsample]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(M, nsample, 3)  # [M, nsample, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(1)
    if return_polar:
        group_polar = xyz2sphere(group_xyz_norm)
        group_xyz_norm = torch.cat([group_xyz_norm, group_polar], dim=-1)

    if points is not None:
        C = points.shape[1]
        group_points = points[group_idx.view(-1).long(), :].view(M, nsample, C)
        new_points = torch.cat([group_xyz_norm, group_points], dim=-1)  # [M, nsample, 3/6+C]
    else:
        new_points = group_xyz_norm
    if return_idx:
        return new_xyz, new_points, new_offset, group_idx
    else:
        return new_xyz, new_points, new_offset

class PointNetSetAbstractionCN2Nor(nn.Module):
    """
    SA Module (normal input) with CN (pre-bn; xyz and normal separate)

    """

    def __init__(self, stride, nsample, in_channel, mlp, return_polar=False, num_sectors=1):
        super(PointNetSetAbstractionCN2Nor, self).__init__()
        self.stride = stride
        self.return_polar = return_polar
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = 6 if return_polar else 3
        self.num_sectors = num_sectors

        self.mlp_l0 = nn.Conv1d(self.pos_channel, mlp[0], 1, bias=False)
        self.norm_l0 = nn.BatchNorm1d(mlp[0])
        if in_channel > 0:
            self.mlp_f0 = nn.Conv1d(in_channel, mlp[0], 1, bias=False)
            self.norm_f0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat_off):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        xyz, points, offset = pos_feat_off  # [N, 3], [N, C], [B]

        new_xyz, new_points, new_offset = sample_and_group(self.stride, self.nsample,
                                                           xyz, points, offset,
                                                           return_polar=self.return_polar,
                                                           num_sectors=self.num_sectors)

        ## new_xyz: sampled points position data, [M, 3]
        ## new_points: sampled points data, [M, nsample, C+3]
        new_points = new_points.transpose(1, 2).contiguous()  # [M, 3+C, nsample]

        # init layer
        loc = self.norm_l0(self.mlp_l0(new_points[:, :self.pos_channel]))
        if points is not None:
            feat = self.norm_f0(self.mlp_f0(new_points[:, self.pos_channel:]))
            new_points = F.relu(loc + feat, inplace=False)
        else:
            new_points = F.relu(loc, inplace=False)

        ## mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = conv(new_points)
            new_points = F.relu(bn(new_points), inplace=False)
        new_points = torch.max(new_points, 2)[0]

        return [new_xyz, new_points, new_offset]


class PointNetFeaturePropagationCN2(nn.Module):
    def __init__(self, prev_channel, skip_channel, mlp):
        super(PointNetFeaturePropagationCN2, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.skip = skip_channel is not None

        self.mlp_f0 = nn.Linear(prev_channel, mlp[0])
        self.norm_f0 = nn.BatchNorm1d(mlp[0])
        if skip_channel is not None:
            self.mlp_s0 = nn.Linear(skip_channel, mlp[0])
            self.norm_s0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat_off1, pos_feat_off2):
        """
        Input:
            xyz1: input points position data, [B, 3, N]
            xyz2: sampled input points position data, [B, 3, S]
            points1: input points data, [B, C, N]
            points2: input points data, [B, C, S]
        Return:
            new_points: upsampled points data, [B, C', N]
        """
        xyz1, points1, offset1 = pos_feat_off1  # [N, 3], [N, C], [B]
        xyz2, points2, offset2 = pos_feat_off2  # [M, 3], [M, C], [B]

        # interpolation
        idx, dist = pointops.knnquery(3, xyz2, xyz1, offset2, offset1)  # [M, 3], [M, 3]
        dist_recip = 1.0 / (dist + 1e-8)  # [M, 3]
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm  # [M, 3]

        points2 = self.norm_f0(self.mlp_f0(points2))
        interpolated_points = torch.cuda.FloatTensor(xyz1.shape[0], points2.shape[1]).zero_()
        for i in range(3):
            interpolated_points += points2[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)

        # init layer
        if self.skip:
            skip = self.norm_s0(self.mlp_s0(points1))
            new_points = F.relu(interpolated_points + skip, inplace=False)
        else:
            new_points = F.relu(interpolated_points, inplace=False)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=False)

        return new_points
