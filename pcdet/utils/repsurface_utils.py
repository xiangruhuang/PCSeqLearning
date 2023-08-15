import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pcdet.utils.polar_utils import xyz2sphere, normal2sphere, sphere2normal
from pcdet.utils.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb
from pcdet.ops.pointops.functions import pointops
from pcdet.utils.sliding_utils import slide_point_factory


def sample_and_group(stride, nsample, center, normal, feature, offset, return_idx=False, return_normal=True,
                     return_polar=False, num_sector=1):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [N, 3]
        points: input points data, [N, C]
    Return:
        new_xyz: sampled points position data, [M, nsample, 3]
        new_points: sampled points data, [M, nsample, 3+D]
    """
    # sample
    if stride > 1:
        new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if num_sector > 1:
            fps_idx = pointops.sectorized_fps(center, offset, new_offset, num_sector)  # [M]
        else:
            fps_idx = pointops.furthestsampling(center, offset, new_offset)  # [M]
        # fps_idx = pointops.furthestsampling(center, offset, new_offset)  # [M]
        new_center = center[fps_idx.long(), :]  # [M, 3]
        new_normal = normal[fps_idx.long(), :]  # [M, 3]
    else:
        new_center = center
        new_normal = normal
        new_offset = offset

    # group
    N, M, D = center.shape[0], new_center.shape[0], normal.shape[1]
    group_idx, _ = pointops.knnquery(nsample, center, new_center, offset, new_offset)  # [M, nsample]
    group_center = center[group_idx.view(-1).long(), :].view(M, nsample, 3)  # [M, nsample, 3]
    group_normal = normal[group_idx.view(-1).long(), :].view(M, nsample, D)  # [M, nsample, 10]
    group_center_norm = group_center - new_center.unsqueeze(1)
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)

    if feature is not None:
        C = feature.shape[1]
        group_feature = feature[group_idx.view(-1).long(), :].view(M, nsample, C)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature],
                                dim=-1) if return_normal \
            else torch.cat([group_center_norm, group_feature], dim=-1)  # [npoint, nsample, C+D]
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

    if return_idx:
        return new_center, new_normal, new_feature, new_offset, group_idx
    else:
        return new_center, new_normal, new_feature, new_offset


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    :param points: [N, G, 3]
    :param idx: [N, G]
    :return: [N, G, 3]
    """
    device = points.device
    N, G, _ = points.shape

    n_indices = torch.arange(N, dtype=torch.long).to(device).view([N, 1]).repeat([1, G])
    new_points = points[n_indices, idx, :]

    return new_points


def sort_factory(s_type):
    if s_type is None:
        return group_by_umbrella
    elif s_type == 'fix':
        return group_by_umbrella_v2
    elif s_type == 'svd':
        return group_by_umbrella_svd
    else:
        raise Exception('No such sorting method')


def group_by_umbrella(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def _fixed_rotate(xyz):
    # y-axis:45deg -> z-axis:45deg
    rot_mat = torch.FloatTensor([[0.5, -0.5, 0.7071], [0.7071, 0.7071, 0.], [-0.5, 0.5, 0.7071]]).to(xyz.device)
    return xyz @ rot_mat


def group_by_umbrella_v2(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(_fixed_rotate(group_xyz_norm))[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def _rotate_by_normal(xyz, normal):
    # keep dim-1 positive
    # refer: https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector
    normal *= (normal[..., 0] > 0).float().unsqueeze(-1) * 2. - 1.

    n_x = normal[..., 0]
    n_y = normal[..., 1]
    n_z = normal[..., 2]
    l_xy = torch.sqrt(n_x ** 2 + n_y ** 2)
    zero = torch.zeros_like(n_x)
    rot = torch.stack([torch.stack([n_y / l_xy, -n_x / l_xy, zero], -1),
                       torch.stack([n_x * n_z / l_xy, n_y * n_z / l_xy, -l_xy], -1),
                       torch.stack([n_x, n_y, n_z], -1)], -1)
    return xyz @ rot


def group_by_umbrella_svd(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    cov = group_xyz_norm.transpose(-2, -1) @ group_xyz_norm
    est_normal = torch.svd(cov)[2][..., -1]
    group_phi = xyz2sphere(_rotate_by_normal(group_xyz_norm, est_normal))[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def dropout_feature(feat, p=0.3, keep_normal=True):
    """
    Dropout features

    Input:
        feat: [N, 10 (center-3 + polar-3 + normal-3 + position-1)]
    """
    if np.random.rand() < p:
        feat[:, 0:3] = 0
    if np.random.rand() < p:
        feat[:, 3:6] = 0
    if not keep_normal and np.random.rand() < p:
        feat[:, 6:9] = 0
    if np.random.rand() < p:
        feat[:, 9:10] = 0
    return feat


def dropout_feature_instance(feat, offset, p=0.5):
    """
    Dropout features (instance-level)

    Input:
        feat: [N, 10 (center-3 + polar-3 + normal-3 + position-1)]
    """

    if np.random.rand() < p:
        feat[:, 0:3] = 0
    if np.random.rand() < p:
        feat[:, 3:6] = 0
    if np.random.rand() < p:
        feat[:, 6:9] = 0
    if np.random.rand() < p:
        feat[:, 9:10] = 0
    return feat


def jitter_normal(normal, factor=0.003, prob=0.95, anisotropic=True):
    if np.random.rand() < prob:
        N, G, K, _ = normal.shape
        angle_shape = (N, G, K, 2) if anisotropic else (N, 1, K, 2)

        sphere = normal2sphere(normal, normalize=True) * 2. - 1.  # range: [-1, 1]
        sphere = torch.clamp(sphere + torch.clamp(torch.randn(angle_shape) * factor, -3 * factor, 3 * factor), -1, 1)
        sphere[..., 0] = (sphere[..., 0] * np.pi + np.pi) / 2.
        sphere[..., 1] = sphere[..., 1] * np.pi
        normal = sphere2normal(sphere)
        return normal


class SurfaceAbstraction(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, stride, nsample, in_channel, mlp, return_normal=True):
        super(SurfaceAbstraction, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.return_normal = return_normal
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_nor_feat_off):
        """
        Input:
            normal: input normal data, [B, 3, N]
            center: input centroid data, [B, 3, N]
            feature: input feature data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        center, normal, feature, offset = pos_nor_feat_off  # [N, 3], [N, 4/10], [N, C], [B]

        new_center, new_normal, new_feature, new_offset = sample_and_group(self.stride, self.nsample, center,
                                                                           normal, feature, offset,
                                                                           return_polar=self.return_polar)

        # new_center: sampled points position data, [M, 3]
        # new_normal: sampled normal feature data, [M, 3/10]
        # new_feature: sampled feature, [M, nsample, 3+3/10+C]
        new_feature = new_feature.transpose(1, 2).contiguous()  # [M, 3+C, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        return [new_center, new_normal, new_feature, new_offset]


class SurfaceAbstractionCN2(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, stride, nsample, feat_channel, pos_channel, mlp, return_normal=True, return_polar=False):
        super(SurfaceAbstractionCN2, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel

        self.mlp_l0 = nn.Conv1d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv1d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm1d(mlp[0])
        self.bn_f0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_nor_feat_off):
        """
        Input:
            normal: input normal data, [B, 3, N]
            center: input centroid data, [B, 3, N]
            feature: input feature data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        center, normal, feature, offset = pos_nor_feat_off  # [N, 3], [N, 4/10], [N, C], [B]

        new_center, new_normal, new_feature, new_offset = sample_and_group(self.stride, self.nsample, center,
                                                                           normal, feature, offset,
                                                                           return_polar=self.return_polar)

        # new_center: sampled points position data, [M, 3]
        # new_normal: sampled normal feature data, [M, 3/10]
        # new_feature: sampled feature, [M, nsample, 3+3/10+C]
        new_feature = new_feature.transpose(1, 2).contiguous()  # [M, 3+C, nsample]

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        return [new_center, new_normal, new_feature, new_offset]


class SurfaceFeaturePropagationCN2(nn.Module):
    def __init__(self, prev_channel, skip_channel, mlp):
        super(SurfaceFeaturePropagationCN2, self).__init__()
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
            xyz1: input points position data, [N, 3]
            xyz2: sampled input points position data, [M, 3]
            points1: input points data, [N, C]
            points2: input points data, [M, C]
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
            new_points = F.relu(interpolated_points + skip)
        else:
            new_points = F.relu(interpolated_points)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella Surface Representation Constructor

    """

    def __init__(self, k, in_channel, random_inv=True, sort='fix', surf_jitter=False, sj_prob=1., sj_factor=0.01, sj_ani=False):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.random_inv = random_inv
        self.surf_jitter = surf_jitter
        self.sj_prob = sj_prob
        self.sj_factor = sj_factor
        self.sj_ani = sj_ani

        self.mlps = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(True),
            nn.Conv1d(in_channel, in_channel, 1, bias=True),
        )
        self.sort_func = sort_factory(sort)

    def forward(self, center, offset):
        """
        Input:
            center: input centroid data, [N, 3]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # umbrella surface construction
        group_xyz = self.sort_func(center, center, offset, offset, k=self.k)  # [N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        # surface position
        group_pos = cal_const(group_normal, group_center)
        # pad NaN with zero
        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
        if self.surf_jitter and self.training:
            group_normal = jitter_normal(group_normal, self.sj_factor, self.sj_prob, self.sj_ani)
        new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        new_feature = new_feature.transpose(1, 2).contiguous()  # [N, C, G]

        # mapping
        new_feature = self.mlps(new_feature)
        # aggregation
        new_feature = torch.sum(new_feature, 2)

        return new_feature


class UmbrellaSurfaceConstructorSlidingPoint(nn.Module):
    """
    Umbrella Surface Representation Constructor with Sliding Points

    """

    def __init__(self, k, in_channel, random_inv=True, slide_type='gaussian', slide_scale=0.5, slide_prob=1., anisotropic=True,
                 drop_feat=False, sort='fix', surf_jitter=False, sj_prob=1., sj_factor=0.01, sj_ani=False):
        super(UmbrellaSurfaceConstructorSlidingPoint, self).__init__()
        self.k = k
        self.random_inv = random_inv
        self.drop_feat = drop_feat
        self.sort = sort
        self.surf_jitter = surf_jitter
        self.sj_prob = sj_prob
        self.sj_factor = sj_factor
        self.sj_ani = sj_ani

        self.mlps = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(True),
            nn.Conv1d(in_channel, in_channel, 1, bias=True),
        )
        self.slider = slide_point_factory(slide_type, slide_scale, slide_prob, anisotropic)
        self.sort_func = sort_factory(sort)

    def forward(self, center, offset):
        """
        Input:
            center: input centroid data, [N, 3]
        Return:
            new_feature: sampled points position data, [N, C]

        """
        # umbrella surface construction
        group_xyz = self.sort_func(center, center, offset, offset, k=self.k)  # [N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # sliding points
        if self.training:
            group_center = self.slider(group_xyz, group_center, offset)
        # polar
        group_polar = xyz2sphere(group_center)
        # surface position
        group_pos = cal_const(group_normal, group_center)
        # pad NaN with zero
        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
        # normal jitter
        if self.surf_jitter and self.training:
            group_normal = jitter_normal(group_normal, self.sj_factor, self.sj_prob, self.sj_ani)

        new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        new_feature = dropout_feature(new_feature) if self.drop_feat else new_feature
        new_feature = new_feature.transpose(1, 2).contiguous()  # [N, C, G]

        # mapping
        new_feature = self.mlps(new_feature)
        # aggregation
        new_feature = torch.sum(new_feature, 2)

        return new_feature
