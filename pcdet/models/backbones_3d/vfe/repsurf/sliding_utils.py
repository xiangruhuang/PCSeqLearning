import torch
from torch import nn
import numpy as np


def slide_point_factory(slide_type, slide_scale, slide_prob, anisotropic):
    if slide_type == 'uniform':
        return UniformSlidePoint(slide_scale, anisotropic, slide_prob)
    elif slide_type == 'gaussian':
        return GaussianSlidePoint(slide_scale, anisotropic, slide_prob)
    else:
        raise Exception('No Such Sliding Type')


def _generate_mask(offset, prob):
    B = offset.shape[0]
    batch_option = np.random.rand(B) < prob

    mask_list = []
    for idx, option in enumerate(batch_option):
        mask_len = offset[0] if idx == 0 else offset[idx] - offset[idx - 1]
        if option:
            mask_list.append(np.ones(mask_len))
        else:
            mask_list.append(np.zeros(mask_len))

    return torch.from_numpy(np.hstack(mask_list)).float()


class UniformSlidePoint(nn.Module):
    def __init__(self, max_scale, anisotropic=False, prob=0.5):
        super(UniformSlidePoint, self).__init__()
        self.max_scale = max_scale
        self.anisotropic = anisotropic
        self.prob = prob

    def forward(self, group_xyz, group_center, offset):
        """
        :param group_xyz: [N, K, 3] / [N, G, K, 3]; K >= 3
        :param group_center: [N, 3] / [N, G, 3]; K >= 3

        :return: [N, 3] / [N, G, 3]
        """
        group_edge = group_xyz - group_center.unsqueeze(-2)  # [N, K, 3] / [N, G, K, 3]
        assert group_edge.dim() in [3, 4]
        if group_edge.dim() == 3:
            N, K, _ = group_edge.shape
            scale_shape = (N, K, 1)
            mask_shape = (N, 1)
        elif group_edge.dim() == 4:
            N, G, K, _ = group_edge.shape
            scale_shape = (N, G, K, 1) if self.anisotropic else (N, 1, K, 1)
            mask_shape = (N, 1, 1)

        # uniform distribution: [0, max_scale]
        group_scale = torch.rand(scale_shape) * self.max_scale
        group_scale = group_scale.to(group_xyz.device)
        group_offset = torch.sum(group_edge * group_scale, dim=-2)  # [N, 3] / [N, G, 3]

        # mask
        mask = _generate_mask(offset, self.prob).to(group_offset.device).view(mask_shape) if self.prob < 1 else 1
        group_center = group_center + group_offset * mask
        return group_center


class GaussianSlidePoint(nn.Module):
    def __init__(self, max_scale, anisotropic=False, prob=0.5):
        super(GaussianSlidePoint, self).__init__()
        self.max_scale = max_scale
        self.anisotropic = anisotropic
        self.prob = prob

    def forward(self, group_xyz, group_center, offset):
        """
        :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K >= 3
        :param group_center: [B, N, 3] / [B, N, G, 3]; K >= 3

        :return: [B, N, 3] / [B, N, G, 3]
        """
        group_edge = group_xyz - group_center.unsqueeze(-2)  # [B, N, K, 3] / [B, N, G, K, 3]
        assert group_edge.dim() in [4, 5]
        if group_edge.dim() == 3:
            N, K, _ = group_edge.shape
            scale_shape = (N, K, 1)
            mask_shape = (N, 1)
        elif group_edge.dim() == 4:
            N, G, K, _ = group_edge.shape
            scale_shape = (N, G, K, 1) if self.anisotropic else (N, 1, K, 1)
            mask_shape = (N, 1, 1)

        # gaussian distribution: [0, max_scale]
        group_scale = torch.clamp(torch.abs(torch.randn(scale_shape)), 0, 3) * self.max_scale / 3.
        group_scale = group_scale.to(group_xyz.device)
        group_offset = torch.sum(group_edge * group_scale, dim=-2)  # [B, N, 3] / [B, N, G, 3]

        # mask
        mask = _generate_mask(offset, self.prob).to(group_offset.device).view(mask_shape) if self.prob < 1 else 1
        group_center = group_center + group_offset * mask
        return group_center
