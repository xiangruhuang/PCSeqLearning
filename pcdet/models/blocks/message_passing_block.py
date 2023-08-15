import torch
from torch import nn

import torch.nn.functional as F
import torch_cluster
from torch_scatter import scatter

from .message_passing import (
    message_passing, # input args: kernel_weights, kernel_pos, ref_bxyz, 
                     #             ref_feat, query_bxyz, e_ref, e_query, num_act_kernels
    message_passing_naive
)


def compute_ball_positions(num_kernel_points):
    """Find K kernel positions evenly distributed inside a unit 3D ball, 
    computed via Farthest Point Sampling.

    Args:
        num_kernel_points: integer, denoted as K.
    Returns:
        kernel_pos [K, 3] kernel positions
    """
    X = Y = Z = torch.linspace(-1, 1, 100)
    
    candidate_points = torch.stack(torch.meshgrid(X, Y, Z), dim=-1).reshape(-1, 3)
    candidate_mask = candidate_points.norm(p=2, dim=-1) <= 1.0
    candidate_points = candidate_points[candidate_mask]

    ratio = (num_kernel_points + 1) / candidate_points.shape[0]
    kernel_pos_index = torch_cluster.fps(candidate_points, None, ratio,
                                         random_start=False)[:num_kernel_points]

    kernel_pos = candidate_points[kernel_pos_index]

    return kernel_pos

def compute_sphere_positions(num_kernel_points):
    """Find K kernel positions evenly distributed inside a unit 3D ball, 
    computed via Farthest Point Sampling.

    Args:
        num_kernel_points: integer, denoted as K.
    Returns:
        kernel_pos [K, 3] kernel positions
    """
    num_kernel_points -= 1
    X = Y = Z = torch.linspace(-1, 1, 100)
    
    candidate_points = torch.stack(torch.meshgrid(X, Y, Z), dim=-1).reshape(-1, 3)
    candidate_norm = candidate_points.norm(p=2, dim=-1, keepdim=True)
    candidate_points = candidate_points / candidate_norm

    ratio = (num_kernel_points + 1) / candidate_points.shape[0]
    kernel_pos_index = torch_cluster.fps(candidate_points, None, ratio,
                                         random_start=False)[:num_kernel_points]

    kernel_pos = candidate_points[kernel_pos_index]
    kernel_pos = torch.cat([torch.zeros(1, 3), kernel_pos], dim=0)

    return kernel_pos


class MessagePassingBlock(nn.Module):
    def __init__(self, block_cfg):
        super(MessagePassingBlock, self).__init__()

        self.num_kernel_points = block_cfg.get("NUM_KERNEL_POINTS", 16)
        self.num_act_kernels = block_cfg.get("NUM_ACT_KERNELS", 3)
        self.radius = block_cfg.get("RADIUS", 1.0)
        self.kernel_loc = block_cfg.get("KERNEL_LOC", "IN_BALL")

        # construct kernel positions
        if self.kernel_loc == "BALL":
            kernel_pos = compute_ball_positions(self.num_kernel_points) * self.radius
        elif self.kernel_loc == "SPHERE":
            kernel_pos = compute_sphere_positions(self.num_kernel_points) * self.radius
        else:
            raise NotImplementedError;
        self.register_buffer('kernel_pos', kernel_pos, persistent=True)

        kernel_weights = torch.randn(self.num_kernel_points, input_channel, output_channel)
        nn.init.xavier_normal_(kernel_weights)
        self.kernel_weights = nn.Parameter(kernel_weights, requires_grad=True)

        self.norm = nn.BatchNorm1d(output_channel)
        self.aggr = 'mean'

    def forward(self, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query):
        """
        Args:
            ref_bxyz [N, 4]
            ref_feat [N, C]
            query_bxyz [M, 4]
            e_ref, e_query: point index of (ref, query), representing edges
        Returns:
            query_feat [M, C_out]
        """
        query_feat = message_passing(self.kernel_weights, self.kernel_pos,
                                     ref_bxyz, ref_feat,
                                     query_bxyz, e_ref, e_query,
                                     self.num_act_kernels)
        query_feat = self.norm(query_feat)
        
        return query_feat

    def __repr__(self):
        return f"""GraphConvBlock(
                       num_kernel_points={self.num_kernel_points},
                       act_kernel={self.num_act_kernels},
                       radius={self.radius},
                       norm={self.norm}
                   )"""

