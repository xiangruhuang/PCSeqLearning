from torch import nn
import torch
from torch import Tensor
import numpy as np
from .kpconv_utils import load_kernel_points
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Union
from ...ops.sparse_kpconv import sparse_kpconv_aggr

class KPConvLayer(MessagePassing):
    _INFLUENCE_TO_RADIUS = 1.5
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_influence_dist,
                 num_kernel_points=15,
                 num_act_kernel_points=6,
                 fixed="center",
                 KP_influence="linear",
                 aggr_mode="max",
                 add_one=False,
                ):
        super(KPConvLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_influence_dist = kernel_influence_dist
        self.num_kernel_points = num_kernel_points
        self.fixed = fixed
        self.KP_influence = KP_influence
        self.aggr = aggr_mode
        self.add_one = add_one
        self.kernel_radius = kernel_influence_dist * self._INFLUENCE_TO_RADIUS

        K_points_numpy = load_kernel_points(
                             self.kernel_radius, num_kernel_points, 
                             num_kernels=1, dimension=3, fixed=fixed
                         ).reshape((num_kernel_points, 3))
        self.num_act_kernel_points = min(num_act_kernel_points, num_kernel_points)
        K_points_torch = torch.from_numpy(K_points_numpy).to(torch.float)
        self.K_points = nn.Parameter(K_points_torch, requires_grad=False)
        weights = torch.empty([num_kernel_points, self.output_channels, self.input_channels], dtype=torch.float)
        torch.nn.init.xavier_normal_(weights)
        self.weight = nn.Parameter(weights)

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, ref, query, edge_index, x):
        """
        Args:
            ref [N, 3]
            query [M, 3]
            edge_index [2, E] from ref to query
            x [N, D] features

        Returns:
            x_out [M, D_out] output features

        """
        assert self.flow == "source_to_target"

        pos = (ref, query) # (source, target)
        x_out = self.propagate(edge_index, x=x, pos=pos)
        return x_out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                      pos_j: Tensor) -> Tensor:
        """ subscript i refers to query, subscript j refers to ref
        message flows from ref to query (or j to i)

        Args:
            x_j [E, D] features from source (or ref)
            pos_i [E, 3] query points
            pos_j [E, 3] ref points

        Returns:
            x_i [E, D_out]
        """
        rel_pos = (pos_j - pos_i) # [E, 3]
        distance = (self.K_points - rel_pos[:, None, :]).norm(p=2, dim=-1) # [E, K] = ([K, 3] - [E, 1, 3]).norm(dim=-1)
        aggr_weights = torch.clamp(1 - distance / self.kernel_influence_dist, min=0.0) # [E, K]

        x_out = sparse_kpconv_aggr(x_j, self.weight, aggr_weights, self.num_act_kernel_points)

        return x_out

#def KPConvMP(
#    query_points,
#    support_points,
#    neighbors_indices,
#    features,
#    K_points,
#    K_values,
#    KP_radius,
#    KP_influence,
#    aggregation_mode,
#):
#    """ Kernel Point Convolution (Message Passing style)
#    
#    Args:
#        query_points [N, 4]: first dimension is batch index
#        ref_points [M, 4]: first dimension is batch index
#        neighbor_indices [2, E]: directed edges from `query_points` to `ref_points`
#        features [M, D]: point-wise feature vectors
#        K_points [K, 3]: kernel point locations
#        K_values [K, D, D_out]: kernel weight matrices
#        KP_radius (float): radius of kernels
#        KP_influence (should be 'linear'): determine the interpolation of weights
#        aggregation_mode (should be sum): how features are aggregated
#
#    Returns:
#        TODO
#    """
    
    

def KPConv_ops(
    query_points,
    support_points,
    neighbors_indices,
    features,
    K_points,
    K_values,
    KP_extent,
    KP_influence,
    aggregation_mode,
):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n0_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return:                    [n_points, out_fdim]
    """
    # Add a fake point in the last row for shadow neighbors
    shadow_point = torch.ones_like(support_points[:1, :]) * 1e6
    support_points = torch.cat([support_points, shadow_point], dim=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    import ipdb; ipdb.set_trace()
    neighbors = gather(support_points, neighbors_indices)

    # Center every neighborhood
    neighbors = neighbors - query_points.unsqueeze(1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors.unsqueeze_(2)
    differences = neighbors - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = torch.sum(differences ** 2, dim=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == "linear":
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / KP_extent, min=0.0)
        all_weights = all_weights.transpose(2, 1)
    else:
        raise ValueError("Unknown influence function type (config.KP_influence)")

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == "closest":
        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        all_weights *= torch.transpose(torch.nn.functional.one_hot(neighbors_1nn, K_points.shape[0]), 1, 2)

    elif aggregation_mode != "sum":
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = gather(features, neighbors_indices)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = torch.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = torch.sum(kernel_outputs, dim=0)

    return output_features
