import torch
from .virtual_array_cuda import (
    virtual_scatter_add_gpu,
    virtual_outer_and_sum_gpu
)

def virtual_scatter_add(core_array_a, virtual_index_a, virtual_weight_a, index_b, dim_b):
    """
    Args:
        core_array_a [U, D]
        virtual_index_a [E] in range [U]
        virtual_weight_a [E] scalars
        index_b [E] in range [Q]
        dim_b: Q
    Returns:
        res_b [Q, D] scattered results
    """
    return virtual_scatter_add_gpu(core_array_a, virtual_index_a, virtual_weight_a, index_b, dim_b)

def virtual_outer_and_sum(core_array_a, virtual_index_a, core_array_b, virtual_index_b, edge_weight):
    """
    Args:
        core_array_a [U1, D1]
        virtual_index_a [E] in range [U1]
        core_array_b [U2, D2]
        virtual_index_b [E] in range [U2]
        virtual_weight [E] scalars
    Returns:
        res_b [D1, D2] the sum of edge-wise weighted outer product
    """

    res = virtual_outer_and_sum_gpu(core_array_a, virtual_index_a, core_array_b, virtual_index_b, edge_weight)
    #res_v = (core_array_a[virtual_index_a].unsqueeze(-1) @ core_array_b[virtual_index_b].unsqueeze(-2)) 
    #res_v = (res_v * edge_weight[:, None, None]).sum(dim=0)
    #assert (res - res_v).abs().max() < 1e-4
    return res
