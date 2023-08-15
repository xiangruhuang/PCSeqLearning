import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
import os
from torch_scatter import scatter, segment_coo

from pcdet.utils import common_utils
from pcdet.utils.timer import Timer

def robust_mean(data, index, max_index):
    data_shape = np.array(list(data.shape))
    data_shape[1:] = 1
    data_shape = list(data_shape)
    weight_sum = scatter(data.new_ones(data_shape), index, dim=0, dim_size=max_index, reduce='sum')
    data_sum = scatter(data, index, dim=0, dim_size=max_index, reduce='sum')
    valid_mask = weight_sum.reshape(-1) > 0.5
    data_sum[valid_mask] = data_sum[valid_mask] / weight_sum[valid_mask]
    return data_sum

def efficient_robust_mean(_data, _index, max_index, sort_index=None):
    """Use segment_coo.

    """
    if sort_index is None:
        sort_index = torch.argsort(_index)
    data = _data[sort_index]
    index = _index[sort_index]
    data_mean = segment_coo(data, index, reduce='mean', dim_size=max_index)
    return data_mean

def efficient_robust_sum(_data, _index, max_index, sort_index=None):
    """Use segment_coo.

    """
    if sort_index is None:
        sort_index = torch.argsort(_index)
    data = _data[sort_index]
    index = _index[sort_index]
    data_mean = segment_coo(data, index, reduce='sum', dim_size=max_index)
    return data_mean

def truncated_robust_mean(_data, _index, max_index, trunc_dist=0.3):
    # compute robust mean first.
    sort_index = torch.argsort(_index)
    data = _data[sort_index]
    index = _index[sort_index]
    data_mean = segment_coo(data, index, reduce='mean', dim_size=max_index)

    data = data.clamp(min=data_mean[index] - trunc_dist, max=data_mean[index] + trunc_dist)
    #invalid_mask = (data - data_mean[index]).abs() > trunc_dist
    #weight[invalid_mask] = 0.01
    #data_mean_sum = segment_coo(data*weight, index, reduce='sum', dim_size=max_index)
    #data_mean_w = segment_coo(weight, index, reduce='sum', dim_size=max_index)
    robust_mean = segment_coo(data, index, reduce='mean', dim_size=max_index)
    
    return robust_mean

def robust_median(data, index, max_index):
    """Deprecated. Too slow.

    """
    min_data, max_data = data.min(), data.max()
    range_data = (max_data - min_data + 1)
    index_data = index * range_data + (data - min_data)
    sorted_index_data, _ = torch.sort(index_data)
    sort_index = torch.argsort(index)
    sorted_index = torch.div(sorted_index_data, range_data, rounding_mode='trunc').long()
    sorted_data = sorted_index_data - range_data * sorted_index + min_data
    degree = segment_coo(torch.ones_like(index), sorted_index,
                         reduce='sum', dim_size=max_index)
    mask = degree > 0
    #degree = scatter(torch.ones_like(index), index, dim=0, dim_size=max_index, reduce='sum')
    retrieve_indices = degree.cumsum(0) - degree + degree // 2
    #retrieve_indices = offset + degree // 2
    median = data.new_zeros(max_index)
    median[mask] = sorted_data[retrieve_indices[mask]]
    median[~mask] = -1e10

    return median

def register_to_next_frame(graph, moving, ref, num_components,
                           angle_regularizer=10, max_iter=20,
                           stopping_delta=5e-2):
    """
    Args:
        graph:
        moving:
            component
            frame
            fxyz
            stationary
        ref:
            frame
            fxyz
            stationary

    Returns:
        moving:
        T [C, F, 4, 4]: transforms per component per frame
        reg_error [C]: registration error of each component
        comp_edge_ratio [C]: the fraction of points in each component 
                             that have a corresponding point.
    """
    # save
    temp_radius = graph.radius
    temp_qmin = graph.qmin
    temp_qmax = graph.qmax

    frame_offset = (ref.frame[0] - moving.frame[0]).long().item()
    graph.radius = (temp_radius**2 + frame_offset**2)**0.5
    
    component_deg = scatter(torch.ones_like(moving.component), moving.component,
                            dim=0, dim_size=num_components, reduce='sum')

    T = torch.diag_embed(moving.fxyz.new_ones(num_components, 4).double())

    # non-stationary parts
    ns_moving = EasyDict(common_utils.filter_dict(moving, ~moving.stationary))
    ns_ref = EasyDict(common_utils.filter_dict(ref, ~ref.stationary))

    last_error = 1e10
    countdown = 3
    for itr in range(max_iter):
        # set dynamic radius
        #if itr > 10:
        #    graph.radius = ((temp_radius/2)**2 + frame_offset**2)**0.5

        # forward direction, from moving to ref
        graph.qmin[0] = frame_offset
        graph.qmax[0] = frame_offset
        f_ref, f_moving, _ = graph(ns_ref, ns_moving)
        
        # backward direction, from ref to moving
        graph.qmin[0] = -frame_offset
        graph.qmax[0] = -frame_offset
        b_moving, b_ref, _ = graph(ns_moving, ns_ref)
        
        # each edge is a triplet that consists of
        # 1. source point index: represented by e_moving
        # 2. target point index: represented by e_ref
        # 3. component index: represented by e_component
        e_moving = torch.cat([f_moving, b_moving], dim=0)
        e_ref = torch.cat([f_ref, b_ref], dim=0)
        e_component = ns_moving.component[e_moving]

        # source and target center of each component.
        # not using scatter due to too much conflicts
        moving_center = efficient_robust_mean(ns_moving.fxyz[e_moving, 1:], e_component, num_components).double()
        ref_center = efficient_robust_mean(ns_ref.fxyz[e_ref, 1:], e_component, num_components).double()
        P = (ns_moving.fxyz[e_moving, 1:] - moving_center[e_component]).double()
        Q = (ns_ref.fxyz[e_ref, 1:] - ref_center[e_component]).double()
        dist = (P - Q).norm(p=2, dim=-1)

        l1_component_error = truncated_robust_mean(dist, e_component, num_components)
        #l1_component_error = robust_median(dist, e_component, num_components)
        #l1_component_error = scatter(dist, e_component, reduce='mean',
        #                             dim_size=num_components, dim=0)

        loss = dist.square().sum()
        
        H = P[:, :, None] * Q[:, None, :]
        cov = robust_mean(H, e_component, num_components)
        regularizer = T[:, :3, :3] * angle_regularizer

        U, S, VT = torch.linalg.svd(cov + regularizer)

        V = VT.transpose(1, 2)
        UT = U.transpose(1, 2)
        sign = torch.ones_like(S)
        sign[:, -1] = (V @ UT).det()
        R = V @ torch.diag_embed(sign) @ UT
        T_i = R.new_zeros(R.shape[0], 4, 4)
        T_i[:, :3, :3] = R
        T_i[:, :3, 3] = ref_center - (moving_center[:, None, :] @ R.transpose(1, 2))[:, 0, :]
        T_i[:, 3, 3] = 1.0
        T = T_i @ T
        ns_moving.fxyz[:, 1:] = (ns_moving.fxyz[:, None, 1:].double() @ R[ns_moving.component].transpose(1, 2))[:, 0, :] + T_i[ns_moving.component, :3, 3]
        if last_error - loss.item() < stopping_delta:
            countdown -= 1
        else:
            countdown = 3
        if (countdown <= 0):
            break
        last_error = loss.item()
        #print(f'iter={itr}, error={loss.item():.6f}, countdown={countdown}')

    # check the fraction of points that are not corresponding to anything (dropped edges)
    # forward direction, from moving to ref
    graph.qmin[0] = frame_offset
    graph.qmax[0] = frame_offset
    f_ref, f_moving, _ = graph(ref, ns_moving)

    # check how many edges are dropped.
    f_component = ns_moving.component[f_moving]
    component_edge_count = scatter(torch.ones_like(f_component), f_component,
                                   dim=0, dim_size=num_components, reduce='sum')
    comp_edge_ratio = component_edge_count / (component_deg.float() + 1e-6)

    # restore
    graph.radius = temp_radius
    graph.qmin[:] = temp_qmin
    graph.qmax[:] = temp_qmax
    moving.fxyz[~moving.stationary, 1:] = ns_moving.fxyz[:, 1:]
    return moving, T, l1_component_error, comp_edge_ratio
