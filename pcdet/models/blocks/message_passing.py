import torch
from torch import nn
from torch_scatter import scatter
from torch.autograd import Function
import numpy as np
from tqdm import tqdm
from pcdet.ops.torch_hash.torch_hash_modules import (
    RadiusGraph,
)
from pcdet.ops.virtual_array.virtual_array_utils import (
    virtual_scatter_add,
    virtual_outer_and_sum
)
import torch_cluster
import dgl
import time

BATCHSIZE=16384
BUFFERSIZE=2**30

def dist2weight(dist):
    edge_weight = 1.0 / (dist + 1e-5) # [E, act_k]
    edge_weight_sum = edge_weight.sum(-1, keepdim=True) # [E, 1]
    edge_weight = edge_weight / edge_weight_sum # [E, act_k]

    return edge_weight

def get_batch_size(S):
    batchsize = BATCHSIZE
    while (batchsize > 1) and (batchsize * S > BUFFERSIZE):
        batchsize = batchsize // 2
    return batchsize

def multi_dim_inc(unique_ref, w, ref_idx, query_idx, query):
    """scatter a to b, yet a is represented by referencing a unique array
        query += scatter(unique_ref[ref_idx] * w, query_idx)
    Args:
        unique_ref [U, D2]: unique
        w [E]: weight scalars multiplied to a ref vector
        ref_idx [E]: in range [U]
        query_idx [E]: in range [Q]
        query [Q, D2]: query idx in range
        
    Returns:
        query [Q, D2]
    """
    query = multiply(unique_ref, w, ref_idx, query_idx, query)
    return query

class MessagePassing(Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, kernel_weights, kernel_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, num_act_kernels):
        """Compute message passing, save memory.
        Args:
            kernel_weights [K, D1, D2]
            kernel_pos [K, 3]
            ref_bxyz [N, 4]
            ref_feat [N, D1]
            query_bxyz [M, 4]
            e_ref [E]
            e_query [E]
            num_act_kernels: act_k
        Returns:
            query_feat [M, D2]
        """

        t0 = time.time()
        pos_diff = (ref_bxyz[e_ref] - query_bxyz[e_query])[:, 1:4] # [E, act_k]
        e_edge, e_kernel = torch_cluster.knn(kernel_pos, pos_diff, num_act_kernels) # [E*act_k], [E*act_k]
        e_edge, e_kernel = e_edge.view(-1, num_act_kernels), e_kernel.view(-1, num_act_kernels) # [E, act_k]
        dist = (pos_diff[e_edge] - kernel_pos[e_kernel]).norm(p=2, dim=-1) # [E, act_k]

        # (e_ref, e_query, e_kernel, e_weight) relationships
        e_ref, e_query, e_kernel = e_ref[e_edge].view(-1), e_query[e_edge].view(-1), e_kernel.view(-1) # [E*act_k]
        e_weight = dist2weight(dist).view(-1) # [E*act_k]

        num_refs, num_queries = ref_bxyz.shape[0], query_bxyz.shape[0]
        num_edges = e_ref.shape[0]
        num_kernels = kernel_weights.shape[0]
        batchsize = num_refs # get_batch_size(num_edges // num_refs * max(kernel_weights.shape[1], kernel_weights.shape[2]))
        num_batches = (num_refs + batchsize - 1) // batchsize

        query_feat = ref_feat.new_zeros(num_queries, kernel_weights.shape[-1])

        for b in range(num_batches):
            start = b * batchsize
            end = min((b+1)*batchsize, num_refs)
            mask = (e_ref >= start) & (e_ref < end) # [Eb]
            comb_indices = e_kernel[mask] * num_refs + e_ref[mask]
            unique_indices, inverse = comb_indices.unique(return_inverse=True, sorted=True)
            unique_refs = unique_indices % num_refs
            unique_kernel = unique_indices // num_refs
            
            kernel_degree = unique_kernel.new_zeros(num_kernels)
            unique_kernel, kernel_counts = unique_kernel.unique(return_counts=True)
            kernel_degree[unique_kernel] = kernel_counts
            
            unique_feat = dgl.ops.segment_mm(ref_feat[unique_refs],
                                             kernel_weights,
                                             seglen_a = kernel_degree.cpu()
                                            ) # [U, D2]
            
            query_feat_b = virtual_scatter_add(unique_feat.float(), inverse, e_weight[mask].float(), e_query[mask], num_queries).to(ref_feat.dtype)

            query_feat += query_feat_b
        
        ctx.save_for_backward(kernel_weights, kernel_pos, ref_feat,
                              e_ref, e_query, e_weight, e_edge, e_kernel)

        return query_feat

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_query_feat):
        """
        Args:
            grad_query_feat [M, D2]

        Returns:
            grad_mlp [D1, D2]
            grad_ref_feat [N, D1]
        """
        t0 = time.time()
        kernel_weights, kernel_pos, ref_feat, \
                e_ref, e_query, e_weight, e_edge, e_kernel = ctx.saved_tensors
        num_refs, num_queries = ref_feat.shape[0], grad_query_feat.shape[0]
        num_kernels = kernel_weights.shape[0]
        num_edges = e_ref.shape[0] # [E, act_k]
        batchsize = num_queries
        #batchsize = get_batch_size(num_edges // num_queries * max(kernel_weights.shape[1], kernel_weights.shape[2]))
        num_batches = (num_queries + batchsize - 1) // batchsize # M // batchsize

        grad_ref_feat = ref_feat.new_zeros(num_refs, kernel_weights.shape[-2])
        kernel_weights_T = kernel_weights.transpose(1, 2)
        for b in range(num_batches):
            start = b * batchsize
            end = min((b+1)*batchsize, num_refs)
            mask = (e_query >= start) & (e_query < end) # [Eb]
            if not mask.any():
                continue
            comb_indices = e_kernel[mask] * num_queries + e_query[mask]
            unique_indices, inverse = comb_indices.unique(return_inverse=True, sorted=True)
            unique_queries = unique_indices % num_queries
            unique_kernel = unique_indices // num_queries
            
            kernel_degree = unique_kernel.new_zeros(num_kernels)
            unique_kernel, kernel_counts = unique_kernel.unique(return_counts=True)
            kernel_degree[unique_kernel] = kernel_counts
            
            unique_grad_ref_feat = dgl.ops.segment_mm(grad_query_feat[unique_queries],
                                                      kernel_weights_T,
                                                      seglen_a = kernel_degree.cpu()
                                                     ) # [U, D1]
            grad_ref_feat_b = virtual_scatter_add(unique_grad_ref_feat, inverse, e_weight[mask], e_ref[mask], num_refs)
            grad_ref_feat += grad_ref_feat_b

        del grad_ref_feat_b

        grad_kernel_weights = []
        for k in range(num_kernels):
            mask = (e_kernel == k)
            grad_kernel_weights_k = torch.zeros_like(kernel_weights[0])
            if mask.any():
                e_ref_k = e_ref[mask]
                e_query_k = e_query[mask]
                e_weight_k = e_weight[mask]
                bs = get_batch_size(max(kernel_weights.shape[-1], kernel_weights.shape[-2]))
                num_b = (e_ref_k.shape[0] + bs - 1) // bs
                for bk in range(num_b):
                    grad_kernel_weights_k += (ref_feat[e_ref_k[bk*bs:(bk+1)*bs]].T * e_weight_k[bk*bs:(bk+1)*bs]) @ grad_query_feat[e_query_k[bk*bs:(bk+1)*bs]]
            grad_kernel_weights.append(grad_kernel_weights_k)
        grad_kernel_weights = torch.stack(grad_kernel_weights, dim=0) # [K, D1, D2]


        return grad_kernel_weights, None, None, grad_ref_feat, None, None, None, None

message_passing = MessagePassing.apply


# this function is only used for debug
def message_passing_naive(kernel_weights, kernel_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, num_act_kernels):
    """Compute message passing, save memory.
    Args:
        kernel_weights [K, D1, D2]
        kernel_pos [K, 3]
        ref_bxyz [N, 4]
        ref_feat [N, D1]
        query_bxyz [M, 4]
        e_ref [E]
        e_query [E]
        num_act_kernels: act_k
    Returns:
        query_feat [M, D2]
    """
    # compute edge weights
    pos_diff = (ref_bxyz[e_ref] - query_bxyz[e_query])[:, 1:4] # [E, act_k]
    e_edge, e_kernel = torch_cluster.knn(kernel_pos, pos_diff, num_act_kernels) # [E*act_k], [E*act_k]
    e_edge, e_kernel = e_edge.view(-1, num_act_kernels), e_kernel.view(-1, num_act_kernels) # [E, act_k]
    dist = (pos_diff[e_edge] - kernel_pos[e_kernel]).norm(p=2, dim=-1) # [E, act_k]
    weight = dist2weight(dist) # [E, act_k]

    query_feat = None
    
    kernel_weights_ = []

    for g in range(weight.shape[-1]):
        e_kernel_g = e_kernel[:, g] # [E]
        edge_feat_g = dgl.ops.gather_mm(ref_feat[e_ref],
                                        kernel_weights,
                                        idx_b = e_kernel_g,
                                        ) * weight[:, g:(g+1)]
        if g == 0:
            edge_feat = edge_feat_g
        else:
            edge_feat += edge_feat_g
        #kernel_weights_.append(kernel_weights[e_kernel_g] * weight_g[:, None, None]) # [E, D1, D2] * [E]
    #edge_kernel_weights = torch.stack(kernel_weights_, dim=0).sum(0) # [E, D1, D2]
    #edge_feat = (ref_feat[e_ref].unsqueeze(-2) @ edge_kernel_weights).squeeze(1) # ([E, 1, D1] @ [E, D1, D2]).squeeze(1) = [E, D2]
    query_feat = scatter(edge_feat, e_query, dim=0, dim_size = query_bxyz.shape[0], reduce='sum')

    return query_feat

if __name__ == '__main__':
    #msg = MessagePassing(32, 128, 20).cuda()
    #feat = torch.randn(1500, 32).cuda()
    #e_query = torch.cat([torch.arange(1500), torch.arange(1500), torch.arange(1500)]).long().cuda()
    #e_ref = torch.cat([torch.arange(1500), torch.arange(1500) + 1, torch.arange(1500) + 2]) % 1500
    #e_ref = e_ref.long().cuda()

    #y = msg(bxyz, feat, bxyz, e_ref, e_query)
    #loss = y.sum()
    #loss.backward()

    d1 = 16
    d2 = 32
    K = 10
    N = 400
    deg = 10

    channels = [d1, d2]
    mlp = nn.Parameter(torch.randn(10, channels[0], channels[-1]).double().cuda(), requires_grad=True)
    mlp_pos = torch.randn(K, 3).double().cuda()

    ref_bxyz = torch.randn(N, 4).double().cuda()
    ref_bxyz[:, 0] = torch.randint(size=[N], high=2).double().cuda()
    query_bxyz = ref_bxyz
    ref_feat = torch.nn.Parameter(torch.randn(N, d1).cuda().double(), requires_grad=True)
    e_ref = torch.arange(N).repeat(deg).long().cuda()
    e_query = torch.from_numpy(np.random.permutation(N*deg) % N).long().cuda()
    query_feat = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)

    with torch.autograd.profiler.profile(use_cuda=True) as prof: 
        query_feat = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    if True:
        with torch.autograd.profiler.profile(use_cuda=True) as prof2: 
            query_feat2 = message_passing_naive(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
        print(prof2.key_averages().table(sort_by="self_cuda_time_total"))
        assert (query_feat - query_feat2).abs().max() < 1e-5
    
    #loss = query_feat.sum()
    #loss.backward()

    #grad_mlp = mlp.grad
    #grad_ref_feat = ref_feat.grad
    #eps = 1e-5
    #for k in tqdm(range(mlp.shape[0])):
    #    for i in tqdm(range(mlp.shape[0])):
    #        for j in tqdm(range(mlp.shape[1])):
    #            mlp[k, i, j].data += eps
    #            query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
    #            loss1 = query_feat1.sum()
    #            grad_ij = (loss1 - loss) / eps
    #            assert (grad_ij - grad_mlp[k, i, j]).abs() < 1e-4, f"{grad_ij}, {grad_mlp[i, j]}"
    #            mlp[k, i, j].data -= eps
    #print('done with mlp testing')

    #for i in tqdm(range(0, ref_feat.shape[0], 10)):
    #    for j in tqdm(range(ref_feat.shape[1])):
    #        ref_feat.data[i, j] += eps
    #        query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
    #        loss1 = query_feat1.sum()
    #        grad_ij = (loss1 - loss) / eps
    #        assert (grad_ij - grad_ref_feat[i, j]).abs() < 1e-4, f"{grad_ij}, {grad_ref_feat[i, j]}"
    #        ref_feat.data[i, j] -= eps

