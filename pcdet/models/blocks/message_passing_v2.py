import numpy as np
import torch
from torch import nn
from torch_scatter import scatter
from torch.autograd import Function
from tqdm import tqdm
import time

@torch.no_grad()
def kernel_dist(index, K0, D1, D2, E):
    """
    Args:
        index [E]
        K0, D1, D2, E: integers
    Returns:
        
    """
    t0 = time.time()
    index = index.long()
    degree = []
    for k in range(K0):
        degree.append((index == k).long().sum())
    degree = torch.stack(degree).to(index).view(-1)
    kernel_offset = degree.cumsum(dim=0) - degree
    
    # find best batch size B, [K0, D1, D2]->[K, D1, D2]
    best_batch_size = -1
    min_cost = 1e30
    _batch_size = 1
    max_degree = degree.max().long().item()
    #print('dist.1', time.time()-t0); t0 = time.time()
    while True:
        batch_size = np.round(_batch_size).astype(np.int64)
        if batch_size > max_degree:
            break
        num_duplicate_kernels = torch.ceil(degree / batch_size).clamp(min=1).sum().item()
        
        mem_cost = num_duplicate_kernels * D1 * D2 + num_duplicate_kernels * batch_size * (D1 + D2)
        compute_cost = num_duplicate_kernels * batch_size * D1 * D2
        cost = mem_cost + 0.01 * compute_cost
        if cost < min_cost:
            min_cost = cost
            best_batch_size = batch_size
        _batch_size *= 2 

    #print('dist.2', time.time()-t0); t0 = time.time()
    B = best_batch_size
    num_duplicates = torch.ceil(degree / B).long().clamp(min=1) # [1, 1, 2, 1]
    offset = num_duplicates.cumsum(dim=0) - num_duplicates # [1, 2, 4, 5] - [1, 1, 2, 1] = [0, 1, 2, 4]
    K = num_duplicates.sum().long().item() # number of kernels (include duplicate)
    
    # [0,1,2,4] -> [0, 1, 1, 0, 1].cumsum() -> [0, 1, 2, 2, 3]
    original_kernel_index = index.new_zeros(K)
    original_kernel_index[offset[1:]] = 1
    original_kernel_index = original_kernel_index.cumsum(dim=0)
    
    # [E] -> [K, B]
    unique_index, inverse = torch.sort(index)
    global_offset = index.new_zeros(E)
    global_offset[inverse] = torch.arange(E).to(index)
    pool_offset = (global_offset - kernel_offset[index]) + offset[index] * B
    
    # map from [K0] to [K]
    return original_kernel_index, pool_offset, K, B

def message_passing_naive(kernel, ref_feat, e_kernel, e_ref, e_query, num_queries, e_weight=None):
    import dgl
    if e_weight is None:
        edge_feat = ref_feat[e_ref]
    else:
        edge_feat = ref_feat[e_ref] * e_weight.view(-1, 1)
    edge_feat = dgl.ops.gather_mm(edge_feat, kernel, idx_b = e_kernel.int())
        
    query_feat = scatter(edge_feat, e_query, dim=0,
                         dim_size=num_queries, reduce='sum')

    return query_feat

def map_to_pool(data, e, w, pool_index, l):
    e_pool = torch.arange(l, device=pool_index.device, dtype=pool_index.dtype) % data.shape[0]
    e_pool[pool_index] = e
    e_pool_mask = torch.zeros(e_pool.shape, dtype=torch.bool, device=e_pool.device)
    e_pool_mask[pool_index] = True
    pool = data[e_pool]
    pool[~e_pool_mask] = 0
    if w is not None:
        pool[pool_index] *= w.view(-1, 1)
    if False: # debug
        pool2 = scatter(data[e], pool_index, dim=0, dim_size=l, reduce='sum')
        assert (pool - pool2).abs().sum() < 1e-3

    return pool

def pool_gemm(kernel, original_kernel_index,
              ref_feat, e_ref, e_query,
              pool_index, K, B, num_queries, e_weight=None):
    """
    Args:
        kernel [K0, D1, D2]
        original_kernel_index [K]: in range [K0]
        ref_feat [N, D1]
        e_ref, e_query [E]: edges from ref to query
        pool_index [E]: map edge to pool
        K: number of duplicate kernels
        B: batch size of each kernel
    Returns:
        query_feat: [M, D2]
    """
    
    dup_kernel = kernel[original_kernel_index] # [K, D1, D2]
    
    pool = map_to_pool(ref_feat, e_ref, e_weight, pool_index, K*B).view(K, B, -1)

    query_pool = pool @ dup_kernel # [K, B, D2]
    del pool, dup_kernel # save memory
    e_query_pool = torch.arange(K*B, device=pool_index.device, dtype=pool_index.dtype) % num_queries
    e_query_pool[pool_index] = e_query

    query_edge_feat = query_pool.view(K*B, -1)
    query_feat = scatter(query_edge_feat, e_query_pool, dim=0,
                         dim_size=num_queries, reduce='sum')

    return query_feat


class MessagePassing(Function):

    @staticmethod
    def forward(ctx, kernel, ref_feat, e_kernel, e_ref, e_query, num_queries,
                dist_info=None, e_weight=None):
        """
        Args:
            kernel [K, D1, D2]: kernel weights
            ref_feat [N, D1]: source features
            e_kernel [E]: (in range [K]) edge associated kernel index
            e_ref [E]: (in range [N]) edge source endpoints
            e_query [E]: (in range [M]) edge target endpoints

        Returns:
            query_feat: [M, D2] 
        """
        assert e_ref.shape[0] == e_kernel.shape[0]
        K0, D1, D2 = list(kernel.shape)
        E = e_kernel.shape[0]
        if dist_info is None:
            original_kernel_index, pool_index, K, B = \
                    kernel_dist(e_kernel, K0, D1, D2, E)
        else:
            original_kernel_index, pool_index, K, B = dist_info

        query_feat = pool_gemm(kernel, original_kernel_index,
                               ref_feat, e_ref, e_query,
                               pool_index, K, B, num_queries, e_weight)

        ctx.save_for_backward(kernel, ref_feat, e_kernel, e_ref, e_query, e_weight, original_kernel_index, pool_index)
        ctx.K = K
        ctx.B = B

        return query_feat

    @staticmethod
    def backward(ctx, grad_query_feat):
        """
        Args:
            grad_query_feat [M, D2] gradient of query features

        Returns:
            grad_ref_feat [N, D1] gradient to ref features
            grad_kernel [K, D1, D2] gradient to kernel weights
        """
        kernel, ref_feat, e_kernel, e_ref, e_query, e_weight, \
                original_kernel_index, pool_index = ctx.saved_tensors
        K = ctx.K
        B = ctx.B
        num_refs = ref_feat.shape[0]

        K0, D1, D2 = list(kernel.shape)
        E = e_kernel.shape[0]

        # gradient of input feature
        grad_ref_feat = pool_gemm(kernel.transpose(1, 2), original_kernel_index,
                                  grad_query_feat, e_query, e_ref,
                                  pool_index, K, B, num_refs, e_weight)

        # gradient of kernel
        pool_ref = map_to_pool(ref_feat, e_ref, e_weight, pool_index, K*B).view(K, B, -1).transpose(1, 2) # [K, D1, B]
        pool_query = map_to_pool(grad_query_feat, e_query, None, pool_index, K*B).view(K, B, -1) # [K, B, D2]
        
        grad_dup_kernel = pool_ref @ pool_query # [K, D1, D2]
        grad_kernel = scatter(grad_dup_kernel, original_kernel_index, dim=0,
                              dim_size=K0, reduce='sum') # [K0, D1, D2]

        return grad_kernel, grad_ref_feat, None, None, None, None, None, None

message_passing = MessagePassing.apply

def initialize_kernel_weight(input_channel, output_channel, num_kernels):
    kernel_weights = nn.Parameter(torch.randn(num_kernels, input_channel, output_channel), requires_grad=True)
    fan_in = input_channel * num_kernels
    gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
    std = gain / np.sqrt(fan_in)
    bound = np.sqrt(3) * std
    with torch.no_grad():
        return kernel_weights.uniform_(-bound, bound)


class MessagePassingBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_kernels, indice_key):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_weights = initialize_kernel_weight(input_channel, output_channel, num_kernels)
        self.key = indice_key
        self.D1 = input_channel
        self.D2 = output_channel
        self.K0 = num_kernels

    def forward(self, ref_feat, e_kernel, e_ref, e_query, num_queries, conv_dict, e_weight=None):
        if f'{self.key}_dist' in conv_dict:
            dist_info = conv_dict[f'{self.key}_dist']
        else:
            dist_info = kernel_dist(e_kernel, self.K0, self.D1, self.D2, e_ref.shape[0])
            conv_dict[f'{self.key}_dist'] = dist_info

        output = message_passing(self.kernel_weights, ref_feat, e_kernel,
                                 e_ref, e_query, num_queries, dist_info, e_weight)

        return output, conv_dict

    def extra_repr(self):
        return f"num_input={self.input_channel}, \nnum_output={self.output_channel}, \nindice_key={self.key}"


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
    mlp = nn.Parameter(torch.randn(K, channels[0], channels[-1]).double().cuda(), requires_grad=True)

    ref_feat = torch.nn.Parameter(torch.randn(N, d1).cuda().double(), requires_grad=True)
    e_ref = torch.arange(N).repeat(deg).long().cuda()
    e_query = torch.from_numpy(np.random.permutation(N*deg) % N).long().cuda()
    e_kernel = torch.arange(N*deg).long().cuda() % K
    e_weight = torch.randn(N*deg).double().cuda().clamp(-1, 1)
    query_feat = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query, N, None, e_weight)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        query_feat = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query, N, None, e_weight)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    if True:
        with torch.autograd.profiler.profile(use_cuda=True) as prof2: 
            query_feat2 = message_passing_naive(mlp, ref_feat, e_kernel, e_ref, e_query, N, e_weight)
        print(prof2.key_averages().table(sort_by="self_cuda_time_total"))
        assert (query_feat - query_feat2).abs().max() < 1e-5
    
    loss = query_feat.sum()
    loss.backward()

    grad_mlp = mlp.grad
    grad_ref_feat = ref_feat.grad
    eps = 1e-5
    for k in tqdm(range(mlp.shape[0])):
        for i in tqdm(range(mlp.shape[0])):
            for j in tqdm(range(mlp.shape[1])):
                mlp[k, i, j].data += eps
                #query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
                query_feat1 = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query, N, None, e_weight)
                loss1 = query_feat1.sum()
                grad_ij = (loss1 - loss) / eps
                assert (grad_ij - grad_mlp[k, i, j]).abs() < 1e-4, f"{grad_ij}, {grad_mlp[i, j]}"
                mlp[k, i, j].data -= eps
    print('done with mlp testing')

    for i in tqdm(range(0, ref_feat.shape[0], 10)):
        for j in tqdm(range(ref_feat.shape[1])):
            ref_feat.data[i, j] += eps
            #query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
            query_feat1 = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query, N, None, e_weight)
            loss1 = query_feat1.sum()
            grad_ij = (loss1 - loss) / eps
            assert (grad_ij - grad_ref_feat[i, j]).abs() < 1e-4, f"{grad_ij}, {grad_ref_feat[i, j]}"
            ref_feat.data[i, j] -= eps

    print('done with ref_feat testing')
