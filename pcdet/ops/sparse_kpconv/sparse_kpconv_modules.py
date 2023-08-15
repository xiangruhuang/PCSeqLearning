import torch
from .sparse_kpconv_cuda import (
    sparse_kpconv_gpu,
    tensor_outer_gpu
)

from torch.autograd import Function, Variable
from torch_scatter import scatter

BATCH_SIZE = 2048
def batched_einsum(X, W, A):
    """
    Args:
        X [N, D1]
        W [K, D2, D1]
        A [N, K]

    Returns:
        Y [N, D2]
    """
    N = X.shape[0]
    Y = X.new_zeros(N, W.shape[1])
    bs = BATCH_SIZE
    num_batch = (N + bs - 1) // bs
    with torch.no_grad():
        for b in range(num_batch):
            X_b = X[b*bs:(b+1)*bs, :].contiguous() # [B, D_in]
            A_b = A[b*bs:(b+1)*bs, :].contiguous() # [B, K]
            Y_b = torch.einsum('bi,koi,bk->bo', X_b, W, A_b)
            Y[b*bs:(b+1)*bs] = Y_b
    return Y

def batched_sparse_einsum(X, W, A, K_act):
    """
    Args:
        X [N, D1]
        W [K, D2, D1]
        A [N, K]
        K_act (int)

    Returns:
        Y [N, D2]
    """
    N = X.shape[0]
    D2, D1 = W.shape[1], X.shape[1]
    K = A.shape[1]
    Y = X.new_zeros(N, W.shape[1])
    with torch.no_grad():
        _, A_indices = torch.sort(A, dim=-1, descending=True)
        A_indices = A_indices[:, :K_act].contiguous()
        for k in range(K):
            mask = (A_indices == k).any(-1) # [N] -> [N_act]
            mask = mask & (A[:, k] > 1e-4) # [N]
            if mask.any():
                XWT = (X[mask] @ W[k].T) #  [N_act, D1] @ [D1, D2] = [N_act, D2]
                Y[mask] += XWT * A[mask, k].unsqueeze(-1) # [N_act, D2]
    return Y

def batched_tensor_outer(T0, T1, T2):
    """
    Args:
        T0 [N, D0]
        T1 [N, D1]
        T2 [N, D2]

    Returns:
        out [D0, D1, D2]
    """
    N = T0.shape[0]
    D0, D1, D2 = T0.shape[1], T1.shape[1], T2.shape[1]
    out = T0.new_zeros(D0, D1, D2)
    bs = BATCH_SIZE
    num_batch = (N + bs - 1) // bs
    with torch.no_grad():
        for b in range(num_batch):
            T0_b = T0[b*bs:(b+1)*bs, :].contiguous() # [B, D0]
            T1_b = T1[b*bs:(b+1)*bs, :].contiguous() # [B, D1]
            T2_b = T2[b*bs:(b+1)*bs, :].contiguous() # [B, D2]
            out += torch.einsum('ni,nj,nk->ijk', T0_b, T1_b, T2_b)
    return out

def batched_sparse_tensor_outer(T0, T1, T2, K_act):
    """
    Args:
        T0 [N, K]
        T1 [N, D1]
        T2 [N, D2]
        K_act (int)

    Returns:
        out [D0, D1, D2]
    """
    N = T0.shape[0]
    K, D1, D2 = T0.shape[1], T1.shape[1], T2.shape[1]
    out = T0.new_zeros(K, D1, D2)
    _, T0_indices = torch.sort(T0, dim=-1, descending=True)
    T0_indices = T0_indices[:, :K_act].contiguous()
    with torch.no_grad():
        for k in range(K):
            mask = (T0_indices == k).any(-1) # [N] -> [N_act]
            mask = mask & (T0[:, k] > 1e-4) # [N]
            if mask.any():
                T0_k = T0[mask, k].contiguous() # [N_act]
                T1_k = T1[mask].contiguous() # [N_act, D1]
                T2_k = T2[mask].contiguous() # [N_act, D2]
                out[k] = (T0_k[:, None] * T1_k).T @ T2_k
    return out

class SparseKPConv(Function):

    @staticmethod
    def forward(ctx, X, W, A, K_act):
        """
        Args:
            X [N, D_in] input features
            W [K, D_out, D_in] kernel parameters
            A [N, K] aggregation weights
            K_act (int) if smaller than K, use kernels with top-K_act weights

        Returns:
            Y [N, D_out]
        """ 
        K = W.shape[0]
        assert K_act <= K, "inappropriate num_act_kernel_points"
        if K_act == K:
            Y = batched_einsum(X, W, A)
        else:
            Y = batched_sparse_einsum(X, W, A, K_act)

        ctx.K_act = K_act
        ctx.save_for_backward(X, W, A)

        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        """
        Args:
            grad_Y [N, D_out]
        
        Returns:
            grad_X [N, D_in]
            grad_W [K, D_out, D_in]
        """
        X, W, A = ctx.saved_tensors
        K_act = ctx.K_act
        K = W.shape[0]
        grad_X = grad_W = grad_a = grad_K = None
        
        if ctx.needs_input_grad[0]:
            W_T = W.transpose(1, 2).contiguous()
            if K_act == K:
                grad_X = batched_einsum(grad_Y, W_T, A)
            else:
                grad_X = batched_sparse_einsum(grad_Y, W_T, A, K_act)

        if ctx.needs_input_grad[1]:
            if K_act == K:
                grad_W = batched_tensor_outer(A, grad_Y, X) # [N, K], [N, D2], [N, D1]
            else:
                grad_W = batched_sparse_tensor_outer(A, grad_Y, X, K_act) # [N, K], [N, D2], [N, D1]

        assert not ctx.needs_input_grad[2], "alpha should not need gradient"
        assert not ctx.needs_input_grad[3], "K_act should not need gradient"

        return grad_X, grad_W, grad_a, grad_K
        

sparse_kpconv_aggr = SparseKPConv.apply

if __name__ == '__main__':
    if True:
        N = 160000
        K = 15
        D1 = 512
        D2 = 512
        K_act = 3
        X = torch.nn.Parameter((torch.rand(N, D1)*1e-3).cuda(), requires_grad=True)
        W = torch.nn.Parameter((torch.rand(K, D2, D1)*1e-3).cuda(), requires_grad=True)
        optimizer = torch.optim.Adam([X, W], lr=1e-4)
        a = torch.rand(N, K).cuda()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            for itr in range(5):
                optimizer.zero_grad()
                loss = sparse_kpconv_aggr(X, W, a, K_act).square().sum() / 2.0
                #loss = torch.einsum('ni,koi,nk->no', X, W, a).square().sum() / 2.0
                loss.backward()
                optimizer.step()
        print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
    
    
    if True:
        N = 3
        K = 5
        D1 = 4
        D2 = 5
        K_act = 2
        X = torch.nn.Parameter((torch.rand(N, D1).double()).cuda(), requires_grad=True)
        W = torch.nn.Parameter((torch.rand(K, D2, D1).double()).cuda(), requires_grad=True)
        a = torch.rand(N, K).double().cuda()
        l0 = sparse_kpconv_aggr(X, W, a, K_act).square().sum() / 2.0
        l0.backward()
        eps = 1e-5
        for n in range(N):
            for i in range(D1):
                X.data[n, i] += eps
                l1 = sparse_kpconv_aggr(X, W, a, K_act).square().sum() / 2.0
                pred_delta = X.grad[n, i]
                gt_delta = (l1 - l0)/eps
                diff = (gt_delta - pred_delta).abs()
                assert diff < 1e-3, f'{diff}'
                #print(f'pass, {pred_delta:.6f}, {gt_delta:.6f}')
                X.data[n, i] -= eps
        
        for k in range(K):
            for i in range(D2):
                for j in range(D1):
                    W.data[k, i, j] += eps
                    l1 = sparse_kpconv_aggr(X, W, a, K_act).square().sum() / 2.0
                    pred_delta = W.grad[k, i, j]
                    gt_delta = (l1 - l0)/eps
                    diff = (gt_delta - pred_delta).abs()
                    try:
                        assert diff < 1e-3, f'{diff}, {pred_delta}, {gt_delta}'
                    except Exception as e:
                        import ipdb; ipdb.set_trace()
                        print(e)
                    #print(f'pass, {pred_delta:.6f}, {gt_delta:.6f}')
                    W.data[k, i, j] -= eps
                
