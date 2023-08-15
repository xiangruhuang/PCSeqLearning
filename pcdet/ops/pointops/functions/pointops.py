from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

try:
    import pointops_cuda
except ImportError as err:
    import warnings
    import os
    from torch.utils.cpp_extension import load
    print(err)

    warnings.warn("Unable to load pointops_cuda cpp extension.")
    pointops_cuda_src = os.path.join(os.path.dirname(__file__), "../src")
    pointops_cuda = load('pointops_cuda', [
        pointops_cuda_src + '/pointops_api.cpp',
        pointops_cuda_src + '/knnquery/knnquery_cuda.cpp',
        pointops_cuda_src + '/knnquery/knnquery_cuda_kernel.cu',
        pointops_cuda_src + '/interpolation/interpolation_cuda.cpp',
        pointops_cuda_src + '/interpolation/interpolation_cuda_kernel.cu',
        pointops_cuda_src + '/sampling/sampling_cuda.cpp',
        pointops_cuda_src + '/sampling/sampling_cuda_kernel.cu',
        pointops_cuda_src + '/subtraction/subtraction_cuda.cpp',
        pointops_cuda_src + '/subtraction/subtraction_cuda_kernel.cu',
        pointops_cuda_src + '/aggregation/aggregation_cuda.cpp',
        pointops_cuda_src + '/aggregation/aggregation_cuda_kernel.cu',
    ], build_directory=pointops_cuda_src, verbose=False)


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx

furthestsampling = FurthestSampling.apply


class SectorizedFurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset, num_sectors):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        # cut into batches
        last_offset = 0
        sizes = []
        new_sizes = []
        indices = []
        for i in range(offset.shape[0]):
            size = offset[i] - last_offset
            batch_xyz = xyz[last_offset:last_offset+size]
            angle = torch.atan2(batch_xyz[:, 0], batch_xyz[:, 1]) # [0, 2*pi]
            sector_range = torch.linspace(angle.min(), angle.max()+1e-4, num_sectors+1)
            for s in range(num_sectors):
                indices.append(
                            torch.where((angle >= sector_range[s]) & (angle < sector_range[s+1]))[0] + last_offset
                        )
                sizes.append(indices[-1].shape[0])
            if i > 0:
                new_size = (new_offset[i] - new_offset[i-1]).item()
            else:
                new_size = new_offset[i].item()
            new_sizes_this_batch = [new_size // num_sectors for i in range(num_sectors)]
            new_sizes_this_batch[-1] += new_size % num_sectors
            new_sizes += new_sizes_this_batch
            last_offset = offset[i]
            
        sizes = torch.tensor(sizes, dtype=torch.long).to(offset)
        sector_offset = sizes.cumsum(dim=0)
        new_sizes = torch.tensor(new_sizes, dtype=torch.long).to(offset)
        new_sector_offset = new_sizes.cumsum(dim=0)
        indices = torch.cat(indices).long().to(offset.device)
        sector_xyz = xyz[indices].contiguous()
                
        # transform to sectors
        new_xyz = []
        n, b, n_max = sector_xyz.shape[0], sector_offset.shape[0], sector_offset[0]
        for i in range(1, b):
            n_max = max(sector_offset[i] - sector_offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_sector_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, sector_xyz, sector_offset.int(), new_sector_offset.int(), tmp, idx)
        idx = indices[idx.long()]
        del tmp
        del sector_xyz
        return idx

sectorized_fps = SectorizedFurthestSampling.apply

class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)

knnquery = KNNQuery.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.cuda.FloatTensor(m, nsample, c)
        pointops_cuda.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        import time
        t0 = time.time()
        n = ctx.n
        idx, = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        print('grouping', time.time()-t0)
        return grad_input, None

grouping = Grouping.apply


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat


class Subtraction(Function):
    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        input: input1: (n, c), input2: (n, c), idx: (n, nsample)
        output:  (n, nsample, c)
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape; nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        pointops_cuda.subtraction_forward_cuda(n, nsample, c, input1, input2, idx, output)
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, nsample, c)
        output: grad_input1: (n, c), grad_input2: (n, c)
        """
        import time
        t0 = time.time()
        idx, = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.subtraction_backward_cuda(n, nsample, c, idx, grad_output, grad_input1, grad_input2)
        print('subtraction', time.time()-t0)
        return grad_input1, grad_input2, None

subtraction = Subtraction.apply


class Aggregation(Function):
    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx: (n, nsample)
        output: (n, c)
        """
        assert input.is_contiguous() and position.is_contiguous() and weight.is_contiguous()
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.aggregation_forward_cuda(n, nsample, c, w_c, input, position, weight, idx, output)
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, c)
        output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight : (n, nsample, c')
        """
        import time
        t0 = time.time()
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        pointops_cuda.aggregation_backward_cuda(n, nsample, c, w_c, input, position, weight, idx, grad_output, grad_input, grad_position, grad_weight)
        print('aggregation', time.time()-t0)
        return grad_input, grad_position, grad_weight, None

aggregation = Aggregation.apply


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)
    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, k), (n, k)
        dist_recip = 1.0 / (dist + 1e-8) # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm # (n, k)

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        import time
        t0 = time.time()
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(m, c).zero_()
        pointops_cuda.interpolation_backward_cuda(n, c, k, grad_output, idx, weight, grad_input)
        print('interpolation', time.time()-t0)
        return None, None, grad_input, None, None, None

interpolation2 = Interpolation.apply
