#ifndef TORCH_HASH_H
#define TORCH_HASH_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

typedef unsigned long long int uint64;
typedef long long int int64;
typedef unsigned int uint32;
typedef int64_t Key;
typedef int64_t index_t;
typedef float Float;

at::Tensor virtual_scatter_add_gpu(at::Tensor core_array_a,
                                   at::Tensor virtual_index_a,
                                   at::Tensor virtual_weight_a,
                                   at::Tensor index_b,
                                   int b_size);

at::Tensor virtual_outer_and_sum_gpu(at::Tensor core_array_a,
                                     at::Tensor virtual_index_a,
                                     at::Tensor core_array_b,
                                     at::Tensor virtual_index_b,
                                     at::Tensor virtual_weight);

#endif
