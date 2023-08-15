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

void sparse_kpconv_gpu(at::Tensor input, at::Tensor kernel, at::Tensor weight,
                       at::Tensor output);

void tensor_outer_gpu(at::Tensor t0, at::Tensor t1, at::Tensor t2,
                      at::Tensor output);

#endif
