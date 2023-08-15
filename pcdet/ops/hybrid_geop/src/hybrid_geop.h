#ifndef HYBRID_GEOP_H
#define HYBRID_GEOP_H

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

void hash_insert_gpu(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                     at::Tensor dims, at::Tensor insert_keys, at::Tensor insert_values
                     );

void hybrid_geop_gpu(at::Tensor keys,
                     at::Tensor values,
                     at::Tensor reverse_indices,
                     at::Tensor dims,
                     at::Tensor qmin,
                     at::Tensor qmax,
                     at::Tensor query_keys,
                     at::Tensor mu,
                     at::Tensor sigma,
                     Float decay_radius);

void svd3_gpu(at::Tensor input,
              at::Tensor output_u,
              at::Tensor output_s,
              at::Tensor output_v);

#endif
