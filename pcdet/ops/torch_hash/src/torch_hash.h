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

void hash_insert_gpu(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                     at::Tensor dims, at::Tensor insert_keys, at::Tensor insert_values
                     );

void correspondence(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                    at::Tensor dims, at::Tensor query_keys, at::Tensor query_values,
                    at::Tensor qmin, at::Tensor qmax, at::Tensor corres_indices);

torch::Tensor radius_graph_gpu(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                               at::Tensor dims, at::Tensor query_keys, at::Tensor query_values,
                               at::Tensor qmin, at::Tensor qmax,
                               at::Tensor radius, int max_num_neighbors, bool sort_by_dist);

void points_in_radius_gpu(at::Tensor keys, at::Tensor values, at::Tensor reverse_indices,
                          at::Tensor dims, at::Tensor query_keys, at::Tensor query_values,
                          at::Tensor qmin, at::Tensor qmax,
                          Float radius_2d, at::Tensor visited);

#endif
