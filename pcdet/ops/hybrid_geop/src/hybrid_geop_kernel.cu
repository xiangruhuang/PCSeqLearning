#include "cuda_runtime.h"
#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <time.h>

#include "hybrid_geop.h"
#include "svd3_cuda_device.h"

#define CHECK_CUDA(x) do { \
    if (!x.type().is_cuda()) { \
          fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
          exit(-1); \
        } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
    if (!x.is_contiguous()) { \
          fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
          exit(-1); \
        } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

__device__ Key EMPTY = -1;
__device__ Key rp0 = 999269;
__device__ Key rp1 = 999437;
__device__ Key rp2 = 1999377;

__device__ index_t map2key(const Key* keys, const Key* dims, int num_dim) {
  index_t ans = 0;
  Key key;
  for (int i = 0; i < num_dim; i++) {
    key = keys[i];
    if (key >= dims[i]) {
      key = dims[i]-1;
    }
    if (key < 0) {
      key = 0;
    }
    ans = ans * dims[i] + key;
  }
  return ans;
}

__device__ index_t hashkey(const Key key, index_t ht_size) {
  return ((key % ht_size) * rp0 + rp1) % ht_size;
}

__global__ void hash_insert_gpu_kernel(
                  Key* ht_keys,
                  Float* ht_values,
                  Key* reverse_indices,
                  index_t ht_size,
                  const Key* dims,
                  int num_dim,
                  const Key* insert_keys, // keys in [N, D]
                  const Float* insert_values,
                  uint32 num_inserts
                  ) {
  unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
  if (threadid < num_inserts) {
    const Key* insert_key_ptr = &insert_keys[threadid*num_dim];
    Key insert_key = map2key(insert_key_ptr, dims, num_dim);
    index_t hash_idx = hashkey(insert_key, ht_size);
    
    const Float* insert_value = &insert_values[threadid*num_dim];
    Key prev = atomicCAS((unsigned long long int*)(&ht_keys[hash_idx]),
                         (unsigned long long int)EMPTY,
                         (unsigned long long int)insert_key);
    while (prev != EMPTY) {
      hash_idx = (hash_idx + 1) % ht_size;
      prev = atomicCAS((unsigned long long int*)(&ht_keys[hash_idx]),
                       (unsigned long long int)EMPTY,
                       (unsigned long long int)insert_key);
    }
    if (prev == EMPTY) {
      // hit
      ht_keys[hash_idx] = insert_key;
      Float* ht_value = &ht_values[hash_idx*num_dim];
      for (int i = 0; i < num_dim; i++) {
        ht_value[i] = insert_value[i];
      }
      reverse_indices[hash_idx] = threadid;
    }
  }
}

__global__ void hybrid_geop_kernel(const Key* keys,
                                   const Float* values,
                                   const Key* reverse_indices,
                                   const Key* dims,
                                   const Key* qmin,
                                   const Key* qmax,
                                   Key* query_keys, // keys in [N, D]
                                   uint32 num_queries,
                                   int num_dim,
                                   index_t ht_size, // hash table size
                                   Float* mu, // 4d geometric centers, first dimension is batch index
                                   Float* sigma, // 3x3 covariance matrices
                                   Float decay_radius // controls how weight decays
                                  ) {
  unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
  if (threadid < num_queries) {
    Key* query_key_ptr = &query_keys[threadid*num_dim];
    int num_combinations = 1;
    for (int i = 0; i < num_dim; i++) {
      num_combinations *= qmax[i] - qmin[i] + 1;
    }

    // first pass: find geometric center mu
    // store in mu_ptr
    Float* mu_ptr = &mu[threadid*num_dim];
    Float weight_sum = 0.0;
    index_t hash_idx;
    for (int i = 0; i < num_dim; i++) {
      mu_ptr[i] = 0.0;
    }
    Key query_key = map2key(query_key_ptr, dims, num_dim);
    hash_idx = hashkey(query_key, ht_size);
    
    while (keys[hash_idx] != -1) {
      if (keys[hash_idx] == query_key) {
        const Float* value_ptr = &values[hash_idx*num_dim];
        // first dimension of value_ptr ignored since it's batch index
        for (int i = 0; i < num_dim; i++) {
          mu_ptr[i] += value_ptr[i];
        }
        weight_sum += 1.0;
      }
      hash_idx = (hash_idx + 1) % ht_size;
    }

    if (weight_sum < 1) {
      weight_sum = 1.0;
    }
    for (int i = 0; i < num_dim; i++) {
      mu_ptr[i] /= weight_sum; 
    }
    
    // second pass: compute covariance matrices
    // store in sigma_ptr
    Float* sigma_ptr = &sigma[threadid*9];
    for (int i = 0; i < 9; i++) {
      sigma_ptr[i] = 0.0;
    }
    Float decay_radius2 = decay_radius * decay_radius;
    weight_sum = 0.0;
    for (int c = 0; c < num_combinations; c++) {
      // temporarily shift keys 
      int temp = c;
      for (int i = 0; i < num_dim; i++) {
        int num_comb_this = qmax[i] - qmin[i] + 1;
        query_key_ptr[i] += temp % num_comb_this + qmin[i];
        temp /= num_comb_this;
      }

      Key query_key = map2key(query_key_ptr, dims, num_dim);
      hash_idx = hashkey(query_key, ht_size);
      
      while (keys[hash_idx] != -1) {
        if (keys[hash_idx] == query_key) {
          const Float* value_ptr = &values[hash_idx*num_dim];
          // first dimension of value_ptr ignored since it's batch index
          Float dist2 = 0.0;
          for (int i = 1; i < 4; i++) {
            Float di = value_ptr[i] - mu_ptr[i];
            dist2 += di * di;
          }
          Float weight = decay_radius2 / (dist2 + decay_radius2);
          for (int i = 1; i < 4; i++) { // row
            Float di = value_ptr[i] - mu_ptr[i];
            for (int j = 1; j < 4; j++) { // column
              Float dj = value_ptr[j] - mu_ptr[j];
              sigma_ptr[(i-1)*3+(j-1)] += di * dj * weight;
            }
          }
          weight_sum += weight;
        }
        hash_idx = (hash_idx + 1) % ht_size;
      }

      // recover shifted keys
      temp = c;
      for (int i = 0; i < num_dim; i++) {
        int num_comb_this = qmax[i] - qmin[i] + 1;
        query_key_ptr[i] -= temp % num_comb_this + qmin[i];
        temp /= num_comb_this;
      }
    }
    if (weight_sum < 1e-6) {
      weight_sum = 1.0;
    }
    for (int i = 0; i < 9; i++) {
      sigma_ptr[i] /= weight_sum;
    }
  } 
}

void hybrid_geop_gpu(at::Tensor keys,
                     at::Tensor values,
                     at::Tensor reverse_indices,
                     at::Tensor dims,
                     at::Tensor qmin,
                     at::Tensor qmax,
                     at::Tensor query_keys,
                     at::Tensor mu,
                     at::Tensor sigma,
                     Float decay_radius) {
  CHECK_INPUT(keys);
  CHECK_INPUT(values);
  CHECK_INPUT(reverse_indices);
  CHECK_INPUT(dims);
  CHECK_INPUT(qmin);
  CHECK_INPUT(qmax);
  CHECK_INPUT(query_keys);
  CHECK_INPUT(mu);
  CHECK_INPUT(sigma);

  const Key* key_data = keys.data<Key>();
  const Float* value_data = values.data<Float>();
  const Key* reverse_index_data = reverse_indices.data<Key>();
  const Key* dim_data = dims.data<Key>();
  const Key* qmin_data = qmin.data<Key>();
  const Key* qmax_data = qmax.data<Key>();
  Key* query_key_data = query_keys.data<Key>();
  Float* mu_data = mu.data<Float>();
  Float* sigma_data = sigma.data<Float>();

  int num_dim = values.size(1);
  index_t ht_size = keys.size(0);
  
  uint32 num_queries = query_keys.size(0);
  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     hybrid_geop_kernel, 0, 0);

  uint32 gridsize = (num_queries + threadblocksize - 1) / threadblocksize;
  hybrid_geop_kernel<<<gridsize, threadblocksize>>>(
    key_data, value_data, reverse_index_data, 
    dim_data, qmin_data, qmax_data, query_key_data,
    num_queries, num_dim, ht_size, mu_data, sigma_data, decay_radius
  );
}

void hash_insert_gpu(at::Tensor keys, at::Tensor values,
                     at::Tensor reverse_indices, at::Tensor dims,
                     at::Tensor insert_keys, at::Tensor insert_values
                     ) {
  CHECK_INPUT(keys);
  CHECK_INPUT(values);
  CHECK_INPUT(reverse_indices);
  CHECK_INPUT(dims);
  CHECK_INPUT(insert_keys);
  CHECK_INPUT(insert_values);

  Key* key_data = keys.data<Key>();
  Key* reverse_indices_data = reverse_indices.data<Key>();
  int num_dim = insert_values.size(1);
  const Key* insert_key_data = insert_keys.data<Key>();
  Float* value_data = values.data<Float>();
  const Float* insert_value_data = insert_values.data<Float>();
  index_t ht_size = keys.size(0);
  uint32 num_inserts = insert_keys.size(0);
  const Key* dims_data = dims.data<Key>();
  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     hash_insert_gpu_kernel, 0, 0);

  uint32 gridsize = (num_inserts + threadblocksize - 1) / threadblocksize;
  hash_insert_gpu_kernel<<<gridsize, threadblocksize>>>(
    key_data, value_data, reverse_indices_data, 
    ht_size, dims_data, num_dim,
    insert_key_data, insert_value_data,
    num_inserts
  );
}
