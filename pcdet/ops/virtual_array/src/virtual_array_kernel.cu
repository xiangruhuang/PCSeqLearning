#include <stdio.h>
#include "stdint.h"
#include "virtual_array.h"
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

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

__global__ void virtual_scatter_add_kernel(
                    const Float* core_array_a, // [U, d2]
                    const Key* virtual_index_a, // [E] in [U]
                    const Float* virtual_weight_a, // [E]
                    const Key* index_b, // [E] in [Q]
                    Float* output_b, // [Q, d2]
                    int b_size,
                    int d2,
                    int num_edges
) {
  unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x; // e in [E]
  if (threadid < num_edges) {
    const Key index_a = virtual_index_a[threadid];
    const Float* core_array_a_ptr = &core_array_a[index_a*d2]; // [d2]
    const Float w = virtual_weight_a[threadid];
    Float* output_b_ptr = &output_b[index_b[threadid]*d2]; // [d2]
    for (int i = 0; i < d2; i++) {
      Float ai = core_array_a_ptr[i] * w;
      atomicAdd(&output_b_ptr[i], ai);
    }
  }
}

at::Tensor virtual_scatter_add_gpu(
      at::Tensor core_array_a,
      at::Tensor virtual_index_a,
      at::Tensor virtual_weight_a,
      at::Tensor index_b,
      int b_size
) {
    
    CHECK_INPUT(core_array_a);
    CHECK_INPUT(virtual_index_a);
    CHECK_INPUT(virtual_weight_a);
    CHECK_INPUT(index_b);

    int d2 = core_array_a.size(1);
    int num_edges = index_b.size(0);
    const Float* core_array_a_data = core_array_a.data<Float>();
    const Key* virtual_index_a_data = virtual_index_a.data<Key>();
    const Float* virtual_weight_a_data = virtual_weight_a.data<Float>();
    const Key* index_b_data = index_b.data<Key>();
    
    torch::Tensor output_b = core_array_a.new_zeros({b_size, d2});
    Float* output_b_data = output_b.data<Float>();
  
    int mingridsize, threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
       virtual_scatter_add_kernel, 0, 0);
  
    int gridsize = (num_edges + threadblocksize - 1) / threadblocksize;
    
    virtual_scatter_add_kernel<<<gridsize, threadblocksize>>>(
      core_array_a_data,
      virtual_index_a_data,
      virtual_weight_a_data,
      index_b_data,
      output_b_data,
      b_size,
      d2,
      num_edges
    );
    return output_b;
}

__global__ void virtual_outer_and_sum_kernel(
                    const Float* core_array_a_data, // [U1, D1]
                    const Key* virtual_index_a_data, // [E] in [U1]
                    const Float* core_array_b_data, // [U2, D2]
                    const Key* virtual_index_b_data, // [E] in [U2]
                    const Float* virtual_weight_data, // [E]
                    Float* output_data, // [d1, d2]
                    int d1,
                    int d2,
                    int num_edges
) {
  unsigned int edgeid = blockIdx.x*blockDim.x + threadIdx.x; // e in [E]
  if (edgeid < num_edges) {
    const Key &rowa = virtual_index_a_data[edgeid];
    const Key &rowb = virtual_index_b_data[edgeid];
    const Float* core_array_a_ptr = &core_array_a_data[rowa*d1];
    const Float* core_array_b_ptr = &core_array_b_data[rowb*d2];
    const Float &w = virtual_weight_data[edgeid];
    for (int i = 0; i < d1; i++) {
      Float ai = core_array_a_ptr[i]*w;
      for (int j = 0; j < d2; j++) {
        Float val = ai*core_array_b_ptr[j];
        atomicAdd(&output_data[i*d1+j], val);
      }
    }
  }
}

at::Tensor virtual_outer_and_sum_gpu(at::Tensor core_array_a,
                                     at::Tensor virtual_index_a,
                                     at::Tensor core_array_b,
                                     at::Tensor virtual_index_b,
                                     at::Tensor virtual_weight
) {
    CHECK_INPUT(core_array_a);
    CHECK_INPUT(virtual_index_a);
    CHECK_INPUT(core_array_b);
    CHECK_INPUT(virtual_index_b);
    CHECK_INPUT(virtual_weight);
    
    int d1 = core_array_a.size(1);
    int d2 = core_array_b.size(1);
    int num_edges = virtual_weight.size(0);

    const Float* core_array_a_data = core_array_a.data<Float>();
    const Key* virtual_index_a_data = virtual_index_a.data<Key>();
    const Float* core_array_b_data = core_array_b.data<Float>();
    const Key* virtual_index_b_data = virtual_index_b.data<Key>();
    const Float* virtual_weight_data = virtual_weight.data<Float>();
    
    torch::Tensor output = core_array_a.new_zeros({d1, d2});
    Float* output_data = output.data<Float>();

    int mingridsize, threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
       virtual_outer_and_sum_kernel, 0, 0);
  
    int gridsize = (num_edges + threadblocksize - 1) / threadblocksize;

    virtual_outer_and_sum_kernel<<<gridsize, threadblocksize>>>(
        core_array_a_data,
        virtual_index_a_data,
        core_array_b_data,
        virtual_index_b_data,
        virtual_weight_data,
        output_data,
        d1,
        d2,
        num_edges
    );
    return output;
}
