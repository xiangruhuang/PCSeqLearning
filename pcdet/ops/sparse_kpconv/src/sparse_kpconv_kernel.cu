#include <stdio.h>
#include "stdint.h"
#include "sparse_kpconv.h"
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cmath>
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

#define EPSILON 1e-6

__global__ void tensor_outer_kernel(
                  const Float* t0,
                  const Float* t1,
                  const Float* t2,
                  Float* out,
                  const int N,
                  const int D0,
                  const int D1,
                  const int D2
                  ) {
  unsigned int d = blockIdx.x*blockDim.x + threadIdx.x;
  if (d < D0*D1*D2) {
    int d0 = d / (D1 * D2);
    int d1 = (d / D2) % D1;
    int d2 = d % D2;
    // d = d0*D1*D2 + d1*D2 + d2
    Float& out_this = out[d];
    out_this = 0.0;
    for (int n = 0; n < N; n++) {
      const Float &t0_this = t0[n*D0+d0];
      const Float &t1_this = t1[n*D1+d1];
      const Float &t2_this = t2[n*D2+d2];
      out_this += t0_this * t1_this * t2_this;
    }
  }
} 

__global__ void sparse_kpconv_kernel(
                  const Float* x, // [N, D1]
                  const Float* W, // [K, D2, D1]
                  const Float* a, // [N, K]
                  Float* y, // [N, D2]
                  const int N,
                  const int K,
                  const int D1,
                  const int D2
                  ) {
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y*blockDim.y + threadIdx.y;
  if (n < N && k < K) {
    // y_n = \sum_k (a_{nk} W_k) x_n
    const Float& a_n_k = a[n*K+k];
    
    if (abs(a_n_k) > EPSILON) {
      const Float* x_n = &x[n*D1];
      Float* y_n = &y[n*D2];
      const Float* W_k = &W[k*D1*D2]; // W_k
      // plain matrix vector multiplication
      // W_k, x_n
      for (int o = 0; o < D2; o++) { // output dim
        Float sum = 0.0;
        const Float* W_k_o = &W_k[o*D1];
        for (int i = 0; i < D1; i++) { // input dim
          sum += x_n[i] * W_k_o[i];
        }
        sum = sum * a_n_k;
        atomicAdd(&y_n[o], sum);
      }
    }
  }
}

void sparse_kpconv_gpu(at::Tensor input, at::Tensor kernel,
                       at::Tensor weight, at::Tensor output) {
  
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);

  // input.size = [N, D1]
  int N = input.size(0);
  int D1 = input.size(1);
  assert((input.sizes() == std::vector<int64_t>{N, D1}));
  // kernel.size = [K, D2, D1]
  int K = kernel.size(0);
  int D2 = kernel.size(1);
  assert((kernel.sizes() == std::vector<int64_t>{K, D2, D1}));
  // weight.size = [N, K]
  assert((weight.sizes() == std::vector<int64_t>{N, K}));
  // output.size = [N, D2]
  assert((output.sizes() == std::vector<int64_t>{N, D2}));

  const Float* input_data = input.data<Float>();
  const Float* kernel_data = kernel.data<Float>();
  const Float* weight_data = weight.data<Float>();
  Float* output_data = output.data<Float>();

  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     sparse_kpconv_kernel, 0, 0);

  int tx = threadblocksize/16;
  dim3 threadblocksize3(tx, 16, 1);
  int gx = (N + tx - 1) / tx;
  int gy = (K + 16 - 1) / 16;
  dim3 gridsize3(gx, gy, 1);
  sparse_kpconv_kernel<<<gridsize3, threadblocksize3>>>(
    input_data, kernel_data, weight_data, output_data,
    N, K, D1, D2
  );
}

void tensor_outer_gpu(at::Tensor t0, at::Tensor t1,
                      at::Tensor t2, at::Tensor output) {
  
  CHECK_INPUT(t0);
  CHECK_INPUT(t1);
  CHECK_INPUT(t2);
  CHECK_INPUT(output);

  // t0.size = [N, D0]
  int N = t0.size(0);
  int D0 = t0.size(1);
  assert((t0.sizes() == std::vector<int64_t>{N, D0}));
  // t1.size = [N, D1]
  int D1 = t1.size(1);
  assert((t1.sizes() == std::vector<int64_t>{N, D1}));
  // weight.size = [N, K]
  int D2 = t2.size(1);
  assert((t2.sizes() == std::vector<int64_t>{N, D2}));
  // output.size = [D0, D1, D2]
  assert((output.sizes() == std::vector<int64_t>{D0, D1, D2}));

  const Float* t0_data = t0.data<Float>();
  const Float* t1_data = t1.data<Float>();
  const Float* t2_data = t2.data<Float>();
  Float* output_data = output.data<Float>();

  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     tensor_outer_kernel, 0, 0);

  uint32 gridsize = (D0*D1*D2 + threadblocksize - 1) / threadblocksize;
  tensor_outer_kernel<<<gridsize, threadblocksize>>>(
    t0_data, t1_data, t2_data, output_data,
    N, D0, D1, D2
  );
}
