
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
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

__global__ void svd3_kernel(Float* input, Float* output_u,
                            Float* output_s, Float* output_v,
                            int num_input) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_input) {
	  svd3(
		  input[tid * 9 + 0], input[tid * 9 + 1], input[tid * 9 + 2], 
		  input[tid * 9 + 3], input[tid * 9 + 4], input[tid * 9 + 5], 
		  input[tid * 9 + 6], input[tid * 9 + 7], input[tid * 9 + 8],

		  output_u[tid * 9 + 0], output_u[tid * 9 + 1], output_u[tid * 9 + 2],
		  output_u[tid * 9 + 3], output_u[tid * 9 + 4], output_u[tid * 9 + 5],
		  output_u[tid * 9 + 6], output_u[tid * 9 + 7], output_u[tid * 9 + 8],

		  output_s[tid * 3 + 0], output_s[tid * 3 + 1], output_s[tid * 3 + 2],

		  output_v[tid * 9 + 0], output_v[tid * 9 + 1], output_v[tid * 9 + 2],
		  output_v[tid * 9 + 3], output_v[tid * 9 + 4], output_v[tid * 9 + 5],
		  output_v[tid * 9 + 6], output_v[tid * 9 + 7], output_v[tid * 9 + 8]
	  );
  }
}

void svd3_gpu(at::Tensor input,
              at::Tensor out_u,
              at::Tensor out_s,
              at::Tensor out_v) {
  CHECK_INPUT(input);
  CHECK_INPUT(out_u);
  CHECK_INPUT(out_s);
  CHECK_INPUT(out_v);
  uint32 num_input = input.size(0);
  Float* input_data = input.data<Float>();
  Float* out_u_data = out_u.data<Float>();
  Float* out_s_data = out_s.data<Float>();
  Float* out_v_data = out_v.data<Float>();
  
  int mingridsize, threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,
     svd3_kernel, 0, 0);

  uint32 gridsize = (num_input + threadblocksize - 1) / threadblocksize;
  svd3_kernel<<<gridsize, threadblocksize>>>(
    input_data,
    out_u_data,
    out_s_data,
    out_v_data,
    num_input
  );
}
