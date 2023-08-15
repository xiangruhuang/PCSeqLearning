#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "torch_hash.h"
#include <ATen/ParallelOpenMP.h>
#include <iostream>

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
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

