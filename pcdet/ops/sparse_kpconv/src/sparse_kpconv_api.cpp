#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "sparse_kpconv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_kpconv_gpu", &sparse_kpconv_gpu, "sparse KPConv op, memory efficient!");
  m.def("tensor_outer_gpu", &tensor_outer_gpu, "outer product of three tensors");
}
