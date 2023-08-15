#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "virtual_array.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("virtual_scatter_add_gpu", &virtual_scatter_add_gpu, "scatter_add a virtual array");
  m.def("virtual_outer_and_sum_gpu", &virtual_outer_and_sum_gpu, "compute edge-wise outer product and sum");
}
