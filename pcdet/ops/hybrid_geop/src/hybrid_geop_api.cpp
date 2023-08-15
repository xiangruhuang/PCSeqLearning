#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "hybrid_geop.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hybrid_geop_gpu", &hybrid_geop_gpu, "compute hybrid geometric primitives");
  m.def("hash_insert_gpu", &hash_insert_gpu, "hash points into gpu hash table");
}
