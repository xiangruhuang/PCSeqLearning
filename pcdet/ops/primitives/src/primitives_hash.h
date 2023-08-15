#ifndef PRIMITIVES_HASH_H
#define PRIMITIVES_HASH_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <unordered_map>

typedef long long int64;

class VoxelHashTable {
 public:
  VoxelHashTable(at::Tensor _point_tensor) {
    point_tensor = _point_tensor;
    this->ndim = point_tensor.size(1);
    this->N = point_tensor.size(0);
    max.resize(ndim);
    min.resize(ndim);
    auto points = point_tensor.accessor<float, 2>();
    for (int i = 0; i < N; i++) {
      for (int dim = 0; dim < ndim; dim++) {
        float &pid = points[i][dim];
        if ((i == 0) || (pid < min[dim])) {
          min[dim] = pid;
        }
        if ((i == 0) || (pid > max[dim])) {
          max[dim] = pid;
        }
      }
    }
    
  };
  void hash(at::Tensor _voxel_size);
  torch::Tensor query_points_in_voxel(
                  std::vector<int> qmin, std::vector<int> qmax);
  torch::Tensor query_point_correspondence(
                  at::Tensor query_tensor,
                  std::vector<int> qmin, std::vector<int> qmax);
  torch::Tensor query_point_edges(
                  at::Tensor query_tensor,
                  std::vector<int> qmin, std::vector<int> qmax,
                  int max_num_neighbors);
  
 private:
  at::Tensor point_tensor;
  at::Tensor voxel_size;
  std::vector<float> max, min;
  std::vector<int64> dims;
  int N, ndim;
  std::unordered_map<int64, std::vector<int>> voxel_hash_table;
};

#endif
