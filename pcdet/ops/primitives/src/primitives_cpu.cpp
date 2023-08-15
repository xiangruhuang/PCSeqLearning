#include <stdio.h>
#include <math.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include "primitives_cpu.h"
#include <unordered_map>
#include "primitives_hash.h"

using namespace std;

vector<torch::Tensor> voxelization(
  at::Tensor point_tensor, at::Tensor voxel_size, bool ambient=false) {
  VoxelHashTable h(point_tensor);
  h.hash(voxel_size);
  vector<int> qmin = {0, 0, 0, 0}, qmax = {0, 0, 0, 0};
  torch::Tensor edges_tensor = h.query_points_in_voxel(qmin, qmax);
  if (ambient) {
    qmin = {-1,-1,-1,0};
    qmax = { 1, 1, 1,0};
    torch::Tensor amb_edges_tensor = h.query_points_in_voxel(qmin, qmax);
    return {edges_tensor, amb_edges_tensor};
  } else {
    return {edges_tensor};
  }
}

torch::Tensor voxel_graph(
  at::Tensor point_tensor, at::Tensor query_tensor,
  at::Tensor voxel_size, int temporal_offset, int max_num_neighbors) {
  VoxelHashTable h(point_tensor);
  h.hash(voxel_size);
  vector<int> qmin = {-1,-1,-1,temporal_offset};
  vector<int> qmax = { 1, 1, 1,temporal_offset};
  torch::Tensor edges_tensor = h.query_point_edges(
                                   query_tensor, qmin, qmax,
                                   max_num_neighbors);
  return edges_tensor;
}

torch::Tensor query_point_correspondence(
  at::Tensor moving_tensor, at::Tensor ref_tensor,
  at::Tensor voxel_size, int temporal_offset) {
  VoxelHashTable h(ref_tensor);
  h.hash(voxel_size);
  vector<int> qmin = {-1,-1,-1,temporal_offset};
  vector<int> qmax = { 1, 1, 1,temporal_offset};
  torch::Tensor edges_tensor = h.query_point_correspondence(
                                   moving_tensor, qmin, qmax);
  
  return edges_tensor;
}

//torch::Tensor query_temporal_correspondences(
//    at::Tensor point_tensor,
//    at::Tensor voxel_size,
//    int temporal_offset) {
// 
//  int ndim = 4;
//  vector<int> range = grid_dimension({point_tensor}, voxel_size);
//  unordered_map<int, vector<int>> umap;
//  vector<int> dims(ndim);
//  build_map(point_tensor, voxel_size, range, dims, umap);
//
//  auto points = point_tensor.accessor<float, 2>();
//
//  int N = point_tensor.size(0);
//  vector<pair<int, int>> edges;
//
//  vector<int> vi(ndim);
//  for (auto it = umap.begin(); it != umap.end(); it++) {
//    int hash_id = it->first;
//    for (int dim = ndim-1; dim >= 0; dim--) {
//      vi[dim] = hash_id % dims[dim];
//      hash_id = hash_id / dims[dim];
//    }
//    int corres = -1;
//    float min_dist = 1e10;
//    for (int itr = 0; itr < 27; itr++) {
//      int temp = itr;
//      bool all_zero = true;
//      for (int dim = 0; dim < ndim; dim++) {
//        d[dim] = (temp % 3)-1;
//        if (d[dim] != 0) {
//          all_zero = false;
//        }
//        temp = temp / 3;
//      }
//      if (all_zero) { continue; }
//      int hash_id_ijk = 0;
//      for (int dim = 0; dim < ndim; dim++) {
//        if (dim > 0) {
//          hash_id_ijk *= dims[dim];
//        }
//        hash_id_ijk += (vi[dim]+d[dim]);
//      }
//      auto it2 = umap.find(hash_id_ijk);
//      if (it2 == umap.end()) {
//        continue;
//      }
//      for (int &j : it2->second) {
//        double dx = qpoints[i][0] - points[j][0];
//        double dy = qpoints[i][1] - points[j][1];
//        double dz = qpoints[i][2] - points[j][2];
//
//        double dist = dx*dx + dy*dy + dz*dz;
//        if (dist < min_dist) {
//          min_dist = dist;
//          corres = j;
//        }
//      }
//    }
//    if (corres != -1) {
//      edges.push_back(std::move(make_pair(i, corres)));
//    }
//  }
//  
//  torch::Tensor corres = torch::zeros({edges.size(), 2}, torch::dtype(torch::kInt32));
//  for (int i = 0; i < edges.size(); i++) {
//    corres[i][0] = edges[i].first;
//    corres[i][1] = edges[i].second;
//  }
//
//  return corres;
//}

//torch::Tensor query_correspondences(
//    at::Tensor target_point_tensor,
//    at::Tensor query_point_tensor,
//    at::Tensor voxel_size) {
//  
//  vector<at::Tensor> point_tensors = {target_point_tensor, query_point_tensor};
//  vector<int> range = grid_dimension(point_tensors, voxel_size);
//  unordered_map<int, vector<int>> umap;
//  unordered_map<int, vector<int>> qmap;
//  int ndim = 3;
//  vector<int> dims(ndim);
//  build_map(target_point_tensor, voxel_size, range, dims, umap);
//  build_map(query_point_tensor, voxel_size, range, dims, qmap);
//
//  auto points = target_point_tensor.accessor<float, 2>();
//  auto qpoints = query_point_tensor.accessor<float, 2>();
//
//  int Q = query_point_tensor.size(0);
//  vector<pair<int, int>> edges;
//
//  vector<int> vi(ndim);
//  for (auto it = qmap.begin(); it != qmap.end(); it++) {
//    int hash_id = it->first;
//    for (int dim = ndim-1; dim >= 0; dim--) {
//      vi[dim] = hash_id % dims[dim];
//      hash_id = hash_id / dims[dim];
//    }
//    int corres = -1;
//    float min_dist = 1e10;
//    for (int itr = 0; itr < 27; itr++) {
//      int temp = itr;
//      bool all_zero = true;
//      for (int dim = 0; dim < ndim; dim++) {
//        d[dim] = (temp % 3)-1;
//        if (d[dim] != 0) {
//          all_zero = false;
//        }
//        temp = temp / 3;
//      }
//      if (all_zero) { continue; }
//      int hash_id_ijk = 0;
//      for (int dim = 0; dim < ndim; dim++) {
//        if (dim > 0) {
//          hash_id_ijk *= dims[dim];
//        }
//        hash_id_ijk += (vi[dim]+d[dim]);
//      }
//      auto it2 = umap.find(hash_id_ijk);
//      if (it2 == umap.end()) {
//        continue;
//      }
//      for (int &j : it2->second) {
//        double dx = qpoints[i][0] - points[j][0];
//        double dy = qpoints[i][1] - points[j][1];
//        double dz = qpoints[i][2] - points[j][2];
//
//        double dist = dx*dx + dy*dy + dz*dz;
//        if (dist < min_dist) {
//          min_dist = dist;
//          corres = j;
//        }
//      }
//    }
//    if (corres != -1) {
//      edges.push_back(std::move(make_pair(i, corres)));
//    }
//  }
//  
//  torch::Tensor corres = torch::zeros({edges.size(), 2}, torch::dtype(torch::kInt32));
//  for (int i = 0; i < edges.size(); i++) {
//    corres[i][0] = edges[i].first;
//    corres[i][1] = edges[i].second;
//  }
//
//  return corres;
//}
