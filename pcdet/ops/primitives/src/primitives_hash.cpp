#include <stdio.h>
#include <math.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include "primitives_cpu.h"
#include <unordered_map>
#include "primitives_hash.h"
#include <cassert>
#include <algorithm>

using namespace std;

void VoxelHashTable::hash(at::Tensor _voxel_size) {
  voxel_size = _voxel_size;
  float* vs = voxel_size.data_ptr<float>();
  dims.resize(ndim);
  for (int dim = 0; dim < ndim; dim++) {
    dims[dim] = round((max[dim] - min[dim])/vs[dim]) + 1;
  }

  auto points = point_tensor.accessor<float, 2>();
  voxel_hash_table.clear();
  vector<int64> vi(ndim);
  for (int i = 0; i < N; i++) {
    int64 hash_id = 0;
    for (int dim = 0; dim < ndim; dim++) {
      vi[dim] = round((points[i][dim] - min[dim]) / vs[dim]);
      hash_id = hash_id * dims[dim] + vi[dim];
    }
    auto it = voxel_hash_table.find(hash_id);
    if (it != voxel_hash_table.end()) {
      it->second.push_back(i);
    } else {
      vector<int> vp = {i};
      voxel_hash_table.insert(std::move(make_pair(hash_id, vp)));
    }
  }
}

torch::Tensor VoxelHashTable::query_points_in_voxel(
    vector<int> qmin, vector<int> qmax) {
  int num_combination = 1;
  for (int dim = 0; dim < ndim; dim++) {
    num_combination *= qmax[dim] - qmin[dim] + 1;
  }
  vector<int64> d(ndim), vi(ndim);
  vector<vector<int64>> adjs(voxel_hash_table.size());
  int num_edges = 0;
  int voxel_id = 0;
  for (auto it = voxel_hash_table.begin();
       it != voxel_hash_table.end(); it++, voxel_id++) {
    int64 hash_id = it->first;
    for (int dim = ndim-1; dim >= 0; dim--) {
      vi[dim] = hash_id % dims[dim];
      hash_id /= dims[dim];
    }
    // enumerate offsets in each dimension
    for (int c = 0; c < num_combination; c++) {
      // get offsets
      int temp = c;
      for (int dim = 0; dim < ndim; dim++) {
        int range = qmax[dim] - qmin[dim] + 1;
        d[dim] = temp % range + qmin[dim];
        temp = temp / range;
      }
      int64 hash_id_ijk = 0;
      bool is_valid = true;
      for (int dim = 0; dim < ndim; dim++) {
        int64 v = vi[dim] + d[dim];
        if (v < 0 || v >= dims[dim]) {
          is_valid = false;
          break;
        }
        hash_id_ijk = hash_id_ijk * dims[dim] + v;
      }
      if (!is_valid) { continue; }
      auto it2 = voxel_hash_table.find(hash_id_ijk);
      if (it2 == voxel_hash_table.end()) { continue; }
      adjs[voxel_id].push_back(hash_id_ijk);
      num_edges += it2->second.size();
    }
  }
  torch::Tensor edges_tensor = torch::zeros(
    {num_edges, 2}, torch::dtype(torch::kInt32)
  );
  auto edges_acc = edges_tensor.accessor<int, 2>();
  int edge_size = 0;
  voxel_id = 0;
  for (auto it = voxel_hash_table.begin();
       it != voxel_hash_table.end(); voxel_id++, it++) {
    for (int64 hash_id_ijk : adjs[voxel_id]) {
      auto it2 = voxel_hash_table.find(hash_id_ijk);
      for (int &idx : it2->second) {
        edges_acc[edge_size][0] = voxel_id;
        edges_acc[edge_size++][1] = idx;
      }
    }
  }
  
  return edges_tensor;
}

torch::Tensor VoxelHashTable::query_point_correspondence(
    at::Tensor query_tensor, vector<int> qmin, vector<int> qmax) {
  int num_combination = 1;
  for (int dim = 0; dim < ndim; dim++) {
    num_combination *= qmax[dim] - qmin[dim] + 1;
  }
  float* vs = voxel_size.data_ptr<float>();
  vector<int64> d(ndim), vi(ndim);
  int num_query = query_tensor.size(0);
  auto points = point_tensor.accessor<float, 2>();
  auto qpoints = query_tensor.accessor<float, 2>();
  vector<pair<int, int>> edges;
  for (int i = 0; i < num_query; i++) {
    for (int dim = 0; dim < ndim; dim++) {
      vi[dim] = round((qpoints[i][dim] - min[dim])/vs[dim]);
    }
    float min_dist = 1e10;
    int corres = -1;
    // enumerate offsets in each dimension
    for (int c = 0; c < num_combination; c++) {
      // get offsets
      int temp = c;
      for (int dim = 0; dim < ndim; dim++) {
        int range = qmax[dim] - qmin[dim] + 1;
        d[dim] = temp % range + qmin[dim];
        temp = temp / range;
      }
      int64 hash_id_ijk = 0;
      bool is_valid = true;
      for (int dim = 0; dim < ndim; dim++) {
        int64 v = vi[dim] + d[dim];
        if (v < 0 || v >= dims[dim]) {
          is_valid = false;
          break;
        }
        hash_id_ijk = hash_id_ijk * dims[dim] + v;
      }
      if (!is_valid) { continue; }
      auto it2 = voxel_hash_table.find(hash_id_ijk);
      if (it2 == voxel_hash_table.end()) { continue; }
      for (int k : it2->second) {
        float dx = qpoints[i][0] - points[k][0];
        float dy = qpoints[i][1] - points[k][1];
        float dz = qpoints[i][2] - points[k][2];
        float dist = dx*dx + dy*dy + dz*dz;
        if (dist < min_dist) {
          corres = k;
          min_dist = dist;
        }
      }
    }
    if (corres == -1) { continue; }
    edges.push_back(make_pair(i, corres));
  }
  torch::Tensor edges_tensor = torch::zeros(
    {edges.size(), 2}, torch::dtype(torch::kInt32)
  );
  auto edges_acc = edges_tensor.accessor<int, 2>();
  int edge_size = 0;
  for (int i = 0; i < edges.size(); i++) {
    edges_acc[i][0] = edges[i].first;
    edges_acc[i][1] = edges[i].second;
  }
  
  return edges_tensor;
}

torch::Tensor VoxelHashTable::query_point_edges(
    at::Tensor query_tensor, vector<int> qmin, vector<int> qmax,
    int max_num_neighbors) {
  int num_combination = 1;
  for (int dim = 0; dim < ndim; dim++) {
    num_combination *= qmax[dim] - qmin[dim] + 1;
  }
  vector<int64> d(ndim), vi(ndim);
  int num_query = query_tensor.size(0);
  auto points = point_tensor.accessor<float, 2>();
  auto qpoints = query_tensor.accessor<float, 2>();
  float* vs = voxel_size.data_ptr<float>();
  vector<pair<int, int>> edges;
  vector<pair<float, int>> corres;
  corres.reserve(max_num_neighbors);
  for (int i = 0; i < num_query; i++) {
    for (int dim = 0; dim < ndim; dim++) {
      vi[dim] = round((qpoints[i][dim] - min[dim])/vs[dim]);
    }
    float min_dist = 1e10;
    corres.resize(0);
    // enumerate offsets in each dimension
    for (int c = 0; c < num_combination; c++) {
      // get offsets
      int temp = c;
      for (int dim = 0; dim < ndim; dim++) {
        int range = qmax[dim] - qmin[dim] + 1;
        d[dim] = temp % range + qmin[dim];
        temp = temp / range;
      }
      int64 hash_id_ijk = 0;
      bool is_valid = true;
      for (int dim = 0; dim < ndim; dim++) {
        int64 v = vi[dim] + d[dim];
        if (v < 0 || v >= dims[dim]) {
          is_valid = false;
          break;
        }
        hash_id_ijk = hash_id_ijk * dims[dim] + v;
      }
      if (!is_valid) { continue; }
      auto it2 = voxel_hash_table.find(hash_id_ijk);
      if (it2 == voxel_hash_table.end()) { continue; }
      for (int k : it2->second) {
        float dx = points[i][0] - qpoints[k][0];
        float dy = points[i][1] - qpoints[k][1];
        float dz = points[i][2] - qpoints[k][2];
        float dist = dx * dx + dy * dy + dz * dz;
        //pair<float, int> cur = make_pair(dist, k);
        //for (int k1 = 0; k1 < corres.size(); k1++) {
        //  float dk = corres[k1].first;
        //  if (cur.first < dk) {
        //    pair<float, int> temp = corres[k1];
        //    corres[k1] = cur;
        //    cur = temp;
        //  }
        //}
        //if (corres.size() < max_num_neighbors) {
        //  corres.push_back(cur);
        //}
        corres.push_back(make_pair(dist, k));
      }
    }
    if (corres.size() > max_num_neighbors) {
      nth_element(corres.begin(),
                  corres.begin()+max_num_neighbors, corres.end());
      corres.resize(max_num_neighbors);
    }
    for (auto p : corres) {
      edges.push_back(make_pair(i, p.second));
    }
  }
  torch::Tensor edges_tensor = torch::zeros(
    {edges.size(), 2}, torch::dtype(torch::kInt32)
  );
  auto edges_acc = edges_tensor.accessor<int, 2>();
  for (int i = 0; i < edges.size(); i++) {
    edges_acc[i][0] = edges[i].first;
    edges_acc[i][1] = edges[i].second;
  }
  
  return edges_tensor;
}
