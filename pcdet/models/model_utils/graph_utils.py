import torch
from torch import nn
from torch_scatter import scatter
from easydict import EasyDict
from torch_cluster import knn

import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix

from pcdet.ops.pointops.functions.pointops import (
    knnquery
)
from pcdet.ops.torch_hash.torch_hash_cuda import (
    hash_insert_gpu,
    correspondence,
    radius_graph_gpu,
    points_in_radius_gpu
)
from .misc_utils import bxyz_to_xyz_index_offset
from pcdet.utils import common_utils


def select_graph(graph_cfgs, i):
    if isinstance(graph_cfgs, list):
        idx = i
        for j in range(len(graph_cfgs)):
            if idx >= graph_cfgs[j]['COUNT']:
                idx -= graph_cfgs[j]['COUNT']
            else:
                graph_cfg = common_utils.indexing_list_elements(graph_cfgs[j], idx)
                break
    else:
        graph_cfg = common_utils.indexing_list_elements(graph_cfgs, i)
    return graph_cfg

def build_graph(graph_cfg, runtime_cfg=None):
    graph = GRAPHS[graph_cfg["TYPE"]]
    return graph(model_cfg=graph_cfg, runtime_cfg=runtime_cfg)

def connected_components(edges, num_nodes=None):
    """
    Args:
        edges: [2, M]
        num_nodes: N
    Returns:
        num_components
        component: [N] representing each node's component id.
    """
    if num_nodes is None:
        num_nodes = edges.max().long().item() + 1
    adj = to_scipy_sparse_matrix(edges, num_nodes=num_nodes)
    num_components, component = sp.csgraph.connected_components(adj)
    return num_components, torch.from_numpy(component).to(edges)

class GraphTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        if "EDGE_DIRECTION" in model_cfg:
            edge_dir = model_cfg["EDGE_DIRECTION"]
            self.edge_dir = torch.tensor(edge_dir).float()
        else:
            self.edge_dir = None

    def forward(self, ref_bxyz, query_bxyz):
        e_ref, e_query = self.build_graph(ref_bxyz, query_bxyz)
        if self.edge_dir is not None:
            direction = query_bxyz[e_query, 1:4] - ref_bxyz[e_ref, 1:4]
            mask = (direction * self.edge_dir).sum(dim=-1) >= -1e-5
            e_ref, e_query = e_ref[mask], e_query[mask]
        return e_ref, e_query


class GraphV2Template(GraphTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(runtime_cfg, model_cfg)
        self.model_cfg = model_cfg
        if "EDGE_DIRECTION" in model_cfg:
            edge_dir = model_cfg["EDGE_DIRECTION"]
            self.edge_dir = torch.tensor(edge_dir).float()
        else:
            self.edge_dir = None

    def forward(self, ref, query):
        e_ref, e_query, e_weight = self.build_graph(ref, query)
        if self.edge_dir is not None:
            direction = query_bxyz[e_query, 1:4] - ref_bxyz[e_ref, 1:4]
            mask = (direction * self.edge_dir).sum(dim=-1) >= -1e-5
            e_ref, e_query, e_weight = e_ref[mask], e_query[mask], e_weight[mask]
        return e_ref, e_query, e_weight
        

class KNNGraph(GraphTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super(KNNGraph, self).__init__(
                                    runtime_cfg=runtime_cfg,
                                    model_cfg=model_cfg,
                                )
        self.k = model_cfg.get("NUM_NEIGHBORS", 32)
    
    def build_graph(self, ref_bxyz, query_bxyz):
        """Build knn graph from ref point cloud to query point cloud,
            each query point connects to k ref points.
        Args:
            ref_bxyz [N, 4]: ref point cloud
            query_bxyz [M, 4]: query point cloud
        Returns:
            edge_idx [2, M*K]: (idx_of_ref, idx_of_query)
        """
        
        ref_xyz, ref_indices, ref_offset = bxyz_to_xyz_index_offset(ref_bxyz)
        query_xyz, query_indices, query_offset = bxyz_to_xyz_index_offset(query_bxyz)
        
        # building a bipartite graph from [N] to [M]
        #N, M = ref_xyz.shape[0], query_xyz.shape[0]
        ref_idx, _ = knnquery(self.k, ref_xyz, query_xyz,
                              ref_offset.int(), query_offset.int()) # [M, k] -> [N]
        ref_idx = ref_idx.long()

        query_idx = torch.arange(query_xyz.shape[0])[:, None].expand(-1, self.k).to(ref_idx) # [M, k] -> [M]

        e_ref, e_query = torch.stack([ref_indices[ref_idx],
                                      query_indices[query_idx]], dim=0).reshape(2, -1)

        return e_ref, e_query
    
    def extra_repr(self):
        return f"num_neighbors={self.k}"


class RadiusGraph(GraphV2Template):
    def __init__(self, runtime_cfg, model_cfg):
        super(RadiusGraph, self).__init__(
                                       runtime_cfg=runtime_cfg,
                                       model_cfg=model_cfg,
                                   )
        self.radius = model_cfg.get("RADIUS", None)
        self.dynamic_radius = model_cfg.get("DYNAMIC_RADIUS", False)
        self.max_num_neighbors = model_cfg.get("MAX_NUM_NEIGHBORS", 32)
        self.sort_by_dist = model_cfg.get("SORT_BY_DIST", False)
        self.util_ratio = 0.5

        qmin = torch.tensor([0] + [-1 for i in range(3)]).int()
        qmax = torch.tensor([0] + [1 for i in range(3)]).int()
        self.register_buffer("qmin", qmin, persistent=False)
        self.register_buffer("qmax", qmax, persistent=False)
        self.relative_key = model_cfg.get("RELATIVE_KEY", 'bxyz')
    
    def build_graph(self, ref, query):
        """Build knn graph from source point cloud to target point cloud,
            each target point connects to k source points.
        Args:
            ref_bxyz [N, 4]: source point cloud
            query_bxyz [M, 4]: target point cloud
        Returns:
            edge_idx [2, M*K]: (idx_of_ref, idx_of_query)
        """
        ref_bxyz = ref[self.relative_key]
        query_bxyz = query[self.relative_key]
        assert ref_bxyz.shape[-1] == 4

        if self.dynamic_radius:
            _, indices = knn(query_bxyz[:, 1:], query_bxyz[:, 1:], 2, query_bxyz[:, 0], query_bxyz[:, 0])
            indices = indices.reshape(-1, 2)[:, 1]
            median_norm = (query_bxyz[:, 1:] - query_bxyz[indices, 1:]).norm(p=2, dim=-1).median() * 1.5
            self.radius = median_norm
            print(f'radius = {self.radius}')
        # find data range, voxel size
        radius = query_bxyz.new_zeros(query_bxyz.shape[0]) + self.radius
        voxel_size = torch.tensor([1-1e-3] + [radius.max().item() for i in range(3)]).to(ref_bxyz.device)
        all_points = torch.cat([ref_bxyz, query_bxyz], axis=0)
        pc_range_min = (all_points.min(0)[0] - voxel_size*2).cuda()
        pc_range_max = (all_points.max(0)[0] + voxel_size*2).cuda()
        voxel_coors_ref = torch.round((ref_bxyz-pc_range_min) / voxel_size).long() + 1
        voxel_coors_query = torch.round((query_bxyz-pc_range_min) / voxel_size).long() + 1
        dims = torch.round((pc_range_max - pc_range_min) / voxel_size).long()+3

        # allocate memory
        hashtable_size = int(ref_bxyz.shape[0] / self.util_ratio)

        keys = voxel_coors_ref.new_zeros(hashtable_size) - 1
        values = ref_bxyz.new_empty(hashtable_size, 4)
        reverse_indices = voxel_coors_ref.new_zeros(hashtable_size)

        # hashing
        hash_insert_gpu(
            keys,
            values,
            reverse_indices,
            dims,
            voxel_coors_ref,
            ref_bxyz)

        edges = radius_graph_gpu(
                    keys,
                    values,
                    reverse_indices,
                    dims,
                    voxel_coors_query,
                    query_bxyz,
                    self.qmin,
                    self.qmax,
                    radius,
                    self.max_num_neighbors,
                    self.sort_by_dist).T

        e_ref, e_query = edges

        return e_ref, e_query, None
    
    def extra_repr(self):
        return f"radius={self.radius}, max_ngbrs={self.max_num_neighbors}, sort={self.sort_by_dist}"


class VoxelGraph(GraphV2Template):
    def __init__(self, runtime_cfg, model_cfg):
        super(VoxelGraph, self).__init__(
                                   runtime_cfg=runtime_cfg,
                                   model_cfg=model_cfg,
                               )
        self.voxel_size = model_cfg.get("VOXEL_SIZE", None)
        self.kernel_offset = model_cfg.get("KERNEL_OFFSET", None)
        self.max_num_neighbors = model_cfg.get("MAX_NUM_NEIGHBORS", 32)
        point_cloud_range = model_cfg.get("POINT_CLOUD_RANGE", None)
        if point_cloud_range is not None:
            pc_range_min = torch.tensor([0] + point_cloud_range[:3], dtype=torch.float32)
            pc_range_max = torch.tensor([1] + point_cloud_range[3:], dtype=torch.float32)
            self.register_buffer("pc_range_min", pc_range_min)
            self.register_buffer("pc_range_max", pc_range_max)
            self.point_cloud_range = point_cloud_range
        else:
            self.pc_range_min = None
            self.pc_range_max = None
            self.point_cloud_range = None
        self.ref_key = model_cfg.get("REF_KEY", 'bxyz')
        self.query_key = model_cfg.get("QUERY_KEY", 'bcenter')

        self.util_ratio = 0.5

        qmin = torch.tensor([0] + [-self.kernel_offset for i in range(3)]).int()
        qmax = torch.tensor([0] + [self.kernel_offset for i in range(3)]).int()
        self.register_buffer("qmin", qmin, persistent=False)
        self.register_buffer("qmax", qmax, persistent=False)
    
    def build_graph(self, ref, query):
        """Build knn graph from source point cloud to target point cloud,
            each target point connects to k source points.
        Args:
            ref_bxyz [N, 4]: source point cloud
            query_bxyz [M, 4]: target point cloud
        Returns:
            edge_idx [2, M*K]: (idx_of_ref, idx_of_query)
        """
        ref_bxyz = ref[self.ref_key]
        query_bxyz = query[self.query_key]
        assert ref_bxyz.shape[-1] == 4
        ref_bxyz = ref_bxyz.float()
        query_bxyz = query_bxyz.float()

        # find data range, voxel size
        radius = query_bxyz.new_zeros(query_bxyz.shape[0]) + 1e5
        voxel_size = torch.tensor([1-1e-3] + self.voxel_size).to(ref_bxyz.device)
        if self.point_cloud_range is None:
            all_points = torch.cat([ref_bxyz, query_bxyz], axis=0)
            self.pc_range_min = (all_points.min(0)[0] - voxel_size*2).cuda()
            self.pc_range_max = (all_points.max(0)[0] + voxel_size*2).cuda()
        self.pc_range_min[0] = 0
        self.pc_range_max[0] = ref_bxyz[:, 0].max()+1
        voxel_coors_ref = torch.floor((ref_bxyz-self.pc_range_min) / voxel_size).long()
        voxel_coors_query = torch.floor((query_bxyz-self.pc_range_min) / voxel_size).long()
        dims = torch.ceil((self.pc_range_max - self.pc_range_min) / voxel_size).long() + 1

        # allocate memory
        hashtable_size = int(ref_bxyz.shape[0] / self.util_ratio)

        keys = voxel_coors_ref.new_zeros(hashtable_size) - 1
        values = ref_bxyz.new_empty(hashtable_size, 4)
        reverse_indices = voxel_coors_ref.new_zeros(hashtable_size)

        # hashing
        hash_insert_gpu(
            keys,
            values,
            reverse_indices,
            dims,
            voxel_coors_ref,
            ref_bxyz)

        edges = radius_graph_gpu(
                    keys,
                    values,
                    reverse_indices,
                    dims,
                    voxel_coors_query,
                    query_bxyz,
                    self.qmin,
                    self.qmax,
                    radius,
                    self.max_num_neighbors,
                    False).T
        
        e0, e1 = edges
        u = e0 * (e1.max()+1) + e1
        u = u.unique()
        e0 = torch.div(u, e1.max()+1, rounding_mode='trunc').long()
        e1 = u % (e1.max()+1)

        return e0, e1, None
    
    def extra_repr(self):
        return f"voxel_size={self.voxel_size}, kernel_offset={self.kernel_offset}"


class VolumeGraph(VoxelGraph):
    def __init__(self, runtime_cfg, model_cfg):
        super(VolumeGraph, self).__init__(
                                     runtime_cfg=runtime_cfg,
                                     model_cfg=model_cfg,
                                 )
        self.use_volume_weight = model_cfg.get("USE_VOLUME_WEIGHT", False)

    def compute_l1_center(self, p):
        mean_proj = (p.l1_proj_min + p.l1_proj_max) / 2 # [V, 3]
        l1_center = p.bxyz[:, 1:] + (p.eigvecs @ mean_proj[:, :, None]).squeeze(-1) # [V, 3] + [V, 3]
        return l1_center

    def compute_proj(self, p, e, diff):
        eigwidth = (p.l1_proj_max - p.l1_proj_min)[e].clamp(min=1e-2) / 2
        eigproj = (diff[:, :, None] * p.eigvecs[e]).sum(dim=1).abs() # [E, 3]
        mask = eigproj > eigwidth
        eigproj[mask] = eigwidth[mask]

        l = p.eigvals.clamp(min=1e-8).sqrt()
        dist = (l[e] * eigproj).norm(p=2, dim=-1)
        
        return dist
    
    def build_graph(self, ref, query):
        e_ref, e_query, e_weight = super(VolumeGraph, self).build_graph(ref, query)
        if self.use_volume_weight:
            ref_l1_center = self.compute_l1_center(ref)
            query_l1_center = self.compute_l1_center(query)
            diff = ref_l1_center[e_ref] - query_l1_center[e_query] # [E, 3]
            l1 = self.compute_proj(ref, e_ref, diff)
            l2 = self.compute_proj(query, e_query, diff)
            dist = (diff.norm(p=2, dim=-1) - l1 - l2).clamp(min=0)
            center_dist = (ref.bcenter[e_ref] - query.bcenter[e_query]).norm(p=2, dim=-1).clamp(min=1e-4) / 2
            e_weight = center_dist.pow(2) / (dist.pow(2) + center_dist.pow(2))
            try:
                assert not e_weight.isnan().any()
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(e)
        else:
            e_weight = None

        return e_ref, e_query, e_weight


class KNNGraphV2(GraphV2Template):
    def __init__(self, runtime_cfg, model_cfg):
        super(KNNGraphV2, self).__init__(
                                    runtime_cfg=runtime_cfg,
                                    model_cfg=model_cfg,
                                )
        self.k = model_cfg.get("NUM_NEIGHBORS", 32)
        self.reweight_key = model_cfg.get("REWEIGHT_KEY", None)
        sigma = model_cfg.get("SIGMA", 5.0)
        self.sigma2 = sigma * sigma
    
    def build_graph(self, ref, query):
        """Build knn graph from ref point cloud to query point cloud,
            each query point connects to k ref points.
        Args:
            ref_bxyz [N, 4]: ref point cloud
            query_bxyz [M, 4]: query point cloud
        Returns:
            edge_idx [2, M*K]: (idx_of_ref, idx_of_query)
        """
        ref_bxyz = ref.bxyz
        query_bxyz = query.bxyz

        ref_xyz, ref_indices, ref_offset = bxyz_to_xyz_index_offset(ref_bxyz)
        query_xyz, query_indices, query_offset = bxyz_to_xyz_index_offset(query_bxyz)
        
        # building a bipartite graph from [N] to [M]
        #N, M = ref_xyz.shape[0], query_xyz.shape[0]
        ref_idx, _ = knnquery(self.k, ref_xyz, query_xyz,
                              ref_offset.int(), query_offset.int()) # [M, k] -> [N]
        ref_idx = ref_idx.long()

        query_idx = torch.arange(query_xyz.shape[0])[:, None].expand(-1, self.k).to(ref_idx) # [M, k] -> [M]

        e_ref, e_query = torch.stack([ref_indices[ref_idx],
                                      query_indices[query_idx]], dim=0).reshape(2, -1)
        if self.reweight_key is not None:
            dist2 = (ref[self.reweight_key][e_ref] - query[self.reweight_key][e_query]).square().sum(dim=-1)
            median_dist2 = dist2.median()
            #e_weight = torch.ones_like(dist2)
            #e_weight[dist2 > self.sigma2] = 0.0
            #e_weight = self.sigma2 / (dist2 + self.sigma2)
            e_weight = median_dist2 / (dist2 + median_dist2)
        else:
            e_weight = None

        return e_ref, e_query, e_weight
    
    def extra_repr(self):
        return f"num_neighbors={self.k}"


GRAPHS = {
    'KNNGraph': KNNGraph,
    'KNNGraphV2': KNNGraphV2,
    'RadiusGraph': RadiusGraph,
    'VoxelGraph': VoxelGraph,
    'VolumeGraph': VolumeGraph,
}

