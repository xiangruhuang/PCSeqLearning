import torch
from torch import nn
from .torch_hash_cuda import (
    hash_insert_gpu,
    correspondence,
    radius_graph_gpu,
    points_in_radius_gpu
)

class RadiusGraph(nn.Module):
    def __init__(self,
                 max_num_points=400000,
                 ndim=3):
        super().__init__()
        self.ndim = ndim

        # dummy variable
        qmin = torch.tensor([0] + [-1 for i in range(ndim)]).int()
        qmax = torch.tensor([0] + [1 for i in range(ndim)]).int()
        self.register_buffer("qmin", qmin, persistent=False)
        self.register_buffer("qmax", qmax, persistent=False)

    def clear(self):
        self.keys[:] = -1

    @torch.no_grad()
    def forward(self, ref, query, radius, num_neighbors, sort_by_dist=False):
        """
        Args:
            ref [N, 1+dim] the first dimension records batch index
            query [M, 1+dim] ..
            radius (float)
            num_neighbors (int)
            sort_by_dist 

        Returns:
            edge_indices [2, E] each column represent edge (idx_of_ref, idx_of_query)
        """
        max_num_points = ref.shape[0] * 2
        keys = torch.zeros(max_num_points).long().to(ref.device)
        values = torch.zeros(max_num_points, self.ndim+1).float().to(ref.device)
        reverse_indices = torch.zeros(max_num_points).long().to(ref.device)
        #self.register_buffer("keys", keys, persistent=False)
        #self.register_buffer("values", values, persistent=False)
        #self.register_buffer("reverse_indices", reverse_indices, persistent=False)

        #assert ref.shape[0] * 2 <= self.max_num_points, f"Too many points, shape={ref.shape[0]} > {self.max_num_points//2}"

        if isinstance(radius, float):
            radius = query.new_zeros(query.shape[0]) + radius
        elif isinstance(radius, int):
            radius = query.new_zeros(query.shape[0]) + radius

        voxel_size = torch.tensor([1-1e-3] + [radius.max().item() for i in range(self.ndim)]).to(ref.device)
        assert ref.shape[1] == self.ndim + 1, f"points must have {self.ndim+1} dimensions"
        all_points = torch.cat([ref, query], axis=0)
        pc_range_min = (all_points.min(0)[0] - voxel_size*2).cuda()
        pc_range_max = (all_points.max(0)[0] + voxel_size*2).cuda()
        
        ref = ref.cuda()
        query = query.cuda()
        voxel_coors_ref = torch.round((ref-pc_range_min) / voxel_size).long() + 1
        voxel_coors_query = torch.round((query-pc_range_min) / voxel_size).long() + 1
        dims = torch.round((pc_range_max - pc_range_min) / voxel_size).long()+3

        keys[:] = -1
        #self.clear()
        
        hash_insert_gpu(
            keys,
            values,
            reverse_indices,
            dims,
            voxel_coors_ref,
            ref)

        edges = radius_graph_gpu(
                    keys,
                    values,
                    reverse_indices,
                    dims,
                    voxel_coors_query,
                    query,
                    self.qmin,
                    self.qmax,
                    radius,
                    num_neighbors,
                    sort_by_dist).T

        return edges

    def extra_repr(self):
        return f"ndim={self.ndim}"


class ChamferDistance(nn.Module):
    def __init__(self, max_num_points=400000, ndim=3, radius_graph=None):
        super().__init__()
        if radius_graph is not None:
            self.radius_graph = radius_graph
            self.ndim = radius_graph.ndim
            self.max_num_points = radius_graph.max_num_points
        else:
            self.radius_graph = RadiusGraph(max_num_points, ndim=ndim)
            self.ndim = ndim
            self.max_num_points = max_num_points
        
    def forward(self, src_bxyz, target_bxyz, radius):
        """Chamfer Distance between src_bxyz and target_bxyz.
            CD = 1/N dist_src_to_target + 1/M dist_target_to_src

        Args:
            src_bxyz: [N, 1+ndim]
            target_bxyz: [M, 1+ndim]
            radius: ignore edges with distance larger than radius

        Returns:
            dist: Chamfer Distance
        """
        fwd_src, fwd_target = self.radius_graph(src_bxyz, target_bxyz, radius, 1, sort_by_dist=True)
        bwd_target, bwd_src = self.radius_graph(target_bxyz, src_bxyz, radius, 1, sort_by_dist=True)

        dist_fwd = (src_bxyz[fwd_src] - target_bxyz[fwd_target]).square().sum(-1).mean()
        dist_bwd = (src_bxyz[bwd_src] - target_bxyz[bwd_target]).square().sum(-1).mean()

        return dist_fwd + dist_bwd

    def __repr__(self):
        return f"ChamferDistance(ndim={self.ndim}, max_npoints={self.max_num_points})"

if __name__ == '__main__':
    #rg = RadiusGraph().cuda()
    #points = torch.randn(100, 4).cuda() * 3
    #points[:, 0] = 0
    #er, eq = rg(points, points, 2, -1)
    #assert (points[er] - points[eq]).norm(p=2, dim=-1).max() < 2
    #print('Test 1 okay')
    #
    #rg = RadiusGraph(ndim=2).cuda()
    #points = torch.randn(100, 3).cuda() * 3
    #points[:, 0] = 0
    #er, eq = rg(points, points, 2, -1)
    #assert (points[er] - points[eq]).norm(p=2, dim=-1).max() < 2
    #print('Test 2 okay')

    rg = RadiusGraph(ndim=2).cuda()
    points = torch.tensor([[0, 0.0, 0.0], [0, 0.1, 0.1], [0, 0.2, 0.2]], dtype=torch.float32).cuda()
    eq, er = rg(points, points, 0.15, 1, sort_by_dist=True)
    #assert eq == [0, 1, 2]
    #assert er == [0, 1, 2]
    print(eq, er)

    from sklearn.neighbors import NearestNeighbors as NN
    import numpy as np
    data = np.load('data.npy')
    #data = np.random.randn(1000, 4)
    data[:, 0] = 0
    tree = NN(n_neighbors=2).fit(data)
    dists, indices = tree.kneighbors(data)
    indices = indices[:, 1]
    dists = dists[:, 1]
    rg = RadiusGraph(ndim=3).cuda()
    data_p = torch.from_numpy(data).cuda().float()
    er, eq = rg(data_p, data_p, 0.5, 2, sort_by_dist=True)
    mask = (eq != er)
    er = er[mask]
    eq = eq[mask]
    for edge_r, edge_q in zip(er, eq):
        dist = (data_p[edge_r] - data_p[edge_q]).norm(p=2)
        try:
            assert (dist - dists[edge_q]).abs() < 1e-3 or (edge_r == indices[edge_q])
        except Exception as e:
            print(edge_q, edge_r, indices[edge_q], (dist - dists[edge_q]).abs(), dists[edge_q])
            print(e)
            assert False
            break

    print('Test sklearn Okay')
    pass

