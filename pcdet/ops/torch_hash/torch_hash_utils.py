import torch
from .torch_hash_cuda import (
    hash_insert_gpu,
    correspondence,
    radius_graph_gpu,
    points_in_radius_gpu
)

class HashTable:
    def __init__(self, size=2 ** 20, dtype=torch.float32, device='cuda:0'):
        self.size = size
        self.keys = torch.zeros(size, dtype=torch.long) - 1
        self.values = torch.zeros(size, 4, dtype=dtype)
        self.reverse_indices = torch.zeros(size, dtype=torch.long)
        self.device = device
        self.qmin = torch.zeros(4, dtype=torch.int32).to(self.device) - 1
        self.qmax = torch.zeros(4, dtype=torch.int32).to(self.device) + 1
        self.keys = self.keys.to(self.device)
        self.values = self.values.to(self.device)
        self.reverse_indices = self.reverse_indices.to(self.device)
        self.corres_pool_size = 10000
        self.corres = torch.zeros(self.corres_pool_size,
                                  dtype=torch.long,
                                  device=self.device)-1

        rp = torch.tensor([999269, 999437, 999377], dtype=torch.long)
        self.rp0, self.rp1, self.rp2 = rp

    def clear(self):
        self.keys[:] = -1

    def points_in_radius_step2(self, query_points, temporal_offset,
                               radius=0.5):
        """

        Args:
            query_points (M, D): query points
            temporal_offset: offset in temporal (last) dimension.

        Returns:
            eq (M): corresponding point index in query_points
            er (M): corresponding point index in ref_points

        """
        try:
            voxel_size = self.voxel_size
            ndim = self.ndim
            pc_range = self.pc_range
        except Exception as e:
            raise ValueError(f'make sure you call hash_into_gpu first: {e}')
        
        query_points = query_points.cuda()

        voxel_coors_query = torch.round(
                                (query_points-pc_range[:ndim]) / voxel_size
                            ).long() + 1
        
        self.qmin[-1] = temporal_offset
        self.qmax[-1] = temporal_offset

        if not hasattr(self, 'visited'):
            self.visited = torch.zeros(self.num_points, dtype=torch.long,
                                       device=self.device)
        else:
            self.visited[:] = 0

        # look up points from hash table
        points_in_radius_gpu(self.keys, self.values, self.reverse_indices,
                             self.dims, voxel_coors_query,
                             query_points, self.qmin, self.qmax,
                             radius, self.visited)

        point_indices = torch.where(self.visited)[0]

        return point_indices
    
    #def hash(self, voxel_coors):
    #    """

    #    Args:
    #        voxel_coors [N, 4]: voxel coordinates

    #    Returns:
    #        keys: [N] integers
    #        hash_indices: [N] integers 

    #    """
    #    insert_keys = torch.zeros_like(voxel_coors[:, 0])
    #    indices = torch.zeros_like(insert_keys)
    #    for i in range(voxel_coors.shape[1]):
    #        insert_keys = insert_keys * self.dims[i] + voxel_coors[:, i]
    #        indices = indices * self.dims[i] + voxel_coors[:, i]
    #        indices = (indices * self.rp0 + self.rp1) 

    #    indices = indices % self.size
    #    return insert_keys.to(torch.long), indices.to(torch.long)

    def find_voxels(self, keys):
        """

        Args:
            keys [N]: integers

        Returns:
            voxel_coors [N, 4]: voxel coordinates
        
        """
        voxel_coors = []
        for i in range(3, -1, -1):
            voxel_coors.append(keys % dims[i])
            keys = keys / dims[i]
        voxel_coors = torch.stack(voxel_coors, dim=-1)
        return voxel_coors

    @torch.no_grad()
    def find_corres(self, ref_points, query_points, voxel_size,
                    pc_range, temporal_offset):
        """

        Args:
            ref_points (N, D): reference points
            query_points (M, D): query points
            temporal_offset: offset in temporal (last) dimension.

        Returns:
            eq (M): corresponding point index in query_points
            er (M): corresponding point index in ref_points

        """
        ref_points = ref_points.cuda()
        query_points = query_points.cuda()
        voxel_size = voxel_size.cuda()

        self.ndim = ndim = ref_points.shape[-1]
        points = torch.cat([ref_points, query_points], dim=0)
        points[:] = torch.max(
                        torch.min(points, pc_range[ndim:]),
                        pc_range[:ndim])
        voxel_coors = torch.round((points-pc_range[:ndim]) / voxel_size
                                 ).long() + 1
        
        self.dims = ((pc_range[ndim:] - pc_range[:ndim]) / voxel_size).long()+3
        #self.dims = (voxel_coors.max(0)[0] - voxel_coors.min(0)[0]) + 3
        
        self.keys[:] = -1
        # hash points into hash table
        hash_insert_gpu(self.keys, self.values, self.reverse_indices, self.dims,
                        voxel_coors[:ref_points.shape[0]], ref_points)
        
        self.qmin[-1] = temporal_offset
        self.qmax[-1] = temporal_offset

        if query_points.shape[0] > self.corres_pool_size:
            self.corres_pool_size = query_points.shape[0]*2
            self.corres = torch.zeros(self.corres_pool_size,
                                      dtype=torch.long,
                                      device=self.device)-1
        else:
            self.corres[:] = -1
        corres = self.corres

        # look up points from hash table
        correspondence(self.keys, self.values, self.reverse_indices,
                       self.dims, voxel_coors[ref_points.shape[0]:],
                       query_points, self.qmin, self.qmax, corres)

        corres = corres[:query_points.shape[0]]
        mask = (corres != -1)
        corres0 = torch.where(mask)[0]
        corres = torch.stack([corres0, corres[mask]], dim=0)

        return corres
    
    @torch.no_grad()
    def hash_into_gpu(self, ref_points, voxel_size, pc_range):
        """Hash Points into GPU

        Args:
            ref_points (N, D): reference points

        Returns:

        """
        if ref_points.device == torch.device('cpu'):
            ref_points = ref_points.cuda()
        self.voxel_size = voxel_size = voxel_size.clone().cuda()
        self.pc_range = pc_range = pc_range.clone().cuda()

        points = ref_points
        self.ndim = ndim = points.shape[-1]
        voxel_coors = torch.round((points-pc_range[:ndim]) / voxel_size
                                 ).long() + 1
        
        self.dims = ((pc_range[ndim:]-pc_range[:ndim]) / voxel_size).long()+3
        
        self.keys[:] = -1
        self.num_points = ref_points.shape[0]
        # hash points into hash table
        hash_insert_gpu(self.keys, self.values, self.reverse_indices,
                        self.dims, voxel_coors, points)
    
    @torch.no_grad()
    def find_corres_step2(self, query_points, temporal_offset):
        """

        Args:
            ref_points (N, D): reference points

        Returns:
            eq (M): corresponding point index in query_points
            er (M): corresponding point index in ref_points

        """
        pc_range = self.pc_range
        voxel_size = self.voxel_size
        ndim = self.ndim

        query_points = query_points.cuda()

        voxel_coors = torch.round(
                          (query_points-pc_range[:ndim]) / voxel_size
                      ).long() + 1
        
        self.qmin[-1] = temporal_offset
        self.qmax[-1] = temporal_offset

        if query_points.shape[0] > self.corres_pool_size:
            self.corres_pool_size = query_points.shape[0]*2
            self.corres = torch.zeros(self.corres_pool_size,
                                      dtype=torch.long,
                                      device=self.device)-1
        else:
            self.corres[:query_points.shape[0]] = -1
        corres = self.corres

        # look up points from hash table
        correspondence(self.keys, self.values, self.reverse_indices,
                       self.dims, voxel_coors,
                       query_points, self.qmin, self.qmax, corres)

        corres = corres[:query_points.shape[0]]
        corres0 = torch.where(corres != -1)[0]
        corres = torch.stack([corres0, corres[corres0]], dim=0)

        return corres 

    @torch.no_grad()
    def voxel_graph(self, ref_points, query_points, voxel_size,
                    pc_range, temporal_offset,
                    radius=0.5, max_num_neighbors=32):
        """

        Args:
            ref_points (N, D): reference points
            query_points (M, D): query points
            temporal_offset: offset in temporal (last) dimension.

        Returns:
            eq (M): corresponding point index in query_points
            er (M): corresponding point index in ref_points

        """
        assert ref_points.shape[0] + query_points.shape[0] < self.size * 2
        ref_points = ref_points.cuda()
        query_points = query_points.cuda()
        voxel_size = voxel_size.cuda()
        pc_range = pc_range.cuda()

        self.ndim = ndim = ref_points.shape[-1]

        voxel_coors_ref = torch.round(
                              (ref_points-pc_range[:ndim]) / voxel_size
                          ).long() + 1
        voxel_coors_query = torch.round(
                                (query_points-pc_range[:ndim]) / voxel_size
                            ).long() + 1

        self.dims = ((pc_range[ndim:]-pc_range[:ndim]) / voxel_size).long()+3
        
        self.keys[:] = -1
        # hash points into hash table
        hash_insert_gpu(self.keys, self.values, self.reverse_indices, self.dims,
                        voxel_coors_ref, ref_points)
        
        self.qmin[-1] = temporal_offset
        self.qmax[-1] = temporal_offset

        if query_points.shape[0]*max_num_neighbors > self.corres_pool_size:
            self.corres_pool_size = query_points.shape[0]*max_num_neighbors*2
            self.corres = torch.zeros(self.corres_pool_size,
                                      dtype=torch.long,
                                      device=self.device)-1
        else:
            self.corres[:(query_points.shape[0]*max_num_neighbors)] = -1
        corres = self.corres

        # look up points from hash table
        voxel_graph_gpu(self.keys, self.values, self.reverse_indices,
                        self.dims, voxel_coors_query,
                        query_points, self.qmin, self.qmax,
                        max_num_neighbors, radius, corres)

        #corres = corres.cpu()
        corres = corres[:(query_points.shape[0]*max_num_neighbors)]
        corres = corres.view(-1, max_num_neighbors)
        mask = (corres != -1)
        corres0, corres1 = torch.where(mask)
        corres = torch.stack([corres0, corres[(corres0, corres1)]], dim=0)

        return corres
    
    @torch.no_grad()
    def voxel_graph_step2(self, query_points, temporal_offset,
                          radius=0.5, max_num_neighbors=32):
        """

        Args:
            ref_points (N, D): reference points
            query_points (M, D): query points
            temporal_offset: offset in temporal (last) dimension.

        Returns:
            eq (M): corresponding point index in query_points
            er (M): corresponding point index in ref_points

        """
        try:
            voxel_size = self.voxel_size
            ndim = self.ndim
            pc_range = self.pc_range
        except Exception as e:
            raise ValueError(f'make sure you call hash_into_gpu first: {e}')
        query_points = query_points.cuda()

        voxel_coors_query = torch.round(
                                (query_points-pc_range[:ndim]) / voxel_size
                            ).long() + 1
        
        self.qmin[-1] = temporal_offset
        self.qmax[-1] = temporal_offset

        if query_points.shape[0]*max_num_neighbors > self.corres_pool_size:
            self.corres_pool_size = query_points.shape[0]*max_num_neighbors*3//2
            self.corres = torch.zeros(self.corres_pool_size,
                                      dtype=torch.long,
                                      device=self.device)-1
        else:
            self.corres[:] = -1
        corres = self.corres

        # look up points from hash table
        voxel_graph_gpu(self.keys, self.values, self.reverse_indices,
                        self.dims, voxel_coors_query,
                        query_points, self.qmin, self.qmax,
                        max_num_neighbors, radius, corres)

        corres = corres[:(query_points.shape[0]*max_num_neighbors)
                        ].view(-1, max_num_neighbors)
        mask = (corres != -1)
        corres0, corres1 = torch.where(mask)
        corres = torch.stack([corres0, corres[(corres0, corres1)]], dim=0)

        return corres

def test_corres(ht, n_ref, n_query, ndim, radius):
    import time
    ref = torch.randn(n_ref, ndim)
    query = torch.randn(n_query, ndim)
    ref[:, -1] = 0
    query[:, -1] = 0
    pall = torch.cat([ref, query], dim=0)
    pc_range = torch.cat(
        [pall.min(0)[0]-3, pall.max(0)[0]+3], dim=0
    ).cuda()

    voxel_size = torch.tensor([radius, radius, radius, radius])
    start_time = time.time()
    eq, er = ht.find_corres(ref, query, voxel_size, pc_range, 0)
    end_time = time.time()
    dist = (ref[er] - query[eq]).norm(p=2, dim=-1)
    er = er[dist <= radius]
    eq = eq[dist <= radius]
    
    tree = NN(n_neighbors=1).fit(ref)
    dists, index_r = tree.kneighbors(query)
    dists, index_r = dists[:, 0], index_r[:, 0]
    index_q = torch.arange(query.shape[0])
    index_r = index_r[dists < radius]
    index_q = index_q[dists < radius]

    assert (er.cpu()-index_r).abs().sum() < 1e-5
    assert (eq.cpu()-index_q).abs().sum() < 1e-5

    return end_time - start_time

def test_vgraph(ht, n_ref, n_query, ndim, radius, n_ngbr=16):
    import time
    ref = torch.randn(n_ref, ndim)
    query = torch.randn(n_query, ndim)
    ref[:, -1] = 0
    query[:, -1] = 0
    pall = torch.cat([ref, query], dim=0)
    pc_range = torch.cat(
        [pall.min(0)[0]-3, pall.max(0)[0]+3], dim=0
    ).cuda()

    voxel_size = torch.tensor([radius, radius, radius, radius])
    start_time = time.time()
    eq, er = ht.voxel_graph(ref, query, voxel_size, pc_range, 0,
                            radius=radius, max_num_neighbors=n_ngbr)
    end_time = time.time()
    dist = (ref[er] - query[eq]).norm(p=2, dim=-1)
    er = er[dist <= radius]
    eq = eq[dist <= radius]
    
    tree = NN(n_neighbors=n_ngbr).fit(ref)
    dists, index_r = tree.kneighbors(query)
    dists, index_r = dists.reshape(-1), torch.tensor(index_r.reshape(-1))
    index_q = torch.arange(query.shape[0]).repeat(n_ngbr, 1).T.reshape(-1)
    index_r = index_r[dists < radius]
    index_q = index_q[dists < radius]

    from torch_scatter import scatter
    checksum1 = scatter(index_r, index_q, dim=0, dim_size=query.shape[0],
                        reduce='sum')
    checksum2 = scatter(er, eq, dim=0, dim_size=query.shape[0],
                        reduce='sum')

    assert (checksum1 - checksum2.cpu()).abs().sum() < 1e-5

    return end_time - start_time

def test_multistep_corres(ht, n_ref, n_query, ndim, radius):
    import time
    ht.clear()
    ref = torch.randn(n_ref, ndim)
    ref[:, -1] = 0
    voxel_size = torch.tensor([radius, radius, radius, radius])
    pc_range = torch.cat(
        [ref.min(0)[0]-3, ref.max(0)[0]+3], dim=0
    ).cuda()
    ht.hash_into_gpu(ref, voxel_size, pc_range)
    
    total_time = 0
    for i in range(1):
        query = torch.randn(n_query, ndim)
        query[:, -1] = 0
        query_cuda = query.cuda()
        total_time -= time.time()
        eq, er = ht.find_corres_step2(query_cuda, 0)
        total_time += time.time()
        dist = (ref[er] - query[eq]).norm(p=2, dim=-1)
        er = er[dist <= radius]
        eq = eq[dist <= radius]
        
        tree = NN(n_neighbors=1).fit(ref)
        dists, index_r = tree.kneighbors(query)
        dists, index_r = dists.reshape(-1), index_r.reshape(-1)
        index_q = torch.arange(query.shape[0])
        index_r = index_r[dists < radius]
        index_q = index_q[dists < radius]

        #assert (er.cpu()-index_r).abs().sum() < 1e-5
        #assert (eq.cpu()-index_q).abs().sum() < 1e-5

    return total_time / 3.0
        
def test_multistep_vgraph(ht, n_ref, n_query, ndim, radius, n_ngbr=16):
    import time
    ht.clear()
    ref = torch.randn(n_ref, ndim)
    ref[:, -1] = 0
    voxel_size = torch.tensor([radius, radius, radius, radius])
    pc_range = torch.cat(
        [ref.min(0)[0]-3, ref.max(0)[0]+3], dim=0
    ).cuda()
    ht.hash_into_gpu(ref, voxel_size, pc_range)
    
    total_time = 0
    for i in range(3):
        query = torch.randn(n_query, ndim)
        query[:, -1] = 0
        query_cuda = query.cuda()
        total_time -= time.time()
        eq, er = ht.voxel_graph_step2(query_cuda, 0, radius=radius,
                                      max_num_neighbors=n_ngbr)
        total_time += time.time()
        dist = (ref[er] - query[eq]).norm(p=2, dim=-1)
        er = er[dist <= radius]
        eq = eq[dist <= radius]
        
        tree = NN(n_neighbors=n_ngbr).fit(ref)
        dists, index_r = tree.kneighbors(query)
        dists, index_r = dists.reshape(-1), torch.tensor(index_r.reshape(-1))
        index_q = torch.arange(query.shape[0]).repeat(n_ngbr, 1).T.reshape(-1)
        index_r = index_r[dists < radius]
        index_q = index_q[dists < radius]

        from torch_scatter import scatter
        checksum1 = scatter(index_r, index_q, dim=0, dim_size=query.shape[0],
                            reduce='sum')
        checksum2 = scatter(er, eq, dim=0, dim_size=query.shape[0],
                            reduce='sum')

        assert (checksum1 - checksum2.cpu()).abs().sum() < 1e-5

    return total_time / 3.0

if __name__ == '__main__':
    from sklearn.neighbors import NearestNeighbors as NN
    ht = HashTable(2**21)
    radius = 0.01
    n_ref, n_query = 1000000, 100
    ndim = 4

    for i in range(3):
        time = test_corres(ht, n_ref, n_query, ndim, radius)
        print(f'corres test {i:03d}: time={time:.6f}')

    for i in range(3):
        time = test_multistep_corres(ht, n_ref, n_query, ndim, radius)
        print(f'multi-step test {i:03d}, time={time:.6f}')
    
    for i in range(3):
        time = test_vgraph(ht, n_ref, n_query, ndim, radius, 16)
        print(f'vgraph test {i:03d}: time={time:.6f}')
    
    for i in range(3):
        time = test_multistep_vgraph(ht, n_ref, n_query, ndim, radius, 16)
        print(f'multi-step vgraph test {i:03d}: time={time:.6f}')
