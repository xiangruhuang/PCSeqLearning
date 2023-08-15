import torch
from torch import nn
from .hybrid_geop_cuda import (
    hash_insert_gpu,
    hybrid_geop_gpu
)
    
def get_unique_voxel_coors(voxel_coors, dims):
    """
    Args:
       voxel_coors [N, 4]
       dims [4]

    Returns:
        unique_coors [V, 4]
        group_ids [N]: in range [V]
    """
    voxel_coors1d = torch.zeros_like(voxel_coors[:, 0]).long()
    ndim = dims.shape[0]
    for i in range(ndim):
        voxel_coors1d = voxel_coors1d * dims[i] + voxel_coors[:, i]
    unique_coors1d, group_ids = voxel_coors1d.unique(return_inverse=True)
    unique_coors = unique_coors1d.new_zeros(unique_coors1d.shape[0], ndim)
    for i in range(ndim-1, -1, -1):
        unique_coors[:, i] = unique_coors1d % dims[i]
        unique_coors1d = torch.div(unique_coors1d, dims[i], rounding_mode='floor').long()
    return unique_coors, group_ids

class PrimitiveFitting(nn.Module):
    """Fit hybrid geometric primitives to point cloud.

    Algorithm:
        1. Given a set of points (b, x, y, z), we first compute their voxel
           coordinates (b, vx, vy, vz) based on provided 3D grid size.
        2. We hash each point into hash table using a hash key computed from (b, vx, vy, vz)
        

    Args:
        points [N, 4]
        edge_indices [2, E]

    Internal Args:
        ht_size (int) : size of hash table, should be at least 2x the maximum number of points.
        keys [H]: the hashtable, storing the hashed key for each inserted point
                  the hashed key is computed based on 4D coordinate (b, vx, vy, vz)
        values [H, 4]: store the actual (b, x, y, z) values

    Returns:
        
    """
    def __init__(self,
                 grid_size,
                 max_num_points=400000,
                 ):
        super(PrimitiveFitting, self).__init__()
        self.ndim = 3
        self.init_hashtable(max_num_points)
        
        # the size of voxel/grid in 4D
        if isinstance(grid_size, list):
            voxel_size = torch.tensor([1] + grid_size).float()
        else:
            voxel_size = torch.tensor([1, grid_size, grid_size, grid_size]).float()
        self.register_buffer('voxel_size', voxel_size)
        
        # query range in voxels
        qmin = torch.tensor([0, -1, -1, -1]).long()
        qmax = torch.tensor([0,  1,  1,  1]).long()
        self.register_buffer('qmin', qmin, persistent=False)
        self.register_buffer('qmax', qmax, persistent=False)

        # controls how weight decays with distance
        self.decay_radius = self.voxel_size[1:].norm(p=2).item() / 2
    
    def init_hashtable(self, max_num_points):
        self.ht_size = max_num_points*2
        self.max_num_points = max_num_points

        # the hash key
        ht_keys = torch.zeros(self.ht_size).long()
        self.register_buffer("ht_keys", ht_keys, persistent=False)

        # the hash value [H, 4]
        ht_values = torch.zeros(self.ht_size, self.ndim+1).float()
        self.register_buffer("ht_values", ht_values, persistent=False)

        # store the original index of the hashed element
        ht_reverse_indices = torch.zeros(self.ht_size).long()
        self.register_buffer("ht_reverse_indices", ht_reverse_indices, persistent=False)

    def hash(self, points):
        """Hash points into hashtable.

        Args:
            points [N, D]: the first dimension is batch index, followed by three channels
                           representing (x, y, z) coordinates, D >= 4.

        Args updated:
            self.ht_keys
            self.ht_values
            self.ht_reverse_indices

        Returns:
            voxel_coors [N, 4] the voxel coordinates (dtype=torch.long)

        """
        self.num_points = points.shape[0]
        assert self.num_points <= self.max_num_points, \
               f"Too many points, shape={points.shape[0]}"
        assert points.shape[1] >= self.ndim + 1, \
               f"points must have at least {self.ndim+1} dimensions"
        
        self.ht_keys[:] = -1 # -1 represent empty by protocol
        self.ht_reverse_indices[:] = -1

        points = points[:, :4].contiguous()
        pc_range_min = (points.min(0)[0] - self.voxel_size*2)
        pc_range_max = (points.max(0)[0] + self.voxel_size*2)

        voxel_coors = torch.round(
                          (points - pc_range_min) / self.voxel_size
                      ).long()+1

        dims = torch.round((pc_range_max - pc_range_min) / self.voxel_size).long()+3
        
        hash_insert_gpu(
            self.ht_keys,
            self.ht_values,
            self.ht_reverse_indices,
            dims,
            voxel_coors,
            points)

        return voxel_coors, dims

    def fit_primitive(self, primitive_coors, dims, num_primitives):
        """
        Args:
            primitive_coors [N, 4]: voxel coordinate of each potential primitive
            dims [4]: dimensionality of each axis
            num_primitives (int)

        Returns:
            mu
            cov_inv
        """
        # parameter of primitives
        mu = torch.zeros(num_primitives, self.ndim+1, dtype=torch.float).to(primitive_coors.device)
        cov = torch.zeros(num_primitives, 3, 3, dtype=torch.float).to(primitive_coors.device)

        hybrid_geop_gpu(
            self.ht_keys,
            self.ht_values,
            self.ht_reverse_indices,
            dims,
            self.qmin,
            self.qmax,
            primitive_coors,
            mu,
            cov,
            self.decay_radius
        )
        cov = (cov + cov.transpose(1, 2))/2.0
        R, S, V = cov.svd()
        S = S.sqrt().clamp(min=1e-6)
        det = R.det()
        flip_mask = (det == -1)
        R[flip_mask, :, 2] *= -1
        R = R * S[:, None, :]
        
        return mu, R

    @torch.no_grad()
    def forward(self, points):
        """
        Args:
            points [N, D]: the first dimension is batch index

        Returns:
            mu [P, 4]: primitive center
            cov_inv [P, 3, 3]: primitive covariance matrix
        """
        
        # hash points into hash table
        voxel_coors, dims = self.hash(points)

        # find unique voxel coordinates
        unique_voxel_coors, group_ids = get_unique_voxel_coors(voxel_coors, dims) # [U, 4]
        num_primitives = unique_voxel_coors.shape[0]
        mu, cov_inv = self.fit_primitive(unique_voxel_coors, dims, num_primitives)

        return mu, cov_inv, group_ids


if __name__ == '__main__':
    n = 1000
    points = torch.randn(n, 3).cuda()
    points[:, 2] = 0
    zeros = torch.zeros(n).cuda()
    points = torch.cat([zeros[:, None], points], axis=-1)
    pf = PrimitiveFitting(0.1)
    pf = pf.cuda()
    pf(points)
