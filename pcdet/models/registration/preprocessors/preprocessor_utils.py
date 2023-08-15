import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

from torch_scatter import scatter, segment_coo
from torch_cluster import knn

from pcdet.utils import common_utils
from .registration_utils import (
    efficient_robust_sum,
    efficient_robust_mean
)
from pcdet.utils.timer import Timer
from pcdet.models.model_utils.grid_sampling import GridSampling3D
from pcdet.ops.voxel.voxel_modules import VoxelAggregation


def grid_sample(point_bxyz, grid_size):
    grid_sampler = GridSampling3D(grid_size=grid_size).cuda()
    _point_bxyz = point_bxyz.clone()
    _point_bxyz[:, 0] = 0
    _, inv = grid_sampler(_point_bxyz, return_inverse=True)
    _point_bxyz = scatter(_point_bxyz, inv, dim_size=inv.max().long()+1, dim=0, reduce='mean')
    voxels = EasyDict(
        bxyz=_point_bxyz,
    )
    return voxels, inv

def iterative_reweighted_ransac(points, pillars, w0, num_pillars, sigma2, stopping_delta=1e-2):
    """
    Args:
        points.bxyz [N, 4]: (batch, x, y, z)
        pillars: dict()
        num_pillars: number of pillars
        w0 [N, 1]: per-point weight
    Returns:
        points.plane_fitting_error [N]: dist from point to the final plane
    """
    assert len(w0.shape) == 2, "w0 must be a two dimensional tensor"
    w = w0
    point_xyz = points.bxyz[:, 1:4]
    for itr in range(50):
        center = segment_coo(
                     point_xyz*w,
                     points.pillar_idx,
                     dim_size=num_pillars,
                     reduce='sum'
                 )
        weight_sum = segment_coo(
                         w,
                         points.pillar_idx,
                         dim_size=num_pillars,
                         reduce='sum'
                     ) + 1e-6
        pillars.center = center / weight_sum

        point_d = point_xyz - pillars.center[points.pillar_idx]
        point_ddT = (w[:, :, None] * point_d[:, :, None]) * point_d[:, None, :]
        pillar_cov = segment_coo(
                         point_ddT,
                         points.pillar_idx,
                         dim_size=num_pillars,
                         reduce='mean'
                     )

        S, Q = torch.linalg.eigh(pillar_cov)
        pillars.normal = Q[:, :, 0]
        points.plane_fitting_error = (point_d * pillars.normal[points.pillar_idx]).sum(-1).abs()
        new_w = (sigma2 / (points.plane_fitting_error.square() + sigma2)).reshape(-1)
        dist_w = (0.5**2)/(point_d.square().sum(dim=-1) + 0.5**2)
        new_w = (new_w * dist_w).reshape(-1, 1)
        delta_w = (new_w - w).abs()
        if delta_w.max() < stopping_delta:
            break
        w = new_w

    return points, pillars


@torch.no_grad()
def compute_min_height_from_ransac(pillar_dims, num_pillars, voxels, pillars, cfg, window_size=4):
    """Estimate pillars.min_z by plane reconstruction.
    Args:

    Returns:

    """
    # map pillars into larger pillars
    _pillar_coords_x = torch.arange(num_pillars).to(voxels.pillar_coords) // pillar_dims[1]
    _pillar_coords_y = torch.arange(num_pillars).to(voxels.pillar_coords) % pillar_dims[1]
    pillars.pillar_coords = torch.stack([_pillar_coords_x, _pillar_coords_y], dim=-1) // window_size
    new_pillar_dims = pillars.pillar_coords.max(0)[0].long() + 1
    num_new_pillars = new_pillar_dims[1]*new_pillar_dims[0]

    # index of new pillars for each old pillar
    pillars.pillar_idx = pillars.pillar_coords[:, 0] * new_pillar_dims[1] + pillars.pillar_coords[:, 1]
    pillars.best_confidence = torch.zeros_like(pillars.density).reshape(-1)
    pillars.best_normal = pillars.min_z.new_zeros(num_pillars, 3)
    pillars.best_center = pillars.min_z.new_zeros(num_pillars, 3)

    voxel_min_z = pillars.min_z[(voxels.pillar_coords[:, 0], voxels.pillar_coords[:, 1])]
    voxels.possibly_ground = (voxels.bxyz[:, -1] - voxel_min_z < 0.5)

    # re-assign voxels into new pillars (different pillar size)
    new_voxels = EasyDict(
        bxyz=voxels.bxyz,
    )
    new_pillars = EasyDict()
    new_voxels.pillar_coords = voxels.pillar_coords // window_size
    new_voxels.pillar_idx = new_voxels.pillar_coords[:, 0] * new_pillar_dims[1] + new_voxels.pillar_coords[:, 1]
    argsort_by_pillar_idx = new_voxels.pillar_idx.argsort()
    new_voxels = EasyDict(common_utils.filter_dict(new_voxels, argsort_by_pillar_idx))
    new_pillars.min_z = scatter(new_voxels.bxyz[:, -1], new_voxels.pillar_idx, dim=0,
                                dim_size=num_new_pillars, reduce='min')
    new_pillars.max_z = scatter(new_voxels.bxyz[:, -1], new_voxels.pillar_idx, dim=0,
                                dim_size=num_new_pillars, reduce='max')
    new_pillars.best_confidence = torch.zeros_like(new_pillars.min_z).reshape(-1)
    new_pillars.best_normal = new_pillars.min_z.new_zeros(num_new_pillars, 3)
    new_pillars.best_normal[:, -1] = 1.0
    new_pillars.best_center = new_pillars.min_z.new_zeros(num_new_pillars, 3)

    #new_pillar_size = pillar_size * window_size
    #half_window_size = window_size // 2
    #cur_pillar_size = pillar_size * window_size
    #for offset_x in [0, half_window_size]:
    #    for offset_y in [0, half_window_size]:
    #        offset = torch.tensor([offset_x, offset_y]).to(pillar_coords)

    #cur_pillar_coords = (points.pillar_coords) // window_size
    #cur_pillar_dims = cur_pillar_coords.max(0)[0].long() + 1
    #cur_pillar_idx = cur_pillar_coords[:, 0] * cur_pillar_dims[1] + cur_pillar_coords[:, 1]
    #cur_num_pillars = cur_pillar_dims[1] * cur_pillar_dims[0]
    #cur_pillar_min_z = scatter(points.bxyz[:, -1], cur_pillar_idx, dim=0,
    #                           dim_size=cur_num_pillars, reduce='min'
    #                          )
    #cur_pillar_max_z = scatter(points.bxyz[:, -1], cur_pillar_idx, dim=0,
    #                           dim_size=cur_num_pillars, reduce='max'
    #                          )

    #_cur_pillar_coords = (_pillar_coords) // window_size
    #_cur_pillar_idx = _cur_pillar_coords[:, 0] * cur_pillar_dims[1] + _cur_pillar_coords[:, 1]

    #pillar_confidence = torch.zeros_like(pillars.min_z).reshape(-1)
    for ratio in tqdm(torch.linspace(0.3, 1, 30)):
        cur_z = new_pillars.min_z * ratio + new_pillars.max_z * (1-ratio)
        #point_center = cur_pillar_coords * cur_pillar_size + pc_range_min
        #point_center = torch.cat([point_center, cur_z.reshape(-1, 1)], dim=-1)
        z_diff = (cur_z[new_voxels.pillar_idx] - new_voxels.bxyz[:, -1])
        w0 = (cfg.SIGMA2 / (z_diff.square() + cfg.SIGMA2)).reshape(-1, 1)
        #sort_idx = cur_pillar_idx.argsort()
        #sorted_point_xyz = _point_bxyz[sort_idx, 1:]
        #sorted_cur_pillar_idx = cur_pillar_idx[sort_idx]
        #sorted_w = w[sort_idx]

        new_voxels, new_pillars = iterative_reweighted_ransac(new_voxels, new_pillars,
                                                              w0, num_new_pillars,
                                                              sigma2=cfg.SIGMA2)

        # update pillar best fit plane
        hit_mask = new_voxels.plane_fitting_error < (cfg.SIGMA2**0.5)
        num_hit = scatter(hit_mask.float(), new_voxels.pillar_idx,
                          dim=0, dim_size=num_new_pillars, reduce='sum')
        mask = new_pillars.best_confidence < num_hit
        if mask.any():
            new_pillars.best_normal[mask] = new_pillars.normal[mask]
            new_pillars.best_center[mask] = new_pillars.center[mask]
            new_pillars.best_confidence[mask] = num_hit[mask]

    covered = torch.zeros_like(pillars.best_confidence).long() - 1
    #sorted_confidence_idx = new_pillars.best_confidence.argsort(descending=True)
    
    #################################################################################################
    # Truncated Least Squares

    xyz, normal, conf = new_pillars.best_center, new_pillars.best_normal, new_pillars.best_confidence
    for threshold in np.logspace(np.log(5)/np.log(10), np.log(0.01)/np.log(10), 100):
        e0, e1 = knn(xyz, xyz, k=cfg.K)
        xyz_diff = xyz[e1] - xyz[e0]
        point_to_plane_dist = (xyz_diff * normal[e0]).sum(dim=-1).abs()
        curvature = point_to_plane_dist / (xyz_diff.norm(p=2, dim=-1) + 1e-4)
        mean_curvature = curvature.reshape(-1, cfg.K).mean(-1)
        if threshold > mean_curvature.max():
            continue
        else:
            valid_mask = mean_curvature < threshold
            xyz = xyz[valid_mask]
            normal = normal[valid_mask]
            conf = conf[valid_mask]

    conf[:] = 1
    #################################################################################################
    #import polyscope as ps; ps.init(); ps.set_up_dir('z_up')
    #ps_v = ps.register_point_cloud('voxels', new_voxels.bxyz[:, 1:].detach().cpu(), radius=2e-4)
    #
    #ps_c4 = ps.register_point_cloud('new pillar centers', xyz.detach().cpu(), radius=3e-3)
    #ps_c4.add_vector_quantity('normal', normal.detach().cpu())
    #ps_c4.add_scalar_quantity('confidence', conf.detach().cpu())

    #voxel_normal = pillars.best_normal[voxels.pillar_idx]
    #voxel_center = pillars.best_center[voxels.pillar_idx]
    #voxel_diff = voxels.bxyz[:, 1:] - voxel_center
    #voxel_normal_z = voxel_normal[:, -1].abs().clamp(min=0.1) * ((voxel_normal[:, -1] >= 0).float() + 1) / 2
    #voxel_height = (voxel_diff * voxel_normal).sum(-1) / voxel_normal_z
    #pillar_min_z = scatter(voxels.bxyz[:, -1] - voxel_height, voxels.pillar_idx, dim=0,
    #                       dim_size=num_pillars, reduce='min').reshape(pillar_dims[0], pillar_dims[1])
    #pillar_xyz = pillars.xyz.clone()
    #pillar_xyz[:, -1] = pillar_min_z.reshape(-1)
    #ps_c3 = ps.register_point_cloud('local_center before prop', pillar_xyz.detach().cpu(), radius=3e-3)
    #ps_c3.add_vector_quantity('normal', pillars.best_normal.detach().cpu())
    #ps_c3.add_scalar_quantity('confidence', pillars.best_confidence.detach().cpu())
    #################################################################################################

    for idx in tqdm(range(conf.shape[0])): #tqdm(sorted_confidence_idx):
        center = xyz[idx].clone()
        nor = normal[idx].clone()
        confidence = conf[idx]
        dist = (pillars.xyz[:, :2] - center[:2]).norm(p=2, dim=-1)
        confidence_ind = confidence / (dist.pow(1.0) + 1)
        mask = confidence_ind > pillars.best_confidence
        pillars.best_center[mask] = center
        pillars.best_normal[mask] = nor
        pillars.best_confidence[mask] = confidence_ind[mask]
        #covered[mask] = idx
    
    #################################################################################################
    #import polyscope as ps; ps.init(); ps.set_up_dir('z_up')
    #ps_p = ps.register_point_cloud('points', voxels.bxyz[:, 1:].detach().cpu(), radius=2e-4)

    #ps_c = ps.register_point_cloud('center', pillars.best_center.detach().cpu(), radius=6e-3)
    #ps_c.add_vector_quantity('normal', pillars.best_normal.detach().cpu())
    #ps_c.add_scalar_quantity('confidence', pillars.best_confidence.detach().cpu())
    #ps_c.add_scalar_quantity('covered', covered.detach().cpu())
    
    #################################################################################################
    voxel_normal = pillars.best_normal[voxels.pillar_idx]
    voxel_center = pillars.best_center[voxels.pillar_idx]
    voxel_diff = voxels.bxyz[:, 1:] - voxel_center
    voxel_normal_z = voxel_normal[:, -1].abs().clamp(min=0.01) * ((voxel_normal[:, -1] >= 0).float() + 1) / 2
    voxel_height = (voxel_diff * voxel_normal).sum(-1) / voxel_normal_z
    #################################################################################################
    ##ps_p.add_vector_quantity('normal', point_normal.detach().cpu(), enabled=False)
    #ps_p.add_scalar_quantity('height', voxel_height.detach().cpu())
    #ps.show()
    ##ps_p.add_scalar_quantity('height near 0', ((point_height < 0.2) & (point_height > -0.2)).detach().cpu())
    ##ps_p.add_scalar_quantity('height > 0.5', (point_height > 0.5).detach().cpu())
    ##ps_p.add_scalar_quantity('possibly_ground', possibly_ground.detach().cpu().float())
    ##ps_p.add_scalar_quantity('confidence', best_pillar_confidence[pillar_idx].detach().cpu())
    #################################################################################################

    pillars.min_z = scatter(voxels.bxyz[:, -1]-voxel_height, voxels.pillar_idx, dim=0,
                            dim_size=num_pillars, reduce='mean').reshape(pillar_dims[0], pillar_dims[1])
    pillars.height = pillars.min_z.clone()
    #pillars.weight[:] = pillars.density > 0.5) #pillars.best_confidence / 100.0
    
    #################################################################################################
    #pillar_xyz = pillars.xyz.clone()
    #pillar_xyz[:, -1] = pillars.min_z.reshape(-1)
    #ps_c2 = ps.register_point_cloud('local_center', pillar_xyz.detach().cpu(), radius=3e-3)
    #ps_c2.add_vector_quantity('normal', pillars.best_normal.detach().cpu())
    #ps_c2.add_scalar_quantity('confidence', pillars.best_confidence.detach().cpu())
    #ps_c2.add_scalar_quantity('covered', covered.detach().cpu())
    #ps_c2.add_color_quantity('color/covered', torch.randn(100000, 3)[covered.detach().cpu()])

    #ps.show()
    #import ipdb; ipdb.set_trace()
    #################################################################################################


    return voxels, pillars

def format_pillars(points, pillar_size, pc_range_min):
    """Computer per-pillar attributes from point clouds
    Args:
        points.bxyz [N, 4]: (batch_id, x, y, z) representing points
        pillar_size [2]: the size of each pillar in x and y dimension
        pc_range_min [3]: min (x,y,z) coordinate of the scene
    Returns:
      pillar_dims: [X, Y], dimension in each axis
      num_pillars: P = X*Y
      
      points = {
        ...
        pillar_coords [N, 2]: discrete pillar coordinates in x and y axis
        pillar_idx [N]: 1D index of the corresponding pillar (per-point)
      }
      pillars = {
        density [X, Y]: 1D index of each pillar
        xyz [P, 3]: mean x,y,z coordinate of each pillar
        min_z [X, Y]: the minimum height (z coordinate) of each pillar
        weight [P]: the weight of each pillar (for future optimization)
      }
    """
    pillars = EasyDict()
    points["pillar_coords"] = torch.div(points.bxyz[:, 1:3] - pc_range_min,
                                        pillar_size, rounding_mode='floor').round().long()
    pillar_dims = points.pillar_coords.max(0)[0].long() + 1
    num_pillars = pillar_dims[1]*pillar_dims[0]
    points["pillar_idx"] = points.pillar_coords[:, 0] * pillar_dims[1] + points.pillar_coords[:, 1]

    pillars["density"] = scatter(torch.ones_like(points.bxyz[:, 0]), points.pillar_idx, dim=0,
                                 dim_size=num_pillars, reduce='sum').reshape(pillar_dims[0], pillar_dims[1])
    pillars["min_z"] = scatter(points.bxyz[:, -1], points.pillar_idx, dim=0,
                               dim_size=num_pillars, reduce='min').reshape(pillar_dims[0], pillar_dims[1])
    pillars["xyz"] = scatter(points.bxyz[:, 1:], points.pillar_idx, dim=0,
                             dim_size=num_pillars, reduce='mean').reshape(-1, 3)
    pillars["weight"] = (pillars.density > 0.5).float().reshape(-1)

    return pillar_dims, num_pillars, points, pillars

def l1_minimization(points, pillars, pillar_dims, cfg, max_countdown=3):
    weight = pillars.weight.reshape(pillar_dims[0], pillar_dims[1])
    pillar_height = nn.Parameter(torch.zeros(pillar_dims[0], pillar_dims[1]).cuda(), requires_grad=True)
    optimizer = torch.optim.AdamW([pillar_height], lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.DECAY_STEPS)
    
    temp_points = EasyDict(
        bxyz=points.bxyz,
        pillar_idx=points.pillar_idx,
        pillar_coords=points.pillar_coords,
        density=pillars.density[(points.pillar_coords[:, 0], points.pillar_coords[:, 1])],
    )
    last_loss = 1e10
    countdown = max_countdown
    for itr in range(cfg.MAX_NUM_ITERS):
        optimizer.zero_grad()
        l1 = ((pillar_height - pillars.min_z)*weight).abs().mean()
        left = ((pillar_height[:-2]-2*pillar_height[1:-1]+pillar_height[2:])*(weight[1:-1]+1e-2)).abs().mean()
        up = ((pillar_height[:, :-2]-2*pillar_height[:, 1:-1]+pillar_height[:, 2:])*(weight[:, 1:-1]+1e-2)).abs().mean()
        t1 = ((pillar_height[:-2, :-2]-2*pillar_height[1:-1, 1:-1]+pillar_height[2:, 2:])*(weight[1:-1, 1:-1]+1e-2)).abs().mean()
        t2 = ((pillar_height[2:, :-2]-2*pillar_height[1:-1, 1:-1]+pillar_height[:-2, 2:])*(weight[1:-1, 1:-1]+1e-2)).abs().mean()
        loss = l1 + (left + up + t1 + t2)*cfg.RIGID_WEIGHT
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if (last_loss - loss.item()) < 1e-4:
            countdown -= 1
        else:
            countdown = 3

        if countdown == 0:
            break
        last_loss = loss.item()
        #print(f'iter={itr}, loss={loss.item()}, lr={lr:.6f}')

    pillars["height"] = pillar_height.data.clone()
    return pillars

def ground_plane_removal(point_bxyz, cfg, warmup=None):
    """
    Args:
        point_bxyz [N, 4]: a point cloud sequence, the first dimension represent frame id
        cfg.PILLAR_SIZE [2]: 2D pillar size in meters
    Returns:
        mask: ground points to be removed
    """
    # initialization
    points = EasyDict(
        bxyz=point_bxyz,
        original_indices=torch.arange(point_bxyz.shape[0]).to(point_bxyz.device),
    )
    pillar_size = torch.tensor(cfg.PILLAR_SIZE).to(points.bxyz)
    pc_range_min = points.bxyz[:, 1:3].min(0)[0] - 0.05
        
    # subsample via voxelization regardless of timestamp for acceleration
    voxels, point_voxel_index = grid_sample(points.bxyz, [0.10, 0.10, 0.03])

    # compute per-pillar attributes from sub-sampled voxels
    pillar_dims, num_pillars, voxels, pillars = \
            format_pillars(voxels, pillar_size, pc_range_min)

    if warmup is not None:
        pillars.height = warmup['pillar_height']
        pillars.min_z = warmup['pillar_min_z']
    else:
        # initialize pillars.min_z via robust optimization
        if cfg.get("RANSAC", False):
            print('Re-estimating Min Height')
            voxels, pillars = compute_min_height_from_ransac(pillar_dims, num_pillars, voxels, pillars, cfg)
        #else:
        #    pillar_min_z_flat = pillar_min_z.reshape(-1)
        #    pillar_min_z_final = pillar_min_z_flat.clone()
        #    pillar_width = pillar_min_z_flat.new_zeros(pillar_min_z_flat.shape[0])
        #    max_z = (point_bxyz[:, -1].max() - point_bxyz[:, -1].min()).float().item()
        #    for dev_z in torch.linspace(0, max_z, 401):
        #        pillar_z = pillar_min_z_flat + dev_z
        #        point_mask = (point_bxyz[:, -1] - pillar_z[pillar_idx]).abs() < 0.10
        #        pillar_deg = scatter(point_mask.float(), pillar_idx.reshape(-1), dim=0,
        #                             dim_size=num_pillars, reduce='sum')
        #        pillar_min = scatter(point_bxyz[point_mask, 1:3], pillar_idx[point_mask].reshape(-1), dim=0,
        #                             dim_size=num_pillars, reduce='min')
        #        pillar_max = scatter(point_bxyz[point_mask, 1:3], pillar_idx[point_mask].reshape(-1), dim=0,
        #                             dim_size=num_pillars, reduce='max')

        #        pillar_mask = (pillar_deg > 200) & ((pillar_max - pillar_min).norm(p=2, dim=-1) > pillar_width - 1e-2)
        #        if pillar_mask.any():
        #            pillar_width[pillar_mask] = (pillar_max - pillar_min).norm(p=2, dim=-1)[pillar_mask]
        #            pillar_min_z_final[pillar_mask] = pillar_z[pillar_mask]
        #            weight[pillar_mask] = 1
        #    pillar_min_z = pillar_min_z_final.reshape(pillar_min_z.shape)

        # jointly optimize ground height in each pillar
        if cfg.get("JointOpt", False):
            print('Joint Optimization for Ground Height...')
            pillars = l1_minimization(voxels, pillars, pillar_dims, cfg)

    voxel_height = pillars.height[(voxels.pillar_coords[:, 0], voxels.pillar_coords[:, 1])]
    voxel_min_z = pillars.min_z[(voxels.pillar_coords[:, 0], voxels.pillar_coords[:, 1])]
    voxel_horizon = voxels.bxyz[:, -1] > voxel_min_z

    #remove_mask = (point_bxyz[:, -1] - point_height < cfg.TRUNCATE_HEIGHT)
    voxel_height = voxels.bxyz[:, -1] - voxel_height
    fitting_error = (voxel_height - voxel_min_z)

    return voxel_height[point_voxel_index], voxel_horizon[point_voxel_index], \
           fitting_error[point_voxel_index], pillars.height, pillars.min_z
