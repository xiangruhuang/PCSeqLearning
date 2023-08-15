import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
import os
from torch_scatter import scatter
from torch_cluster import knn

from .preprocessor_utils import ground_plane_removal
from pcdet.utils import common_utils
from pcdet.utils.timer import Timer
from pcdet.models.model_utils import graph_utils
from pcdet.ops.voxel import VoxelAggregation

def plane_analysis(points, planes, e_plane, num_planes, cfg): #dist_thresh, count_gain, decision_thresh):
    # number of points within distance threshold `dist_thresh`
    valid_mask = (points.plane_dist < cfg.dist_thresh).float()
    plane_count = scatter(valid_mask, e_plane, dim=0, dim_size=num_planes, reduce='sum')

    # fitting error (weighted)
    plane_error = scatter(points.plane_dist*points.weight, e_plane, dim=0, dim_size=num_planes, reduce='sum')
    plane_weight_sum = scatter(points.weight, e_plane, dim=0, dim_size=num_planes, reduce='sum')
    plane_mean_error = plane_error / (plane_weight_sum + 1e-5)

    # compute fitness
    plane_fitness = (plane_count * cfg.count_gain).clamp(max=0.55) + (cfg.decision_thresh / (cfg.decision_thresh + plane_mean_error)).clamp(max=0.55)

    planes.fitness = plane_fitness
    planes.mean_error = plane_mean_error
    
    return points, planes

def pca_fitting(ref_points, e_plane, cfg): #stride, k, dist_thresh, count_gain, sigma, decision_thresh):
    num_planes = e_plane.max().long().item()+1
    # plane fitting
    points, planes = ransac(ref_points.bxyz, e_plane, num_planes, cfg.sigma)
    
    # evaluate fitness of planes
    points, planes = plane_analysis(points, planes, e_plane, num_planes, cfg) #dist_thresh, count_gain, decision_thresh)
    sigvals = planes.eigvals.clamp(min=0).sqrt()
    diff = planes.l1_proj_max - planes.l1_proj_min
    #plane_mask = (planes.fitness > 1.0) & (sigvals[:, 0] < 0.03) & (sigvals[:, 1] > 0.4) & (sigvals[:, 2] > 0.8)
    plane_mask = (planes.fitness > 1.0) & (diff[:, 0] < 0.1) & (diff[:, 1] > 1.) & (diff[:, 2] > 1.5)
    point_mask = planes.fitness[e_plane] > 1.0
    point_mask &= points.weight > 0.5
    planes['weight'] = scatter(point_mask.float(), e_plane, dim=0, dim_size=num_planes, reduce='sum') / planes.degree
    points['weight'] = 1.0 / planes.degree[e_plane] # overwrite
    
    # transform plane id
    map2new_id = torch.zeros(num_planes, dtype=torch.long).to(ref_points.bxyz.device) - 1
    map2new_id[plane_mask] = torch.arange(plane_mask.long().sum()).to(ref_points.bxyz.device)
    points.plane_id = map2new_id[e_plane]

    #planes = common_utils.apply_to_dict(planes, lambda x: x.numpy())
    planes = EasyDict(common_utils.filter_dict(planes, plane_mask))
    e0, e1 = knn(planes.bxyz[:, 1:], planes.bxyz[:, 1:], k=5, batch_x=planes.bxyz[:, 0].round().long(), batch_y=planes.bxyz[:, 0].round().long())
    dist = (planes.bxyz[e0, 1:3] - planes.bxyz[e1, 1:3]).norm(p=2, dim=-1).clamp(min=1)
    mask = (planes.bxyz[e0, -1] > planes.bxyz[e1, -1] + dist*0.5).long()
    remove_mask = scatter(mask, e0, dim=0, dim_size=planes.bxyz.shape[0], reduce='max').bool()
    planes = EasyDict(common_utils.filter_dict(planes, ~remove_mask))
    points.pop('plane_dist')
    ref_points.update(points)

    if 'bcenter' in ref_points:
        planes.bcenter = scatter(ref_points.bcenter, e_plane, dim=0, dim_size=num_planes, reduce='mean')

    #points = common_utils.filter_dict(points, ~point_mask)
    #points = common_utils.apply_to_dict(points, lambda x: x.numpy())
    return ref_points, planes


def ransac(point_bxyz, e_plane, num_planes, sigma, stopping_delta=1e-2):
    """
    Args:
        point_bxyz [N, 4]: point coordinates, batch index is the first dimension
        e_plane [N]: partition group id of each point
        num_planes (integer): number of partitions (planes), (denoted as P)
        sigma2: reweighting parameters
        stopping_delta: algorithmic parameter

    Returns:
        points: point-wise dictionary {
            weight [N]: indicating the likelihood of this point belong to a plane,
                        higher means more likely
            coords [N, 3]: the local rank coordinates
        }
        planes: plane-wise dictionary {
            bxyz [P, 4]: center coordinates (b,x,y,z) per plane
            eigvals [P, 3]: per plane PCA eigenvalues
            eigvecs [P, 3, 3]: per plane PCA eigenvectors
            normal [P, 3]: per plane normal vector
        }
    """
    point_weight = torch.ones(point_bxyz.shape[0], dtype=torch.float32, device=point_bxyz.device)
    sigma2 = sigma*sigma
    plane_degree = scatter(torch.ones_like(point_weight).long(), e_plane, dim=0, dim_size=num_planes, reduce='sum')
    
    for itr in range(100):
        # compute plane center
        plane_bxyz = scatter(point_bxyz*point_weight[:, None], e_plane, dim=0, dim_size=num_planes, reduce='sum')
        plane_weight_sum = scatter(point_weight, e_plane, dim=0, dim_size=num_planes, reduce='sum')
        plane_bxyz = plane_bxyz / (plane_weight_sum[:, None] + 1e-6)

        # compute
        point_d = point_bxyz[:, 1:] - plane_bxyz[e_plane, 1:]
        point_ddT = point_d[:, None, :] * point_d[:, :, None] * point_weight[:, None, None]
        plane_ddT = scatter(point_ddT, e_plane, dim=0, dim_size=num_planes, reduce='mean')
        eigvals, eigvecs = torch.linalg.eigh(plane_ddT)
        plane_normal = eigvecs[:, :, 0]
        p2plane_dist = (point_d * plane_normal[e_plane]).sum(-1).abs()
        new_point_weight = sigma2 / (p2plane_dist ** 2 +sigma2)
        delta_max = (new_point_weight - point_weight).abs().max()
        point_weight = new_point_weight
        if delta_max < stopping_delta:
            break
    
    point_coords = torch.stack([torch.ones_like(point_weight),
                                (eigvecs[e_plane, :, 1] * point_d).sum(-1),
                                (eigvecs[e_plane, :, 2] * point_d).sum(-1)], dim=-1)
    #point_coords[:, 1:] = point_coords[:, 1:] - point_coords[:, 1:].min(0)[0]
    #point_coords[:, 1:] /= point_coords[:, 1:].max(0)[0].clamp(min=1e-5)
    
    plane_max0 = scatter((point_d * eigvecs[e_plane, :, 0]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min0 = scatter((point_d * eigvecs[e_plane, :, 0]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='min')
    plane_max1 = scatter((point_d * eigvecs[e_plane, :, 1]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min1 = scatter((point_d * eigvecs[e_plane, :, 1]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='min')
    plane_max2 = scatter((point_d * eigvecs[e_plane, :, 2]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='max')
    plane_min2 = scatter((point_d * eigvecs[e_plane, :, 2]).sum(-1), e_plane, dim=0, dim_size=num_planes, reduce='min')

    l1_proj_min = torch.stack([plane_min0, plane_min1, plane_min2], dim=-1)
    l1_proj_max = torch.stack([plane_max0, plane_max1, plane_max2], dim=-1)

    points = EasyDict(
        weight=point_weight,
        coords=point_coords,
        plane_dist=p2plane_dist,
    )

    planes = EasyDict(
        bxyz=plane_bxyz,
        degree=plane_degree,
        eigvals=eigvals,
        eigvecs=eigvecs,
        normal=plane_normal,
        l1_proj_min=l1_proj_min,
        l1_proj_max=l1_proj_max,
    )

    return points, planes

class GroundPlaneRemover(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.fake_params = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        self.forward_dict = EasyDict()

    def output_stats(self, segmentation_label, ground_mask, sequence_id, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        num_removed_foreground_points = ((segmentation_label[ground_mask] > 0) & (segmentation_label[ground_mask] <= 7)).sum().item()
        #num_removed_background_points = (segmentation_label[ground_mask] > 7).sum().item()
        num_removed_ground_points = (segmentation_label[ground_mask] >= 17).sum().item()
        num_removed_points = ground_mask.long().sum().item()

        num_foreground_points = ((segmentation_label > 0) & (segmentation_label <= 7)).sum().item()
        #num_background_points = (segmentation_label > 7).sum().item()
        num_ground_points = (segmentation_label >= 17).sum().item()
        
        ground_point_coverage = num_removed_ground_points / (num_ground_points + 1e-6)
        ground_point_precision = num_removed_ground_points / (num_removed_points + 1e-6)
        foreground_point_precision = num_removed_foreground_points / (num_removed_points + 1e-6)
        foreground_point_coverage = num_removed_foreground_points / (num_foreground_points + 1e-6)

        with open(f'{log_dir}/{sequence_id}.txt', 'w') as fout:
            fout.write(f'{self.model_cfg}\n')
            fout.write(f'#removed_points={num_removed_points}\n')
            fout.write(f'#removed_foreground={num_removed_foreground_points}\n')
            fout.write(f'#removed_ground={num_removed_ground_points}\n')
            fout.write(f'ground_precision={ground_point_precision:.6f}\n')
            fout.write(f'ground_coverage={ground_point_coverage:.6f}\n')
            fout.write(f'foreground_precision={foreground_point_precision:.6f}\n')
            fout.write(f'foreground_coverage={foreground_point_coverage:.6f}\n')

    def forward(self, seq_dict):
        frame_id = seq_dict['frame_id'][0]
        sequence_id = frame_id[:-4]
        point_sweep = seq_dict['point_sweep']
        point_fxyz = seq_dict['point_fxyz']
        sweeps = EasyDict(
            sxyz=point_fxyz,
            feat=seq_dict['point_feat'],
            #segmentation_label=seq_dict['segmentation_label'],
            #instance_label=seq_dict['instance_label'],
        )
        for key in ['segmentation_label', 'instance_label']:
            if key in seq_dict:
                sweeps[key] = seq_dict[key]

        point_height = point_fxyz.new_zeros(point_fxyz.shape[0])
        path = f'{self.model_cfg.DIR}/{sequence_id}'
        if os.path.exists(f'{path}/pillar_height.pth'):
            saved_dict = torch.load(f'{path}/pillar_height.pth', map_location=point_height.device)
            print(f'loading ground plane for sequence {sequence_id}')
            point_height, point_horizon, point_error, pillar_height, pillar_min_z = \
                    ground_plane_removal(point_fxyz, self.model_cfg, warmup=saved_dict)
        #if os.path.exists(f'{path}/ground_plane.txt'):
        #    # LOAD the point height and truncate with TRUNCATE_HEIGHT
        #    print(f'loading ground plane for sequence {sequence_id}')
        #    for sweep in range(point_sweep.min().item(), point_sweep.max().item()+1):
        #        sweep_mask = (point_sweep == sweep).reshape(-1)
        #        import ipdb; ipdb.set_trace()
        #        point_height_s = np.load(f'{path}/{sweep:04d}_point_height.npy').astype(np.int32) / 100
        #        point_height_s = torch.from_numpy(point_height_s).to(point_height)
        #        point_height[sweep_mask] = point_height_s
        else:
            with Timer('Ground Removal'):
                point_height, point_horizon, point_error, pillar_height, pillar_min_z = ground_plane_removal(point_fxyz, self.model_cfg)
            output_dir = path
            os.makedirs(output_dir, exist_ok=True)
            torch.save(dict(pillar_height=pillar_height, pillar_min_z=pillar_min_z), f'{path}/pillar_height.pth')
            #for sweep in range(point_sweep.min().item(), point_sweep.max().item()+1):
            #    sweep_mask = (point_sweep == sweep).reshape(-1)
            #    if not sweep_mask.any():
            #        continue
            #    hh = point_height[sweep_mask].detach().cpu().numpy()
            #    hh = np.floor(np.clip(hh, a_min=0.0, a_max=1.0) * 100).astype(np.int32).clip(min=0, max=255).astype(np.uint8)
            #    common_utils.save_as_npy(hh, f'{output_dir}/{sweep:04d}_point_height.npy')
        seq_dict['point_horizon'] = point_horizon
        seq_dict['point_error'] = point_error

        for height in self.model_cfg.TRUNCATE_HEIGHT:
            ground_mask = point_height < height
            LOG_DIR = self.model_cfg.LOG_DIR+f'/height{height}'
            if 'segmentation_label' in sweeps:
                self.output_stats(sweeps.segmentation_label, ground_mask, sequence_id, LOG_DIR)

        seq_dict['point_height'] = point_height
        seq_dict['ground_mask'] = ground_mask

        print(f'Removing Ground: {ground_mask.numel()} --> {ground_mask.sum().item()}')
        for key in ['point_fxyz', 'segmentation_label', 'point_sweep',
                    'point_height', 'instance_label', 'point_horizon']:
            if key in seq_dict:
                seq_dict[f'full_{key}'] = seq_dict[key].clone()
                seq_dict[key] = seq_dict[key][~ground_mask]
        seq_dict.pop('ground_mask')

        return seq_dict

    def extra_repr(self):
        return f"{self.model_cfg}"

    def get_output_feature_dim(self):
        return 0

if __name__ == '__main__':
    from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', type=str, default=None)
    parser.add_argument('pcsequence_file', type=str, default=None)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    runtime_cfg = dict()
    remover = GroundPlaneRemover(cfg, runtime_cfg)
    if args.pcsequence_file.endswith('.npy'):
        pcseq = torch.from_numpy(np.load(args.pcsequence_file)).cuda()
    seq_points = EasyDict(dict(
                     point_fxyz=pcseq,
                     point_sweep=pcseq[:, 0].long(),
                     segmentation_label=torch.zeros_like(pcseq[:, 0]).long(),
                     instance_label=torch.zeros_like(pcseq[:, 0]).long(),
                     point_feat=torch.zeros_like(pcseq[:, 1:]),
                     frame_id='fake_000',
                 ))
    seq_points = remover(seq_points)

    import polyscope as ps
    ps.set_up_dir('z_up')
    ps.set_ground_plane_mode('none')
    ps.init()
    ps_seq = ps.register_point_cloud('pcseq', seq_points.point_fxyz[:, 1:].detach().cpu(), radius=2e-4)
    ps_seq.add_scalar_quantity('frame', seq_points.point_sweep.detach().cpu())
    ps_seq.add_scalar_quantity('point_height', seq_points.point_height.detach().cpu())
    ps_seq.add_scalar_quantity('point_height > 0.25', seq_points.point_height.detach().cpu() > 0.25)
    ps_seq.add_scalar_quantity('point_height > 0.5', seq_points.point_height.detach().cpu() > 0.5)
    ps_seq.add_scalar_quantity('point_height > 0.75', seq_points.point_height.detach().cpu() > 0.75)
    ps_seq.add_scalar_quantity('point_height > 0.15', seq_points.point_height.detach().cpu() > 0.15)

    import ipdb; ipdb.set_trace()
    ps.show()

