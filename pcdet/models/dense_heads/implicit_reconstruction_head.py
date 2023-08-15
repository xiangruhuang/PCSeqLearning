import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils, polar_utils
from torch_scatter import scatter

from .reconstruction_head_template import ReconstructionHeadTemplate
from ...models.blocks import MLP
from pcdet.ops.torch_hash import RadiusGraph

class ImplicitReconstructionHead(ReconstructionHeadTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg)
        input_channels = runtime_cfg["num_point_features"]

        self.model_cfg = model_cfg

        channels = self.model_cfg.get("CHANNELS", None)
        self.latent_dim = channels[-1]
        channels[0] += 3 # for appending xyz
        self.mlp = MLP(channels)
        self.point_bxyz_key = self.input_key + '_bxyz'
        self.point_feature_key = self.input_key + '_feat'

        self.occupancy = nn.Sequential(
                             nn.Linear(channels[-1], 1),
                         )

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.radius_graph = RadiusGraph(ndim=2)
        self.num_samples = model_cfg.get("NUM_SAMPLES", None)
        self.num_samples_per_dim = int(self.num_samples ** (1/3.0))
        self.num_samples = self.num_samples_per_dim ** 3
        self.radius = model_cfg.get("RADIUS", None) 
        self.spherical_radius = model_cfg.get("SPHERICAL_RADIUS", None)
        self.occupancy_certainty_decay = model_cfg.get("OCCUPANCY_CERTAINTY_DECAY", None)
        self.forward_dict = {}

    def build_losses(self, losses_cfg):
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'BCELogits':
            self.reg_loss_func = nn.BCEWithLogitsLoss(reduction='none')
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss

    def ball_sample(self, points):
        """
        Args:
            points [N, 3]
            (self.num_samples)
            (self.radius)

        Returns:
            perturbed_points [N, num_samples, 3]
            perturbation [N, num_samples, 3]
        """
        noise_x = torch.linspace(-self.radius/2.0, self.radius/2.0, self.num_samples_per_dim)
        noise_y = torch.linspace(-self.radius/2.0, self.radius/2.0, self.num_samples_per_dim)
        noise_z = torch.linspace(-self.radius/2.0, self.radius/2.0, self.num_samples_per_dim)
        noise = torch.meshgrid(noise_x, noise_y, noise_z)
        noise = torch.stack(noise, dim=-1).view(-1, 3).to(points).repeat(points.shape[0], 1, 1)
        #noise = torch.rand(points.shape[0], self.num_samples, 3).to(points) # [0, 1)
        #noise = (noise - 0.5) * (self.radius / (3 ** 0.5))
        #noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4)
        #noise = noise / noise_norm * noise_norm.clamp(max=self.radius)

        perturbed_points = points.clone().unsqueeze(1)
        perturbed_points = perturbed_points.expand(points.shape[0], self.num_samples, 3).contiguous()
        perturbed_points += noise

        return perturbed_points, noise

    def compute_occupancy(self, lidar_xyz, sampled_xyz, rho, e_ref, e_query):
        """Estimate if sampled points are visible by the lidar camera
        Args:
            lidar_xyz [N, 3]
            sampled_xyz [N*num_samples, 3]
            rho [N]
            e_ref [M] in range [0, N)
            e_query [M] in range [0, N*num_samples)

        Return:
            occupancy [N*num_samples]
            occupancy_certainty [N*num_samples]
        """
        num_queries = sampled_xyz.shape[0]
        projected_rho = rho[e_ref]

        lidar_dir = lidar_xyz[e_ref]
        lidar_dir = lidar_dir / lidar_dir.norm(p=2, dim=-1, keepdim=True)

        sampled_xyz = sampled_xyz[e_query]
        projected_xyz = (sampled_xyz * lidar_dir).sum(dim=-1, keepdim=True) * lidar_dir
        projected_dist = projected_xyz.norm(p=2, dim=-1)

        occupancy = projected_rho <= projected_dist
        dist_gap = projected_dist - projected_rho
        assert e_query.max() < num_queries
        occupancy = scatter(occupancy.float(), e_query, dim=0,
                            dim_size=num_queries, reduce='sum')
        occupancy = occupancy.long().clamp(max=1)
        
        occupancy_certainty = (self.occupancy_certainty_decay - dist_gap
                               ).clamp(min=0) / self.occupancy_certainty_decay
        occupancy_certainty = scatter(occupancy_certainty, e_query, dim=0,
                                      dim_size=num_queries, reduce='sum').clamp(max=1, min=0)
        
        return occupancy, occupancy_certainty

    def get_loss(self):
        gt_occupancy = self.forward_dict['gt_occupancy']
        pred_occupancy_logits = self.forward_dict['pred_occupancy_logits']
        pred_occupancy = self.forward_dict['pred_occupancy']

        occupancy_certainty = self.forward_dict['occupancy_certainty']

        loss_src = self.reg_loss_func(pred_occupancy_logits, gt_occupancy.float())
        loss = (loss_src * occupancy_certainty).mean()
        occupancy_acc = self.forward_dict['correctness'].mean().item()

        tb_dict = dict(
            occupancy_acc=occupancy_acc
        )
        
        return loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            lidar_bxyz: [N, 4] the entire point cloud, the first dim 
                               represents batch index
            query_bxyz: [M, 4] the positions to sample around,
                               the first dim represents batch index
        """

        lidar_bxyz = batch_dict['point_bxyz']
        lidar_xyz = lidar_bxyz[:, 1:4].clone() # [N, 3(xyz)]
        lidar_batch_idx = lidar_bxyz[:, 0].round().long()

        query_bxyz = batch_dict[self.point_bxyz_key]
        query_xyz = query_bxyz[:, 1:4].clone() # [M, 3(xyz)]
        query_batch_idx = query_bxyz[:, 0].round().long()
        batch_size = batch_dict['batch_size']
        
        # compute coordinates relative to top lidar origin
        batch_dict['top_lidar_origin'] = torch.cat([batch_dict['top_lidar_origin'].new_zeros(1, 1), batch_dict['top_lidar_origin']], dim=-1)[None, :].expand(batch_size, -1, -1)
        top_lidar_origin = batch_dict['top_lidar_origin'].reshape(batch_size, -1, 4) # [num_batch, num_sweeps, 4(bxyz)]
        top_lidar_origin = top_lidar_origin[:, 0, 1:4] # [num_batch, 3(xyz)], taking the first sweep of every batch
        lidar_xyz -= top_lidar_origin[lidar_batch_idx]
        query_xyz -= top_lidar_origin[query_batch_idx]

        # retrieve spherical coordinates
        rho, polar, azimuth = polar_utils.cartesian2spherical(lidar_xyz)
        lidar_spherical = torch.stack([lidar_batch_idx, polar, azimuth], dim=-1)
        
        # perturb each point with radius `self.radius`
        sampled_xyz, sampled_noise = self.ball_sample(query_xyz) # [M, num_querys, 3]
        sampled_xyz, sampled_noise = sampled_xyz.view(-1, 3), sampled_noise.view(-1, 3)
        sampled_batch_idx = query_batch_idx.repeat_interleave(self.num_samples)

        # convert each point coordinates into spherical coordinates
        sampled_rho, sampled_polar, sampled_azimuth = polar_utils.cartesian2spherical(sampled_xyz)
        sampled_spherical = torch.stack([sampled_batch_idx, sampled_polar, sampled_azimuth], axis=-1)

        # find the nearest projection onto lidar lines
        e_ref, e_query = self.radius_graph(lidar_spherical, sampled_spherical,
                                           self.spherical_radius, 1, sort_by_dist=True)
        spherical_dist = (lidar_spherical[e_ref] - sampled_spherical[e_query]).norm(p=2, dim=-1)
        spherical_dist_query = spherical_dist.new_full(sampled_xyz.shape[:1], 1000)
        spherical_dist_query[e_query] = spherical_dist
        spherical_certainty = (self.spherical_radius - spherical_dist_query).clamp(min=0) / self.spherical_radius
        
        # compute occupancy of each sampled location
        gt_occupancy, occupancy_certainty = self.compute_occupancy(lidar_xyz, sampled_xyz,
                                                                   rho, e_ref, e_query)
        occupancy_certainty = occupancy_certainty * spherical_certainty
        sampled_xyz += top_lidar_origin[sampled_batch_idx]

        batch_dict['sampled_bxyz'] = torch.cat([sampled_batch_idx.unsqueeze(-1),
                                                sampled_xyz], dim=-1)
        batch_dict['spherical_dist'] = spherical_dist_query
        batch_dict['occupancy_certainty'] = occupancy_certainty
        batch_dict['spherical_certainty'] = spherical_certainty

        query_features = batch_dict[self.point_feature_key]
        sampled_features = query_features.repeat_interleave(self.num_samples, dim=0)

        feat_pos = torch.cat([sampled_features, sampled_noise], axis=-1) # [N, num_samples, C + 3]
        occupancy_feature = self.mlp(feat_pos) # [N*num_samples, 1]
        pred_occupancy_logits = self.occupancy(occupancy_feature).squeeze(-1) # [N*num_samples]
        pred_occupancy = nn.Sigmoid()(pred_occupancy_logits)
        
        batch_dict['query_bxyz'] = query_bxyz
        for i in [0, 1000, 3000]:
            batch_dict[f'query_feat_dist_{i}'] = (query_features - query_features[i]).norm(p=2, dim=-1).clamp(max=10).exp()
        batch_dict['query_bxyz_center'] = query_bxyz[[0,1000, 3000], :]

        self.forward_dict['pred_occupancy_logits'] = pred_occupancy_logits
        self.forward_dict['pred_occupancy'] = pred_occupancy
        self.forward_dict['gt_occupancy'] = gt_occupancy
        self.forward_dict['pred_occupancy'] = pred_occupancy.round().long()
        self.forward_dict['correctness'] = (pred_occupancy.round().long() == gt_occupancy).float()
        self.forward_dict['occupancy_certainty'] = spherical_certainty

        batch_dict.update(self.forward_dict)

        return batch_dict
