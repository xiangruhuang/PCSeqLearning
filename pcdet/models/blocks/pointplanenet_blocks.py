import torch
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F
from functools import partial
from easydict import EasyDict

from .block_templates import (
    DownBlockTemplate,
    UpBlockTemplate,
)
from pcdet.utils import common_utils

def resample_points(points, planes, cfg):
    plane_id = points.plane_id
    mask = (plane_id != -1)
    num_planes = planes.bxyz.shape[0]
    
    plane_degree = scatter(points.feat.new_ones(mask.long().sum()), plane_id[mask],
                           dim=0, dim_size=num_planes, reduce='sum')
    coords = torch.rand([num_planes, cfg.num_points_per_plane, 2]).to(points.bxyz)
    coords = coords * planes.l1_proj_min[:, None, 1:] + (1-coords) * planes.l1_proj_max[:, None, 1:] # [P, L, 2]
    coords = torch.cat([torch.ones_like(coords[:, :, 1:]), coords], dim=-1) # [P, L, 3]

    P = torch.stack([planes.bxyz[:, 1:], planes.eigvecs[:, :, 1], planes.eigvecs[:, :, 2]], dim=1) # [P, 3, 3]
    sampled_xyz = coords @ P
    sampled_batch_idx = planes.bxyz[:, None, 0:1].repeat(1, cfg.num_points_per_plane, 1) # [P, L, 1]
    sampled_bxyz = torch.cat([sampled_batch_idx, sampled_xyz], dim=-1).reshape(-1, 4)
    sampled_feat = (coords @ planes.feat).view(-1, planes.feat.shape[-1]) # [P, L, 3] @ [P, 3, D] = [P, L, D]

    sampled_plane_id = torch.arange(num_planes).repeat(cfg.num_points_per_plane, 1).T.reshape(-1).to(plane_id)
    sampled_weight = sampled_xyz.new_zeros(num_planes, cfg.num_points_per_plane)
    sampled_weight[plane_degree > 0.5] = (1.0 / plane_degree[plane_degree > 0.5, None] / cfg.num_points_per_plane)
    sampled_weight = sampled_weight.reshape(-1)
    sampled_points = EasyDict(dict(
                        coords=coords.reshape(-1, 3),
                        feat=sampled_feat,
                        bxyz=sampled_bxyz,
                        plane_id=sampled_plane_id,
                        weight=sampled_weight,
                        ))
    sampled_points = EasyDict(common_utils.filter_dict(sampled_points, sampled_points.weight > 1e-6))
    sp_points = EasyDict(common_utils.filter_dict(points, ~mask, ignore_keys=['name']))
    sp_points.weight = points.feat.new_ones(sp_points.bxyz.shape[0])
    sp_points.pop('name')
    ret_points = EasyDict(common_utils.torch_concat_dicts([sampled_points, sp_points]))
    #import polyscope as ps; ps.init(); ps.set_up_dir('z_up')
    #import ipdb; ipdb.set_trace()
    #ps.register_point_cloud('points', ret_points.bxyz[:, 1:].detach().cpu().numpy(), radius=0.0025)
    #ps.register_point_cloud('sp_points', sp_points.bxyz[:, 1:].detach().cpu().numpy(), radius=0.0025)
    #ps.register_point_cloud('plane_points', sampled_points.bxyz[:, 1:].detach().cpu().numpy(), radius=0.0025)
    #ps.show()

    return ret_points

class PointPlaneNetFlatBlock(DownBlockTemplate):
    def __init__(self, block_cfg, graph_cfg, *args):
        super().__init__(block_cfg, None, graph_cfg, *args) # no sampler
        self.pos_channel = 3
        self.in_channel = in_channel = block_cfg["INPUT_CHANNEL"]
        mlp_channels = block_cfg["MLP_CHANNELS"]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.key = block_cfg["KEY"]

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.mlp_l0 = nn.Linear(self.pos_channel, mlp_channels[0], bias=False)
        self.norm_l0 = norm_fn(mlp_channels[0])
        if in_channel > 0:
            self.mlp_f0 = nn.Linear(in_channel, mlp_channels[0], bias=False)
            self.norm_f0 = norm_fn(mlp_channels[0])

        last_channel = mlp_channels[0]
        for out_channel in mlp_channels[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(norm_fn(out_channel))
            last_channel = out_channel
        self.num_point_features = last_channel
        num_points_per_plane = block_cfg.get("NUM_POINTS_PER_PLANE", 10)
        self.resample_cfg = EasyDict(dict(
                                     num_points_per_plane=num_points_per_plane,
                                    ))

    def forward(self, ref, runtime_dict):
        """
        Input:
            ref_bxyz [N, 4]: input points, first dimension indicates batch index
            ref_feat [N, C]: per-point feature vectors
        Return:
            query_bxyz: sampled points [M, 4]
            query_feat: per-sampled-point feature vector [M, C_out]
        """
        query = EasyDict(ref.copy())

        if f'{self.key}_graph' in runtime_dict:
            e_ref, e_query, e_weight, sampled_ref = runtime_dict[f'{self.key}_graph']
        else:
            sampled_ref = resample_points(ref, planes, self.resample_cfg)
            assert ref.bxyz.shape[0] > 0
            assert query.bxyz.shape[0] > 0
            e_ref, e_query, e_weight = self.graph(sampled_ref, query)
            runtime_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight, sampled_ref
            runtime_dict[f'{self.key}_ref_bcenter'] = ref.bcenter
            runtime_dict[f'{self.key}_query_bcenter'] = query.bcenter

        # init layer
        pos_diff = (sampled_ref.bxyz[e_ref] - query.bxyz[e_query])[:, 1:4] # [E, 3]
        
        pos_feat = self.norm_l0(self.mlp_l0(pos_diff)) # [E, 3] -> [E, D]
        if self.in_channel > 0:
            ref_feat2 = self.norm_f0(self.mlp_f0(sampled_ref.feat))
            edge_feat = F.relu(pos_feat + ref_feat2[e_ref], inplace=False)
        else:
            edge_feat = F.relu(pos_feat, inplace=False)

        ## mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_feat = conv(edge_feat)
            edge_feat = F.relu(bn(edge_feat), inplace=False)
        query_feat = scatter(edge_feat, e_query, dim=0,
                             dim_size=query_bxyz.shape[0], reduce='mean')
        if query_feat.shape[-1] == ref.feat.shape[-1]:
            query.feat = ref.feat + query.feat

        return query, runtime_dict 


class PointPlaneNetDownBlock(DownBlockTemplate):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg, *args):
        super().__init__(block_cfg, sampler_cfg, graph_cfg, *args)
        self.pos_channel = 3
        self.in_channel = in_channel = block_cfg["INPUT_CHANNEL"]
        mlp_channels = block_cfg["MLP_CHANNELS"]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.key = block_cfg["KEY"]

        self.mlp_l0 = nn.Linear(self.pos_channel, mlp_channels[0], bias=False)
        self.norm_l0 = nn.BatchNorm1d(mlp_channels[0])
        if in_channel > 0:
            self.mlp_f0 = nn.Linear(in_channel, mlp_channels[0], bias=False)
            self.norm_f0 = nn.BatchNorm1d(mlp_channels[0])

        last_channel = mlp_channels[0]
        for out_channel in mlp_channels[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.num_point_features = last_channel
        
        num_points_per_plane = block_cfg.get("NUM_POINTS_PER_PLANE", 10)
        self.resample_cfg = EasyDict(dict(
                                     num_points_per_plane=num_points_per_plane,
                                    ))

    def forward(self, ref, planes, runtime_dict):
        """
        Input:
            ref_bxyz [N, 4]: input points, first dimension indicates batch index
            ref_feat [N, C]: per-point feature vectors
        Return:
            query_bxyz: sampled points [M, 4]
            query_feat: per-sampled-point feature vector [M, C_out]
        """
        
        if self.sampler:
            query = self.sampler(ref)
        else:
            query = EasyDict(ref.copy())
        
        sampled_ref = resample_points(ref, planes, self.resample_cfg)

        if f'{self.key}_graph' in runtime_dict:
            e_ref, e_query, e_weight, sampled_ref = runtime_dict[f'{self.key}_graph']
        else:
            assert ref.bxyz.shape[0] > 0
            assert query.bxyz.shape[0] > 0
            e_ref, e_query, e_weight = self.graph(sampled_ref, query)
            #e_weight = sampled_ref.weight[e_ref]
            runtime_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight, sampled_ref
            #runtime_dict[f'{self.key}_ref_bxyz'] = ref.bxyz
            #runtime_dict[f'{self.key}_query_bxyz'] = query.bxyz

        # init layer
        pos_diff = (sampled_ref.bxyz[e_ref] - query.bxyz[e_query])[:, 1:4] # [E, 3]
        
        pos_feat = self.norm_l0(self.mlp_l0(pos_diff)) # [E, 3] -> [E, D]
        if self.in_channel > 0:
            ref_feat2 = self.norm_f0(self.mlp_f0(sampled_ref.feat))
            edge_feat = F.relu(pos_feat + ref_feat2[e_ref], inplace=False)
        else:
            edge_feat = F.relu(pos_feat, inplace=False)

        ## mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_feat = conv(edge_feat)
            edge_feat = F.relu(bn(edge_feat), inplace=False)
        query.feat = scatter(edge_feat, e_query, dim=0,
                             dim_size=query.bxyz.shape[0], reduce='max')
        mask = (query.plane_id != -1)
        num_planes = planes.bxyz.shape[0]
        plane_degree = scatter(query.feat.new_ones(mask.long().sum()), query.plane_id[mask],
                               dim=0, dim_size=num_planes, reduce='sum')
        mask2 = plane_degree[query.plane_id[mask]] < 3
        query.plane_id[mask][mask2] = -1

        import ipdb; ipdb.set_trace()

        return query, runtime_dict


class PointPlaneNetUpBlock(UpBlockTemplate):
    def __init__(self, block_cfg, **kwargs):
        super().__init__(block_cfg, **kwargs)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        skip_channel = block_cfg.get("SKIP_CHANNEL", None)
        prev_channel = block_cfg["PREV_CHANNEL"]
        mlp_channels = block_cfg["MLP_CHANNELS"]
        self.skip = skip_channel is not None

        self.mlp_f0 = nn.Linear(prev_channel, mlp_channels[0], bias=False)
        self.norm_f0 = nn.BatchNorm1d(mlp_channels[0])
        if skip_channel is not None:
            self.mlp_s0 = nn.Linear(skip_channel, mlp_channels[0], bias=False)
            self.norm_s0 = nn.BatchNorm1d(mlp_channels[0])

        last_channel = mlp_channels[0]
        for out_channel in mlp_channels[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.num_point_features = last_channel

    def forward(self, ref, query, runtime_dict):
        """
        Args:
            ref_bxyz [N, 4]: sampled points, first dimension indicates batch index
            ref_feat [N, C]: per-point feature vectors
            query_bxyz: original points [M, 4]
            query_skip_feat: features from skip connections
            
        Returns:
            query_feat: per-sampled-point feature vector [M, C_out]
        """
        e_ref, e_query, _ = self.graph(ref, query)

        pos_dist = (ref.bxyz[e_ref, 1:4] - query.bxyz[e_query, 1:4]).norm(p=2, dim=-1) # [E]
        pos_dist = 1.0 / (pos_dist + 1e-8)

        weight_sum = scatter(pos_dist, e_query, dim=0,
                             dim_size=query.bxyz.shape[0], reduce='sum')
        weight = pos_dist / weight_sum[e_query] # [E]

        ref_feat2 = self.norm_f0(self.mlp_f0(ref.feat))[e_ref]
        query_feat = scatter(ref_feat2*weight[:, None], e_query, dim=0,
                             dim_size=query.bxyz.shape[0], reduce='sum')

        if self.skip:
            query_skip_feat = self.norm_s0(self.mlp_s0(query.feat))
            query.feat = F.relu(query_feat + query_skip_feat, inplace=False)
        else:
            query.feat = F.relu(query_feat, inplace=False)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            query.feat = F.relu(bn(conv(query.feat)), inplace=False)

        return query

