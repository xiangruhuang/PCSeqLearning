import torch
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F
from functools import partial

from .block_templates import (
    DownBlockTemplate,
    UpBlockTemplate,
)


class PointNet2FlatBlock(DownBlockTemplate):
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
            e_ref, e_query, e_weight = runtime_dict[f'{self.key}_graph']
        else:
            #assert ref.bxyz.shape[0] > 0
            #assert query.bxyz.shape[0] > 0
            e_ref, e_query, e_weight = self.graph(ref, query)
            runtime_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight
            #runtime_dict[f'{self.key}_ref_bcenter'] = ref.bcenter
            #runtime_dict[f'{self.key}_query_bcenter'] = query.bcenter

        # init layer
        pos_diff = (ref.bxyz[e_ref] - query.bxyz[e_query])[:, 1:4] # [E, 3]
        
        pos_feat = self.norm_l0(self.mlp_l0(pos_diff)) # [E, 3] -> [E, D]
        if self.in_channel > 0:
            ref_feat2 = self.norm_f0(self.mlp_f0(ref.feat))
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


class PointNet2DownBlock(DownBlockTemplate):
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

    def forward(self, ref, runtime_dict):
        """
        Input:
            ref_bxyz [N, 4]: input points, first dimension indicates batch index
            ref_feat [N, C]: per-point feature vectors
        Return:
            query_bxyz: sampled points [M, 4]
            query_feat: per-sampled-point feature vector [M, C_out]
        """

        if self.sampler:
            query = self.sampler(ref, runtime_dict)
        else:
            query = EasyDict(ref.copy())

        if f'{self.key}_graph' in runtime_dict:
            e_ref, e_query, e_weight = runtime_dict[f'{self.key}_graph']
        else:
            #assert ref.bxyz.shape[0] > 0
            #assert query.bxyz.shape[0] > 0
            e_ref, e_query, e_weight = self.graph(ref, query)
            runtime_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight
            #runtime_dict[f'{self.key}_ref_bxyz'] = ref.bxyz
            #runtime_dict[f'{self.key}_query_bxyz'] = query.bxyz

        # init layer
        pos_diff = (ref.bxyz[e_ref] - query.bxyz[e_query])[:, 1:4] # [E, 3]
        
        pos_feat = self.norm_l0(self.mlp_l0(pos_diff)) # [E, 3] -> [E, D]
        if self.in_channel > 0:
            ref_feat2 = self.norm_f0(self.mlp_f0(ref.feat))
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

        return query, runtime_dict


class PointNet2UpBlock(UpBlockTemplate):
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


#class PointNet2V2UpBlock(UpBlockTemplate):
#    def __init__(self, block_cfg, **kwargs):
#        super().__init__(block_cfg, **kwargs)
#        self.mlp_convs = nn.ModuleList()
#        self.mlp_bns = nn.ModuleList()
#        skip_channel = block_cfg.get("skip_channel", None)
#        prev_channel = block_cfg["prev_channel"]
#        mlp_channels = block_cfg["mlp_channels"]
#        self.skip = skip_channel is not None
#
#        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#        self.mlp_l0 = nn.Linear(3, mlp_channels[0], bias=False)
#        self.norm_l0 = norm_fn(mlp_channels[0])
#        self.mlp_f0 = nn.Linear(prev_channel, mlp_channels[0], bias=False)
#        self.norm_f0 = norm_fn(mlp_channels[0])
#        if skip_channel is not None:
#            self.mlp_s0 = nn.Linear(skip_channel, mlp_channels[0], bias=False)
#            self.norm_s0 = norm_fn(mlp_channels[0])
#
#        last_channel = mlp_channels[0]
#        for out_channel in mlp_channels[1:]:
#            self.mlp_convs.append(nn.Linear(last_channel, out_channel, bias=False))
#            self.mlp_bns.append(norm_fn(out_channel))
#            last_channel = out_channel
#        self.num_point_features = last_channel
#
#    def forward(self, ref_bxyz, ref_feat,
#                query_bxyz, query_skip_feat):
#        """
#        Args:
#            ref_bxyz [N, 4]: sampled points, first dimension indicates batch index
#            ref_feat [N, C]: per-point feature vectors
#            query_bxyz: original points [M, 4]
#            query_skip_feat: features from skip connections
#            
#        Returns:
#            query_feat: per-sampled-point feature vector [M, C_out]
#        """
#        e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
#        
#        pos_diff = (ref_bxyz[e_ref] - query_bxyz[e_query])[:, 1:4] # [E, 3]
#        
#        pos_feat = self.norm_l0(self.mlp_l0(pos_diff)) # [E, 3] -> [E, D]
#
#        pos_dist = pos_diff.norm(p=2, dim=-1) # [E]
#        pos_dist = 1.0 / (pos_dist + 1e-8)
#
#        weight_sum = scatter(pos_dist, e_query, dim=0,
#                             dim_size=query_bxyz.shape[0], reduce='sum')
#        weight = pos_dist / weight_sum[e_query] # [E]
#
#        ref_feat2 = self.norm_f0(self.mlp_f0(ref_feat))[e_ref]
#        ref_feat2 = F.relu(ref_feat2 + pos_feat, inplace=False)
#
#        # mlp
#        for i, conv in enumerate(self.mlp_convs):
#            bn = self.mlp_bns[i]
#            ref_feat2 = F.relu(bn(conv(ref_feat2)), inplace=False)
#
#        query_feat = scatter(ref_feat2*weight[:, None], e_query, dim=0,
#                             dim_size=query_bxyz.shape[0], reduce='sum')
#
#        return query_feat, e_ref, e_query

