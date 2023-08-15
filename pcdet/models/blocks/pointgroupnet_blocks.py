import torch
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F

from .pointnet2_blocks import (
    PointNet2DownBlock,
    PointNet2UpBlock,
)

class PointGroupNetDownBlock(PointNet2DownBlock):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg, grouper_cfg, fusion_cfg):
        super().__init__(block_cfg, sampler_cfg, graph_cfg, grouper_cfg, fusion_cfg)

    def forward(self, ref_bxyz, ref_feat, **kwargs):
        """
        Input:
            ref_bxyz [N, 4]: input points, first dimension indicates batch index
            ref_feat [N, C]: per-point feature vectors
        Return:
            query_bxyz: sampled points [M, 4]
            query_feat: per-sampled-point feature vector [M, C_out]
        """
        if self.sampler:
            query_bxyz = self.sampler(ref_bxyz)
        else:
            query_bxyz = ref_bxyz

        if self.graph:
            e_ref, e_query = self.graph(ref_bxyz, query_bxyz)

        # init layer
        pos_diff = (ref_bxyz[e_ref] - query_bxyz[e_query])[:, 1:4] # [E, 3]
        
        pos_feat = self.norm_l0(self.mlp_l0(pos_diff)) # [E, 3] -> [E, D]
        ref_feat2 = self.norm_f0(self.mlp_f0(ref_feat))
        edge_feat = F.relu(pos_feat + ref_feat2[e_ref], inplace=False)

        ## mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_feat = conv(edge_feat)
            edge_feat = F.relu(bn(edge_feat), inplace=False)
        query_feat = scatter(edge_feat, e_query, dim=0,
                             dim_size=query_bxyz.shape[0], reduce='max')

        group_ids = self.grouper(query_bxyz)
        num_groups = group_ids.max().item()+1
        fused_query_feat = self.fusion(query_bxyz, query_feat, group_ids)

        return query_bxyz, fused_query_feat, group_ids

class PointGroupNetUpBlock(PointNet2UpBlock):
    def __init__(self, block_cfg):
        super().__init__(block_cfg)

    def forward(self, ref_bxyz, ref_feat,
                query_bxyz, query_skip_feat, **kwargs):
        """
        Args:
            ref_bxyz [N, 4]: sampled points, first dimension indicates batch index
            ref_feat [N, C]: per-point feature vectors
            query_bxyz: original points [M, 4]
            query_skip_feat: features from skip connections
            
        Returns:
            query_feat: per-sampled-point feature vector [M, C_out]
        """
        e_ref, e_query = self.graph(ref_bxyz, query_bxyz)

        pos_dist = (ref_bxyz[e_ref, 1:4] - query_bxyz[e_query, 1:4]).norm(p=2, dim=-1) # [E]
        pos_dist = 1.0 / (pos_dist + 1e-8)

        weight_sum = scatter(pos_dist, e_query, dim=0,
                             dim_size=query_bxyz.shape[0], reduce='sum')
        weight = pos_dist / weight_sum[e_query] # [E]

        ref_feat2 = self.norm_f0(self.mlp_f0(ref_feat))[e_ref]
        query_feat = scatter(ref_feat2*weight[:, None], e_query, dim=0,
                             dim_size=query_bxyz.shape[0], reduce='sum')

        if self.skip:
            query_skip_feat = self.norm_s0(self.mlp_s0(query_skip_feat))
            query_feat = F.relu(query_feat + query_skip_feat, inplace=False)
        else:
            query_feat = F.relu(query_feat, inplace=False)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            query_feat = F.relu(bn(conv(query_feat)), inplace=False)

        return query_feat

