import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch_scatter import scatter

from pcdet.models.blocks import BLOCKS

class GraphConvDown(nn.Module):
    def __init__(self, in_channel, 
                 sampler_cfg, grouper_cfg, block_cfg):
        super(GraphConvDown, self).__init__()
        block = BLOCKS[block_cfg.get("TYPE")]
        self.conv = block(in_channel,
                          block_cfg['DOWN_CHANNEL'],
                          block_cfg)

        if sampler_cfg is not None:
            sampler = SAMPLERS[sampler_cfg.pop("TYPE")]
            self.sampler = sampler(
                               runtime_cfg=None,
                               model_cfg=sampler_cfg,
                           )
        
        if grouper_cfg is not None:
            grouper = GROUPERS[grouper_cfg.pop("TYPE")]
            self.grouper = grouper(
                               runtime_cfg=None,
                               model_cfg=grouper_cfg,
                           )

    def forward(self, point_bxyz, point_feat):
        """
        Input:
            point_bxyz [N, 4]: input points, first dimension indicates batch index
            point_feat [N, C]: per-point feature vectors
        Return:
            new_bxyz: sampled points [M, 4]
            new_feat: per-sampled-point feature vector [M, C_out]
        """
        if self.sampler:
            new_bxyz = self.sampler(point_bxyz)
        else:
            new_bxyz = point_bxyz
        if self.grouper:
            e_point, e_new = self.grouper(point_bxyz, new_bxyz)
        
        new_feat = self.conv(point_bxyz, point_feat, new_bxyz, e_point, e_new)

        return new_bxyz, new_feat


class GraphConvUp(nn.Module):
    def __init__(self, input_channel, skip_channel, block_cfg):
        super(GraphConvUp, self).__init__()

        block = BLOCKS[block_cfg.get("TYPE")]
        self.conv = block(in_channel,
                          skip_channel,
                          block_cfg)

        self.grouper = GROUPERS['KNNGrouper'](
                           runtime_cfg=None,
                           model_cfg=dict(
                               num_neighbors=3,
                           )
                       )

    def forward(self, ref_bxyz, ref_feat, query_bxyz, query_skip_feat):
        """
        Input:

        Return:
            query_feat [M, C_out]
        """

        e_ref, e_query = self.grouper(ref_bxyz, query_bxyz)
        query_feat = self.conv(ref_bxyz, ref_feat, query_bxyz,
                               query_skip_feat, e_ref, e_query)

        return query_feat
