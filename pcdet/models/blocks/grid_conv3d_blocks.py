import torch
import numpy as np
from torch import nn
from easydict import EasyDict

import torch.nn.functional as F
import torch_cluster
from torch_scatter import scatter
from functools import partial

from .block_templates import (
    DownBlockTemplate,
    UpBlockTemplate
)
from .message_passing_v2 import MessagePassingBlock
from .assigners import ASSIGNERS


class GridConvFlatBlock(DownBlockTemplate):
    def __init__(self, block_cfg, graph_cfg, assigner_cfg):
        super().__init__(block_cfg, None, graph_cfg, assigner_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.relu = block_cfg.get("RELU", True)
        self.key = block_cfg['KEY']

        self.message_passing = MessagePassingBlock(input_channel, output_channel, 27, self.key)
        
    def forward(self, ref, conv_dict):
        query = EasyDict(ref.copy())

        if f'{self.key}_graph' in conv_dict:
            e_ref, e_query, e_weight, e_kernel = conv_dict[f'{self.key}_graph']
        else:
            e_ref, e_query, e_weight = self.graph(ref, query)
            e_kernel = self.kernel_assigner(ref, query, e_ref, e_query) # in range [0, 27)
            conv_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight, e_kernel

        query.feat, conv_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcenter.shape[0], conv_dict, e_weight)

        if self.norm:
            query.feat = self.norm(query.feat)

        if self.relu:
            if self.act:
                query.feat = self.act(query.feat)

        return query, conv_dict


class GridConvDownBlock(DownBlockTemplate):
    def __init__(self, block_cfg, sampler_cfg, graph_cfg, assigner_cfg):
        super().__init__(block_cfg, sampler_cfg, graph_cfg, assigner_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']
        
        self.message_passing = MessagePassingBlock(input_channel, output_channel, 27, self.key)
        
    def forward(self, ref, conv_dict):
        if self.sampler is not None:
            query = self.sampler(ref)
        else:
            query = EasyDict(ref.copy())

        if f'{self.key}_graph' in conv_dict:
            e_ref, e_query, e_weight, e_kernel = conv_dict[f'{self.key}_graph']
        else:
            e_ref, e_query, e_weight = self.graph(ref, query)
            e_kernel = self.kernel_assigner(ref, query, e_ref, e_query) # in range [0, 27)
            conv_dict[f'{self.key}_graph'] = e_ref, e_query, e_weight, e_kernel

        query.feat, conv_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcenter.shape[0], conv_dict, e_weight)

        if self.norm:
            query.feat = self.norm(query.feat)

        if self.act:
            query.feat = self.act(query.feat)

        return query, conv_dict


class GridConvUpBlock(UpBlockTemplate):
    def __init__(self, block_cfg, graph_cfg, assigner_cfg):
        super().__init__(block_cfg, graph_cfg, assigner_cfg)
        input_channel = block_cfg["INPUT_CHANNEL"]
        output_channel = block_cfg["OUTPUT_CHANNEL"]
        self.key = block_cfg['KEY']

        self.message_passing = MessagePassingBlock(input_channel, output_channel, 27, self.key)
        
    def forward(self, ref, query, conv_dict):
        assert f'{self.key}_graph' in conv_dict
        e_query, e_ref, e_weight, e_kernel = conv_dict[f'{self.key}_graph']

        query.feat, conv_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcenter.shape[0], conv_dict, e_weight)
        
        if self.norm:
            query.feat = self.norm(query.feat)

        if self.act:
            query.feat = self.act(query.feat)

        return query, conv_dict
