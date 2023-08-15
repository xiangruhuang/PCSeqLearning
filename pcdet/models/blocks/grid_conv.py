import torch
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F
from functools import partial
import torch_cluster

from .message_passing_v2 import MessagePassingBlock
from .assigners import ASSIGNERS

def build_norm(block_cfg):
    norm_cfg = block_cfg.get("NORM_CFG", None)
    if norm_cfg is not None:
        if 'OUTPUT_CHANNEL' in block_cfg:
            output_channel = block_cfg["OUTPUT_CHANNEL"]
        elif 'MLP_CHANNELS' in block_cfg:
            output_channel = block_cfg["MLP_CHANNELS"][-1]
        norm = nn.BatchNorm1d(output_channel, **norm_cfg)
    else:
        norm = None
    return norm

def build_act(block_cfg):
    act_cfg = block_cfg.get("ACTIVATION", None)
    if act_cfg is not None:
        if act_cfg == 'ReLU':
            act = nn.ReLU()
        else:
            raise ValueError("Unrecognized Activation {act_cfg}")
    else:
        act = None
    return act


class GridConv(nn.Module):
    def __init__(self, assigner, conv_cfg):
        super().__init__()
        input_channel = conv_cfg["INPUT_CHANNEL"]
        output_channel = conv_cfg["OUTPUT_CHANNEL"]
        self.input_channel = output_channel
        self.output_channel = output_channel
        self.key = conv_cfg['KEY']
        self.assigner = assigner

        self.norm = build_norm(conv_cfg)
        self.act = build_act(conv_cfg)
        
        self.message_passing = MessagePassingBlock(input_channel, output_channel, 27, self.key)
        
    def forward(self, ref, query, graph, conv_dict):
        e_ref, e_query, e_weight = graph
        e_kernel = self.assigner(ref, query, e_ref, e_query)
        assert e_kernel.shape[0] == e_ref.shape[0]

        query_feat, conv_dict = self.message_passing(
                                    ref.feat, e_kernel, e_ref, e_query,
                                    query.bcenter.shape[0], conv_dict, e_weight)

        if self.norm:
            query_feat = self.norm(query_feat)

        if self.act:
            query_feat = self.act(query_feat)

        return query_feat, conv_dict
