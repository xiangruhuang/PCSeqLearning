import torch
import torch.nn as nn

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
from pcdet.models.model_utils import graph_utils
from pcdet.models.model_utils import graph_utils
from pcdet.models.model_utils import graph_utils
from .post_processors import build_post_processor
from pcdet.models.blocks import PointGroupNetDownBlock, PointGroupNetUpBlock


class PointGroupNet(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointGroupNet, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.graphs = model_cfg.get("GRAPHS", None)
        self.groupers = model_cfg.get("GROUPERS", None)
        self.fusions = model_cfg.get("FUSIONS", None)

        self.sa_channels = model_cfg.get("SA_CHANNELS", None)
        self.fp_channels = model_cfg.get("FP_CHANNELS", None)
        
        self.scale = runtime_cfg.get("scale", 1)
        #fp_channels = model_cfg["FP_CHANNELS"]
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.down_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i, sa_channels in enumerate(self.sa_channels):
            sampler_cfg = common_utils.indexing_list_elements(self.samplers, i)
            graph_cfg = graph_utils.select_graph(self.graphs, i)
            grouper_cfg = common_utils.indexing_list_elements(self.groupers, i)
            fusion_cfg = common_utils.indexing_list_elements(self.fusions, i)

            sa_channels = [int(self.scale*c) for c in sa_channels]
            block_cfg = dict(
                in_channel=cur_channel,
                mlp_channels=sa_channels,
            )
            down_module = PointGroupNetDownBlock(block_cfg,
                                                 sampler_cfg,
                                                 graph_cfg,
                                                 grouper_cfg,
                                                 fusion_cfg)
            self.down_modules.append(down_module)
            channel_stack.append(cur_channel)
            cur_channel = sa_channels[-1]

        self.up_modules = nn.ModuleList()
        for i, fp_channels in enumerate(self.fp_channels):
            fp_channels = [int(self.scale*c) for c in fp_channels]
            block_cfg = dict(
                skip_channel=channel_stack.pop(),
                prev_channel=cur_channel,
                mlp_channels=fp_channels,
            )
            up_module = PointGroupNetUpBlock(block_cfg)
            self.up_modules.append(up_module)
            cur_channel = fp_channels[-1] 

        self.num_point_features = cur_channel
        
        runtime_cfg['input_key'] = self.output_key
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def forward(self, batch_dict):
        point_bxyz = batch_dict[f'{self.input_key}_bxyz']
        point_feat = batch_dict[f'{self.input_key}_feat']

        data_stack = []
        data_stack.append([point_bxyz, point_feat])
        
        for i, down_module in enumerate(self.down_modules):
            key = f'pointgroupnet_down{len(self.sa_channels)-i}_out'

            batch_dict[f'{key}_ref'] = point_bxyz
            point_bxyz, point_feat, group_ids = down_module(point_bxyz, point_feat)
            batch_dict[f'{key}_query'] = point_bxyz
            data_stack.append([point_bxyz, point_feat])
            batch_dict[f'{key}_bxyz'] = point_bxyz
            batch_dict[f'{key}_feat'] = point_feat
            batch_dict[f'{key}_group_id'] = group_ids
            #print(f'DownBlock({i}): memory={torch.cuda.memory_allocated()/2**30}')
            #print(f'DownBlock({i}): max_memory={torch.cuda.max_memory_allocated()/2**30}')

        point_bxyz_ref, point_feat_ref = data_stack.pop()
        for i, up_module in enumerate(self.up_modules):
            point_bxyz_query, point_feat_query = data_stack.pop()
            point_feat_query = up_module(point_bxyz_ref, point_feat_ref,
                                         point_bxyz_query, point_feat_query)
            point_bxyz_ref, point_feat_ref = point_bxyz_query, point_feat_query
            key = f'pointgroupnet_up{i+1}_out'
            batch_dict[f'{key}_bxyz'] = point_bxyz_ref
            batch_dict[f'{key}_feat'] = point_feat_ref
            #print(f'UpBlock({i}): memory={torch.cuda.memory_allocated()/2**30}')
            #print(f'UpBlock({i}): max_memory={torch.cuda.max_memory_allocated()/2**30}')

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = point_bxyz_ref
            batch_dict[f'{self.output_key}_feat'] = point_feat_ref

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)
            #print(f'PostProcessor: memory={torch.cuda.memory_allocated()/2**30}')

        return batch_dict
