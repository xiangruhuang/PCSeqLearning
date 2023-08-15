import torch
import torch.nn as nn
from easydict import EasyDict

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
from .post_processors import build_post_processor
from pcdet.models.blocks import PointNet2DownBlock, PointNet2UpBlock, SelfAttentionBlock, PointNet2FlatBlock


class PointNet2(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointNet2, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.graphs = model_cfg.get("GRAPHS", None)
        self.sa_channels = model_cfg.get("SA_CHANNELS", None)
        self.fp_channels = model_cfg.get("FP_CHANNELS", None)
        self.keys = model_cfg.get("KEYS", None)
        self.num_global_channels = model_cfg.get("NUM_GLOBAL_CHANNELS", 0)
        self.attributes = model_cfg.get("ATTRIBUTES", None)
        
        self.scale = runtime_cfg.get("scale", 1)
        #fp_channels = model_cfg["FP_CHANNELS"]
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.down_modules = nn.ModuleList()
        self.down_flat_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i, sa_channel in enumerate(self.sa_channels):
            sampler_cfg = common_utils.indexing_list_elements(self.samplers, i)
            graph_cfg = graph_utils.select_graph(self.graphs, i)
            sc = int(self.scale*sa_channel)
            block_cfg = dict(
                INPUT_CHANNEL=cur_channel,
                MLP_CHANNELS=[sc, sc, sc],
                KEY=self.keys[i],
            )
            down_module = PointNet2DownBlock(block_cfg,
                                             sampler_cfg,
                                             graph_cfg)
            self.down_modules.append(down_module)
            channel_stack.append(cur_channel)
            cur_channel = sc
        
        self.global_modules = nn.ModuleList()
        for i in range(self.num_global_channels):
            block_cfg = dict(
                OUTPUT_CHANNEL=cur_channel,
                INPUT_CHANNEL=cur_channel,
                num_heads=8,
            )
            global_module = SelfAttentionBlock(block_cfg)
            self.global_modules.append(global_module)

        self.up_modules = nn.ModuleList()
        for i, fp_channel in enumerate(self.fp_channels):
            fc = int(self.scale*fp_channel)
            skip_channel = channel_stack.pop()
            if (i < len(self.fp_channels) - 1) and (i > 0):
                up_channels = [fc, fc, fc // 2]
            else:
                up_channels = [fc, fc, fc]
            block_cfg = dict(
                SKIP_CHANNEL=None,
                PREV_CHANNEL=cur_channel,
                MLP_CHANNELS=up_channels,
            )
            graph_cfg = graph_utils.select_graph(self.graphs, -i-1)
            up_module = PointNet2UpBlock(block_cfg, graph_cfg=graph_cfg)

            self.up_modules.append(up_module)
            cur_channel = up_channels[-1]

        self.num_point_features = cur_channel
        
        runtime_cfg['input_key'] = self.output_key
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def forward(self, batch_dict):
        pointwise = EasyDict(dict(
                        name='input',
                    ))
        for attr in self.attributes:
            attr_val = batch_dict[f'{self.input_key}_{attr}']
            pointwise[attr] = attr_val

        data_stack = []
        data_stack.append(pointwise)

        runtime_dict = EasyDict(dict())
        
        for i, down_module in enumerate(self.down_modules):
            key = f'pointnet2_down{len(self.sa_channels)-i}'
            #batch_dict[f'{key}_ref'] = point_bxyz
            pointwise, runtime_dict = down_module(pointwise, runtime_dict)
            
            data_stack.append(EasyDict(pointwise.copy()))
            for k, v in pointwise.items():
                batch_dict[f'{key}_{k}'] = v
        
        for key in runtime_dict.keys():
            if key.endswith('_graph'):
                e_ref, e_query, e_weight = runtime_dict[key]
                batch_dict[f'{key}_edges'] = torch.stack([e_ref, e_query], dim=0)
                if e_weight is not None:
                    batch_dict[f'{key}_weight'] = e_weight
        
        for key in runtime_dict.keys():
            if key.endswith('_graph'):
                e_ref, e_query, e_weight = runtime_dict[key]
                batch_dict[f'{key}_edges'] = torch.stack([e_ref, e_query], dim=0)
                if e_weight is not None:
                    batch_dict[f'{key}_weight'] = e_weight

        ref = data_stack.pop()

        for i, up_module in enumerate(self.up_modules):
            key = f'pointnet2_up{i+1}'
            # upsampling
            #point_bxyz_query, point_skip_feat_query = data_stack.pop()
            query = data_stack.pop()
            ref = up_module(ref, query, runtime_dict)
            
            for k, v in ref.items():
                batch_dict[f'{key}_{k}'] = v
            
            #point_feat_query, up_ref, up_query = up_module(point_bxyz_ref, point_feat_ref,
            #                                               point_bxyz_query, point_skip_feat_query)
            #point_bxyz_ref, point_feat_ref = point_bxyz_query, point_feat_query

            #batch_dict[f'{key}_bxyz'] = point_bxyz_ref
            #batch_dict[f'{key}_feat'] = point_feat_ref
            #batch_dict[f'{key}_up_edges'] = torch.stack([up_query, up_ref], dim=0)

        batch_dict.update(runtime_dict)

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = ref.bxyz
            batch_dict[f'{self.output_key}_feat'] = ref.feat

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)

        return batch_dict
