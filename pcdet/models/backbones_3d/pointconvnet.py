import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
from .post_processors import build_post_processor
from pcdet.models.blocks import (
    GridConvDownBlock,
    GridConvFlatBlock,
    GridConvUpBlock,
)

class PointConvNet(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointConvNet, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.assigners = model_cfg.get("ASSIGNERS", None)
        self.graphs = model_cfg.get("GRAPHS", None)
        self.sa_channels = model_cfg.get("SA_CHANNELS", None)
        self.fp_channels = model_cfg.get("FP_CHANNELS", None)
        self.num_global_channels = model_cfg.get("NUM_GLOBAL_CHANNELS", 0)
        self.keys = model_cfg.get("KEYS", None)
        self.norm_cfg = model_cfg.get("NORM_CFG", None)
        self.activation = model_cfg.get("ACTIVATION", None)
        
        self.scale = runtime_cfg.get("scale", 1)
        #fp_channels = model_cfg["FP_CHANNELS"]
        self.output_key = model_cfg.get("OUTPUT_KEY", None)

        self.down_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        self.num_down_layers = len(self.sa_channels)
        for i, sa_channels in enumerate(self.sa_channels):
            sampler_cfg = common_utils.indexing_list_elements(self.samplers, i)

            graph_cfg = graph_utils.select_graph(self.graphs, i)
            prev_graph_cfg = graph_utils.select_graph(self.graphs, max(i-1, 0))

            assigner_cfg = common_utils.indexing_list_elements(self.assigners, i)
            prev_assigner_cfg = common_utils.indexing_list_elements(self.assigners, max(i-1, 0))

            keys = self.keys[i]
            sa_channels = [int(self.scale*c) for c in sa_channels]
            
            down_module = nn.ModuleList()
            for j, sc in enumerate(sa_channels):
                block_cfg = dict(
                    INPUT_CHANNEL=cur_channel,
                    OUTPUT_CHANNEL=sc,
                    KEY=keys[j],
                    NORM_CFG=self.norm_cfg,
                    ACTIVATION=self.activation,
                )
                if j == 0:
                    down_module_j = GridConvDownBlock(block_cfg,
                                                      sampler_cfg,
                                                      prev_graph_cfg,
                                                      prev_assigner_cfg)
                else:
                    down_module_j = GridConvFlatBlock(block_cfg,
                                                      graph_cfg,
                                                      assigner_cfg)
                down_module.append(down_module_j)

                cur_channel = sc

            self.down_modules.append(down_module)
            channel_stack.append(cur_channel)
        
        self.up_modules = nn.ModuleList()
        self.skip_modules = nn.ModuleList()
        self.merge_modules = nn.ModuleList()
        self.num_up_layers = len(self.fp_channels)
        for i, fp_channels in enumerate(self.fp_channels):
            graph_cfg = graph_utils.select_graph(self.graphs, -i-1)

            assigner_cfg = common_utils.indexing_list_elements(self.assigners, -i-1)

            fc0, fc1, fc2 = [int(self.scale*c) for c in fp_channels]
            key0, key1, key2 = self.keys[-i-1][:3][::-1]
            skip_channel = channel_stack.pop()
            self.skip_modules.append(
                nn.ModuleList([
                    GridConvFlatBlock(
                        dict(
                            INPUT_CHANNEL=skip_channel,
                            OUTPUT_CHANNEL=fc0,
                            KEY=key0,
                            NORM_CFG=self.norm_cfg,
                            ACTIVATION=self.activation,
                        ),
                        graph_cfg,
                        assigner_cfg,
                    ),
                    GridConvFlatBlock(
                        dict(
                            INPUT_CHANNEL=fc0,
                            OUTPUT_CHANNEL=fc0,
                            KEY=key0,
                            RELU=False,
                            NORM_CFG=self.norm_cfg,
                            ACTIVATION=self.activation,
                        ),
                        graph_cfg,
                        assigner_cfg,
                    )]
                ))
            self.merge_modules.append(
                GridConvFlatBlock(
                    dict(
                        INPUT_CHANNEL=fc0+skip_channel,
                        OUTPUT_CHANNEL=fc1,
                        KEY=key1,
                        NORM_CFG=self.norm_cfg,
                        ACTIVATION=self.activation,
                    ),
                    graph_cfg,
                    assigner_cfg,
                ))
            
            self.up_modules.append(
                GridConvUpBlock(
                    dict(
                        INPUT_CHANNEL=fc1,
                        OUTPUT_CHANNEL=fc2,
                        KEY=key2,
                        NORM_CFG=self.norm_cfg,
                        ACTIVATION=self.activation,
                    ),
                    graph_cfg=None,
                    assigner_cfg=None,
                ))
            
            cur_channel = fc2

        self.num_point_features = cur_channel
        
        runtime_cfg['input_key'] = self.output_key
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def forward(self, batch_dict):
        #point_bxyz = batch_dict[f'{self.input_key}_bcenter']
        #point_feat = batch_dict[f'{self.input_key}_feat']

        voxelwise = EasyDict(dict(
                        name='input',
                        bcoords=batch_dict[f'{self.input_key}_bcoords'],
                        bcenter=batch_dict[f'{self.input_key}_bcenter'],
                        bxyz=batch_dict[f'{self.input_key}_bxyz'],
                        feat=batch_dict[f'{self.input_key}_feat'],
                    ))
        
        data_stack = []
        data_stack.append(voxelwise)

        runtime_dict = {}
        for i in range(self.num_down_layers):
            down_module = self.down_modules[i]
            key = f'pointconvnet_down{len(self.sa_channels)-i}'
            for j, down_module_j in enumerate(down_module):
                voxelwise, runtime_dict = down_module_j(voxelwise, runtime_dict)
            data_stack.append(EasyDict(voxelwise.copy()))
            for attr in voxelwise.keys():
                batch_dict[f'{key}_{attr}'] = voxelwise[attr]

        for key in runtime_dict.keys():
            if key.endswith('_graph'):
                e_ref, e_query, e_weight, e_kernel = runtime_dict[key]
                batch_dict[f'{key}_edges'] = torch.stack([e_ref, e_query], dim=0)
                if e_weight is not None:
                    batch_dict[f'{key}_weight'] = e_weight

        ref = data_stack.pop()

        skip = EasyDict(ref.copy())
        for i in range(self.num_up_layers):
            up_module = self.up_modules[i]
            skip_modules = self.skip_modules[i] if i < len(self.skip_modules) else None
            merge_module = self.merge_modules[i]

            key = f'pointconvnet_up{i+1}'
            if skip_modules:
                # skip transformation and merging
                identity = skip.feat
                for skip_module in skip_modules:
                    skip, runtime_dict = skip_module(skip, runtime_dict)
                skip.feat = F.relu(skip.feat + identity)

            concat = EasyDict(ref.copy())
            concat.feat = torch.cat([ref.feat, skip.feat], dim=-1)
            merge, runtime_dict = merge_module(concat, runtime_dict)
            num_ref_points = ref.bcoords.shape[0]
            ref.feat = merge.feat + concat.feat.view(num_ref_points, -1, 2).sum(dim=2)

            # upsampling
            query = data_stack.pop()
            skip = EasyDict(query.copy())
            ref, runtime_dict = up_module(ref, query, runtime_dict)

            for attr in ref.keys():
                batch_dict[f'{key}_{attr}'] = ref[attr]

        batch_dict.update(runtime_dict)

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = ref.bcenter
            batch_dict[f'{self.output_key}_feat'] = ref.feat

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)

        return batch_dict
