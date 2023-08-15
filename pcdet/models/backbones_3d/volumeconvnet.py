import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
from .post_processors import build_post_processor
from pcdet.models.blocks import (
    VolumeConvDownBlock,
    VolumeConvFlatBlock,
    VolumeConvUpBlock,
)


class VolumeConvNet(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(VolumeConvNet, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.use_void_kernels = model_cfg.get("USE_VOID_KERNELS", False)
        self.use_volume_weight = model_cfg.get("USE_VOLUME_WEIGHT", False)
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.assigners = model_cfg.get("ASSIGNERS", None)
        self.graphs = model_cfg.get("GRAPHS", None)
        self.volumes = model_cfg.get("VOLUMES", None)
        self.sa_channels = model_cfg.get("SA_CHANNELS", None)
        self.fp_channels = model_cfg.get("FP_CHANNELS", None)
        self.num_global_channels = model_cfg.get("NUM_GLOBAL_CHANNELS", 0)
        self.keys = model_cfg.get("KEYS", None)
        self.norm_cfg = model_cfg.get("NORM_CFG", None)
        self.activation = model_cfg.get("ACTIVATION", None)
        
        self.scale = runtime_cfg.get("scale", 1)
        self.output_key = model_cfg.get("OUTPUT_KEY", None)

        self.down_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i, sa_channels in enumerate(self.sa_channels):
            sampler_cfg = common_utils.indexing_list_elements(self.samplers, i)
            volume_cfg = common_utils.indexing_list_elements(self.volumes, i)
            graph_cfg = graph_utils.select_graph(self.graphs, i)
            prev_graph_cfg = graph_utils.select_graph(self.graphs, max(i-1, 0))
            prev_volume_cfg = common_utils.indexing_list_elements(self.volumes, max(i-1, 0))
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
                    USE_VOID_KERNELS=self.use_void_kernels,
                    NORM_CFG=self.norm_cfg,
                    ACTIVATION=self.activation,
                )
                if j == 0:
                    down_module_j = VolumeConvDownBlock(block_cfg,
                                                        sampler_cfg,
                                                        prev_graph_cfg,
                                                        prev_assigner_cfg,
                                                        prev_volume_cfg)
                else:
                    down_module_j = VolumeConvFlatBlock(block_cfg,
                                                        graph_cfg,
                                                        assigner_cfg,
                                                        volume_cfg)
                down_module.append(down_module_j)

                cur_channel = sc

            self.down_modules.append(down_module)
            channel_stack.append(cur_channel)
        
        self.up_modules = nn.ModuleList()
        self.skip_modules = nn.ModuleList()
        self.merge_modules = nn.ModuleList()
        for i, fp_channels in enumerate(self.fp_channels):
            graph_cfg = graph_utils.select_graph(self.graphs, -i-1)
            
            assigner_cfg = common_utils.indexing_list_elements(self.assigners, -i-1)
            
            volume_cfg = common_utils.indexing_list_elements(self.volumes, -i-1)

            fp_channels = [int(self.scale*c) for c in fp_channels]
            fc0, fc1, fc2 = fp_channels[0], fp_channels[1], fp_channels[-1]
            key0, key1, key2 = self.keys[-i-1][:3][::-1]
            skip_channel = channel_stack.pop()
            self.skip_modules.append(
                nn.ModuleList([
                    VolumeConvFlatBlock(
                        dict(
                            INPUT_CHANNEL=skip_channel,
                            OUTPUT_CHANNEL=fc0,
                            KEY=key0,
                            USE_VOID_KERNELS=self.use_void_kernels,
                            NORM_CFG=self.norm_cfg,
                            ACTIVATION=self.activation,
                        ),
                        graph_cfg,
                        assigner_cfg,
                        volume_cfg,
                    ),
                    *[VolumeConvFlatBlock(
                        dict(
                            INPUT_CHANNEL=fc0,
                            OUTPUT_CHANNEL=fc0,
                            KEY=key0,
                            RELU=False,
                            USE_VOID_KERNELS=self.use_void_kernels,
                            NORM_CFG=self.norm_cfg,
                            ACTIVATION=self.activation,
                        ),
                        graph_cfg,
                        assigner_cfg,
                        volume_cfg,
                    ) for _ in range(len(fp_channels)-2)]
                    ]
                ))
            self.merge_modules.append(
                VolumeConvFlatBlock(
                    dict(
                        INPUT_CHANNEL=fc0*2,
                        OUTPUT_CHANNEL=fc1,
                        KEY=key1,
                        USE_VOID_KERNELS=self.use_void_kernels,
                        NORM_CFG=self.norm_cfg,
                        ACTIVATION=self.activation,
                    ),
                    graph_cfg,
                    assigner_cfg,
                    volume_cfg,
                ))
            
            self.up_modules.append(
                VolumeConvUpBlock(
                    dict(
                        INPUT_CHANNEL=fc1,
                        OUTPUT_CHANNEL=fc2,
                        KEY=key2,
                        USE_VOID_KERNELS=self.use_void_kernels,
                        NORM_CFG=self.norm_cfg,
                        ACTIVATION=self.activation,
                    ),
                    assigner_cfg=None,
                    graph_cfg=None,
                ))
            
            cur_channel = fc2

        self.num_point_features = cur_channel
        
        runtime_cfg['input_key'] = self.output_key
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def forward(self, batch_dict):
        base_bxyz = batch_dict['point_bxyz']

        voxelwise = EasyDict(dict(
                        name='input',
                        bcoords=batch_dict[f'{self.input_key}_bcoords'],
                        bcenter=batch_dict[f'{self.input_key}_bcenter'],
                        bxyz=batch_dict[f'{self.input_key}_bxyz'],
                        feat=batch_dict[f'{self.input_key}_feat'],
                        #eigvals=batch_dict[f'{self.input_key}_eigvals'],
                        #eigvecs=batch_dict[f'{self.input_key}_eigvecs'],
                    ))

        data_stack = []
        data_stack.append(voxelwise)
        
        runtime_dict = {}
        runtime_dict['base_bxyz'] = base_bxyz
        for i, down_module in enumerate(self.down_modules):
            key = f'volumeconvnet_down{len(self.sa_channels)-i}'
            for j, down_module_j in enumerate(down_module):
                voxelwise, runtime_dict = down_module_j(voxelwise, runtime_dict)

            data_stack.append(EasyDict(voxelwise.copy()))
            for k, v in voxelwise.items():
                batch_dict[f'{key}_{k}'] = v

        for key in runtime_dict.keys():
            if key.endswith('_graph'):
                e_ref, e_query, e_weight, e_kernel = runtime_dict[key]
                batch_dict[f'{key}_edges'] = torch.stack([e_ref, e_query], dim=0)
                if e_weight is not None:
                    batch_dict[f'{key}_weight'] = e_weight

        ref = data_stack.pop()

        skip = EasyDict(ref.copy())
        for i, (up_module, skip_modules, merge_module) in enumerate(zip(self.up_modules, self.skip_modules, self.merge_modules)):
            key = f'volumeconvnet_up{i+1}'

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

            for k, v in ref.items():
                batch_dict[f'{key}_{k}'] = v

        batch_dict.update(runtime_dict)

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = ref.bxyz
            batch_dict[f'{self.output_key}_feat'] = ref.feat

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)

        return batch_dict
