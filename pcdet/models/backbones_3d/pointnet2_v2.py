import torch
import torch.nn as nn

from pcdet.utils import common_utils
from pcdet.models.model_utils import graph_utils
from .post_processors import build_post_processor
from pcdet.models.blocks import PointNet2DownBlock, PointNet2UpBlock, SelfAttentionBlock, PointNet2FlatBlock


class PointNet2V2(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointNet2V2, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        if model_cfg.get("INPUT_KEY", None) is not None:
            self.input_key = model_cfg.get("INPUT_KEY", None)
        else:
            self.input_key = runtime_cfg.get("input_key", 'point')
        
        self.samplers = model_cfg.get("SAMPLERS", None)
        self.graphs = model_cfg.get("GRAPHS", None)
        self.sa_channels = model_cfg.get("SA_CHANNELS", None)
        self.fp_channels = model_cfg.get("FP_CHANNELS", None)
        self.num_global_channels = model_cfg.get("NUM_GLOBAL_CHANNELS", 0)
        
        self.scale = runtime_cfg.get("scale", 1)
        #fp_channels = model_cfg["FP_CHANNELS"]
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.down_modules = nn.ModuleList()
        self.down_flat_modules = nn.ModuleList()

        cur_channel = input_channels
        channel_stack = []
        for i, sa_channel in enumerate(self.sa_channels):
            sampler_cfg = common_utils.indexing_list_elements(self.samplers, i)
            graph_cfg = graph_utils.select_graph(self.graphs, i*2)
            graph_flat_cfg = graph_utils.select_graph(self.graphs, i*2+1)
            sc = int(self.scale*sa_channel)
            block_cfg = dict(
                in_channel=cur_channel,
                mlp_channels=[sc, sc, sc],
            )
            down_module = PointNet2DownBlock(block_cfg,
                                             sampler_cfg,
                                             graph_cfg)
            self.down_modules.append(down_module)
            block_cfg = dict(
                in_channel=sc,
                mlp_channels=[sc, sc, sc],
            )
            flat_module = PointNet2FlatBlock(block_cfg,
                                             graph_flat_cfg)
            self.down_flat_modules.append(flat_module)
            channel_stack.append(cur_channel)
            cur_channel = sc
        
        self.global_modules = nn.ModuleList()
        for i in range(self.num_global_channels):
            block_cfg = dict(
                in_channel=cur_channel,
                out_channel=cur_channel,
                num_heads=8,
            )
            global_module = SelfAttentionBlock(block_cfg)
            self.global_modules.append(global_module)

        self.up_modules = nn.ModuleList()
        self.skip_modules = nn.ModuleList()
        self.merge_modules = nn.ModuleList()
        for i, fp_channel in enumerate(self.fp_channels):
            fc = int(self.scale*fp_channel)
            skip_channel = channel_stack.pop()
            if (i < len(self.fp_channels) - 1):
                up_channels = [fc, fc, fc // 2]
            else:
                up_channels = [fc, fc, fc]
            block_cfg = dict(
                skip_channel=None,
                prev_channel=cur_channel,
                mlp_channels=up_channels,
            )
            graph_cfg = graph_utils.select_graph(self.graphs, -i*2-2)
            up_module = PointNet2UpBlock(block_cfg, graph_cfg=graph_cfg)
            graph_cfg = graph_utils.select_graph(self.graphs, -i*2-1)
            self.skip_modules.append(
                PointNet2FlatBlock(
                    dict(
                        in_channel=fc,
                        mlp_channels=[fc, fc, fc],
                    ),
                    graph_cfg,
                ))
            
            self.merge_modules.append(
                PointNet2FlatBlock(
                    dict(
                        in_channel=fc*2,
                        mlp_channels=[fc, fc, fc],
                    ),
                    graph_cfg,
                ))

            self.up_modules.append(up_module)
            cur_channel = up_channels[-1]

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
        
        for i, (down_module, down_flat_module) in enumerate(zip(self.down_modules, self.down_flat_modules)):
            key = f'pointnet2_down{len(self.sa_channels)-i}_out'
            batch_dict[f'{key}_ref'] = point_bxyz
            point_bxyz, point_feat, down_ref, down_query = down_module(point_bxyz, point_feat)
            #print(f'down {i}, {torch.cuda.max_memory_allocated()/2**30:.6f} GB')
            point_bxyz, point_feat, down_flat_ref, down_flat_query = down_flat_module(point_bxyz, point_feat)
            #print(f'down flat {i}, {torch.cuda.max_memory_allocated()/2**30:.6f} GB')
            #print(f'down {i}, num_nodes = {point_bxyz.shape[0]}, num_feat={point_feat.shape[-1]}')
            batch_dict[f'{key}_query'] = point_bxyz
            data_stack.append([point_bxyz, point_feat])
            batch_dict[f'{key}_bxyz'] = point_bxyz
            batch_dict[f'{key}_feat'] = point_feat
            batch_dict[f'{key}_edges'] = torch.stack([down_query, down_ref], dim=0)
            batch_dict[f'{key}_flat_edges'] = torch.stack([down_flat_query, down_flat_ref], dim=0)

        point_bxyz_ref, point_feat_ref = data_stack.pop()
        for i, global_module in enumerate(self.global_modules):
            point_feat_ref = global_module(point_bxyz_ref, point_feat_ref)

        point_skip_feat_ref = point_feat_ref
        for i, (up_module, skip_module, merge_module) in enumerate(zip(self.up_modules, self.skip_modules, self.merge_modules)):
            key = f'pointnet2_up{i+1}_out'
            # skip transformation and merging
            _, point_skip_feat_ref, skip_ref, skip_query = skip_module(point_bxyz_ref, point_skip_feat_ref)
            batch_dict[f'{key}_ref'] = point_bxyz_ref
            #print(f'skip {i}, {torch.cuda.max_memory_allocated()/2**30:.6f} GB')
            point_concat_feat_ref = torch.cat([point_feat_ref, point_skip_feat_ref], dim=-1)
            _, point_merge_feat_ref, merge_ref, merge_query = merge_module(point_bxyz_ref, point_concat_feat_ref)
            #print(f'merge {i}, {torch.cuda.max_memory_allocated()/2**30:.6f} GB')
            num_ref_points = point_bxyz_ref.shape[0]
            point_feat_ref = point_merge_feat_ref \
                             + point_concat_feat_ref.view(num_ref_points, -1, 2).sum(dim=2)

            # upsampling
            point_bxyz_query, point_skip_feat_query = data_stack.pop()
            point_feat_query, up_ref, up_query = up_module(point_bxyz_ref, point_feat_ref,
                                                           point_bxyz_query, None)
            #print(f'up {i}, {torch.cuda.max_memory_allocated()/2**30:.6f} GB')
            point_bxyz_ref, point_feat_ref = point_bxyz_query, point_feat_query
            point_skip_feat_ref = point_skip_feat_query

            batch_dict[f'{key}_bxyz'] = point_bxyz_ref
            batch_dict[f'{key}_feat'] = point_feat_ref
            batch_dict[f'{key}_skip_edges'] = torch.stack([skip_query, skip_ref], dim=0)
            batch_dict[f'{key}_merge_edges'] = torch.stack([merge_query, merge_ref], dim=0)
            batch_dict[f'{key}_up_edges'] = torch.stack([up_query, up_ref], dim=0)

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = point_bxyz_ref
            batch_dict[f'{self.output_key}_feat'] = point_feat_ref

        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)

        return batch_dict
