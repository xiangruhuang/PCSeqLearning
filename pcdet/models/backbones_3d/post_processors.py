import torch
import torch.nn as nn
import copy
from torch_scatter import scatter

from pcdet.models.blocks import build_conv_layer, build_norm_layer
from pcdet.models.blocks import (
    PointNetFeaturePropagationCN2,
    batch_index_to_offset
)

class Conv2dPostProcessor(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        self.output_shape = model_cfg.get("OUTPUT_SHAPE", None)
        self.num_attached_conv = num_attached_conv = model_cfg.get("NUM_ATTACHED_CONV", 2)
        self.scale = runtime_cfg.get("scale", 1.0)
        conv_kwargs= model_cfg.get("CONV_KWARGS", dict(kernel_size=3, dilation=2,
                                                       padding=2, stride=1))
        conv_cfg = model_cfg.get("CONV_CFG", dict(type='Conv2d', bias=False))
        norm_cfg = model_cfg.get("NORM_CFG", dict(type='BatchNorm2d', eps=1e-3, momentum=0.01))

        conv_in_channel = model_cfg.get("CONV_IN_CHANNEL", 64)
        conv_out_channel = model_cfg.get("CONV_OUT_CHANNEL", 64)
        conv_in_channel = int(conv_in_channel*self.scale)
        conv_out_channel = int(conv_out_channel*self.scale)

        conv_list = []
        for i in range(num_attached_conv):

            if isinstance(conv_kwargs, dict):
                conv_kwargs_i = conv_kwargs
            elif isinstance(conv_kwargs, list):
                assert len(conv_kwargs) == num_attached_conv
                conv_kwargs_i = conv_kwargs[i]

            if i > 0:
                conv_in_channel = conv_out_channel
            conv = build_conv_layer(
                conv_in_channel,
                conv_out_channel,
                conv_cfg,
                **conv_kwargs_i,
                )

            if norm_cfg is None:
                convnormrelu = nn.Sequential(
                    conv,
                    nn.ReLU(inplace=True)
                )
            else:
                convnormrelu = nn.Sequential(
                    conv,
                    build_norm_layer(conv_out_channel, norm_cfg),
                    nn.ReLU(inplace=True)
                )
            conv_list.append(convnormrelu)
        
        self.conv_layer = nn.ModuleList(conv_list)
        self.num_point_features = conv_out_channel
    
    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            boundary_mask = (this_coors[:, 1] < nx) & (this_coors[:, 2] < ny)
            this_coors = this_coors[boundary_mask]
            xy_indices = this_coors[:, 1] * ny + this_coors[:, 2]
            xy_indices = xy_indices.type(torch.long)

            voxels = voxel_feat[batch_mask, :][boundary_mask] #[n, c]

            canvas = scatter(voxels, xy_indices, reduce='mean', dim_size=ny*nx, dim=0)
            canvas = canvas.view(nx, ny, -1)

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, dim=0).transpose(1, 3)

        return batch_canvas

    def forward(self, batch_dict):
        output = batch_dict['sst_voxel_out_feat']
        output = self.recover_bev(output, voxel_info['voxel_coords'], batch_size)

        if self.num_attached_conv > 0:
            for conv in self.conv_layer:
                temp = conv(output)
                if temp.shape == output.shape and self.conv_shortcut:
                    output = temp + output
                else:
                    output = temp

        output_list = []
        output_list.append(output)


class PointNet2PostProcessor(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        fp_channel = model_cfg.get("FP_CHANNEL", None)
        self.input_key = runtime_cfg.get("input_key", None)
        self.query_key = model_cfg.get("QUERY_KEY", None)
        self.scale = runtime_cfg.get("scale", 1.0)

        cur_channel = input_channels
        fp_channel = [int(c*self.scale) for c in fp_channel]
        self.fp_module = PointNetFeaturePropagationCN2(cur_channel, None, fp_channel)

        self.num_point_features = fp_channel[-1]
    
    def convert_to_bxyz(self, pos_feat_off):
        xyz = pos_feat_off[0]
        assert xyz.shape[-1] == 3, f"expecting xyz to have shape [..., 3], got {xyz.shape}"
        offset = pos_feat_off[2]
        batch_idx = []
        last_offset = 0
        for i, offset_i in enumerate(offset):
            batch_idx.append(torch.full([offset_i-last_offset, 1], i).long().to(xyz.device))
            last_offset = offset_i
        batch_idx = torch.cat(batch_idx, dim=0)
        bxyz = torch.cat([batch_idx, xyz], dim=-1)

        return bxyz

    def forward(self, batch_dict):
        point_bxyz = batch_dict[f'{self.input_key}_bxyz']
        point_feat = batch_dict[f'{self.input_key}_feat']
        point_batch_index = point_bxyz[:, 0].round().long()
        point_offset = batch_index_to_offset(point_batch_index)

        query_bxyz = batch_dict[f'{self.query_key}_bxyz']
        query_batch_index = query_bxyz[:, 0].round().long()
        query_offset = batch_index_to_offset(query_batch_index)

        pos_feat_off = [point_bxyz[:, 1:4].contiguous(), point_feat, point_offset]
        pos_feat_off_skip = [query_bxyz[:, 1:4].contiguous(), None, query_offset]

        pos_feat_off_skip[1] = self.fp_module(pos_feat_off_skip, pos_feat_off)
        batch_dict[f'{self.query_key}_post_feat'] = pos_feat_off_skip[1]

        return batch_dict

POSTPROCESSORS = {
    'Conv2d': Conv2dPostProcessor,
    'PointNet2': PointNet2PostProcessor
}

def build_post_processor(post_processor_cfg, runtime_cfg):
    post_processor = post_processor_cfg.get("TYPE", None)
    if post_processor is None:
        return None 
    else:
        return POSTPROCESSORS[post_processor](
                   model_cfg=post_processor_cfg,
                   runtime_cfg=runtime_cfg
               )

