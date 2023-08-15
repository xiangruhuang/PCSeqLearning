from torch import nn
import torch
from pcdet.models.blocks.kpconv_blocks import SimpleBlock, KPDualBlock, FPBlockUp
from pcdet.ops.torch_hash.torch_hash_modules import RadiusGraph

class KPConv(nn.Module):
    def __init__(self,
                 model_cfg,
                 input_channels,
                 grid_size):
        super().__init__()
        self.model_cfg = model_cfg
        down_conv_cfg = model_cfg["down_conv"]
        max_num_points = model_cfg.get("MAX_NUM_POINTS", 200000)
        max_num_neighbors = model_cfg.get("MAX_NUM_NEIGHBORS", 38)
        self.input_key = model_cfg["INPUT"]
        self.output_key = model_cfg["OUTPUT"]
        self.neighbor_finder = RadiusGraph(
                                   max_num_points=max_num_points
                               )

        up_conv_cfg = model_cfg["up_conv"]
        self.build_down_conv(down_conv_cfg)
        self.build_up_conv(up_conv_cfg, down_conv_cfg)

        self.num_point_features = 64
        self.backbone_channels = {}

    def build_down_conv(self, cfg):
        max_num_neighbors = cfg["max_num_neighbors"]
        channels = cfg["channels"]
        grid_size = cfg["grid_size"]
        grid_size_ratio = cfg["grid_size_ratio"]
        prev_grid_size_ratio = cfg["prev_grid_size_ratio"]
        block_names = cfg["block_names"]
        has_bottleneck = cfg["has_bottleneck"]
        bn_momentum = cfg["bn_momentum"]
        num_kernel_points = cfg["num_kernel_points"]
        num_act_kernel_points = cfg["num_act_kernel_points"]
        num_down_modules = len(channels)
        down_modules = nn.ModuleList()
        for i in range(num_down_modules):
            grid_size_i = [gi * grid_size for gi in grid_size_ratio[i]]
            prev_grid_size_i = [gi * grid_size for gi in prev_grid_size_ratio[i]]
            block = KPDualBlock(
                        block_names[i],
                        channels[i],
                        grid_size_i,
                        prev_grid_size_i,
                        has_bottleneck[i],
                        max_num_neighbors[i],
                        num_kernel_points=num_kernel_points,
                        num_act_kernel_points=num_act_kernel_points,
                        neighbor_finder=self.neighbor_finder,
                        bn_momentum=bn_momentum[i]
                    )
            down_modules.append(block)
        self.down_modules = down_modules

    def build_up_conv(self, cfg, down_cfg):
        channels = cfg["channels"]
        up_k = cfg["up_k"]
        bn_momentum = cfg["bn_momentum"]
        grid_size = down_cfg["grid_size"]
        grid_size_ratio = down_cfg["grid_size_ratio"]
        num_up_modules = len(channels)
        up_modules = nn.ModuleList()
        for i in range(num_up_modules):
            block = FPBlockUp(
                        channels[i],
                        neighbor_finder=self.neighbor_finder,
                        up_k=up_k[i],
                        grid_size=grid_size_ratio[-1-i][-1]*grid_size,
                        bn_momentum=bn_momentum[i],
                    )
            up_modules.append(block)
        self.up_modules = up_modules

    def forward(self, batch_dict):
        points = batch_dict[self.input_key][:, :4].contiguous()
        point_features = batch_dict[self.input_key][:, 1:].contiguous()
        data_dict = dict(
            pos = points,
            x = point_features,
            vis_dict=dict(
                pos=[points],
            )
        )
        stack_down = []
        for i in range(len(self.down_modules)):
            data_dict = self.down_modules[i](data_dict)
            if i < len(self.down_modules) - 1:
                stack_down.append(data_dict)

        for i in range(len(self.up_modules)):
            data_dict = self.up_modules[i](data_dict, stack_down.pop())
        vis_dict = data_dict['vis_dict']
        for i, pos in enumerate(vis_dict['pos']):
            batch_dict[f'pos{i}'] = pos
        batch_dict[self.output_key] = data_dict['x']

        return batch_dict
