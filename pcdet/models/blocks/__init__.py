from torch import nn
from easydict import EasyDict
from .pointnet2_blocks import (
    PointNet2DownBlock,
    PointNet2UpBlock,
    #PointNet2V2UpBlock,
    PointNet2FlatBlock,
)

from .pointplanenet_blocks import (
    PointPlaneNetDownBlock,
    PointPlaneNetUpBlock,
    #PointPlaneNetV2UpBlock,
    PointPlaneNetFlatBlock,
)
#from .pointconv_blocks import (
#    PointConvDownBlock,
#    PointConvUpBlock,
#    PointConvV2UpBlock,
#    PointConvFlatBlock,
#)
from .grid_conv3d_blocks import (
    GridConvDownBlock,
    GridConvFlatBlock,
    GridConvUpBlock,
)
from .volume_conv3d_blocks import (
    VolumeConvDownBlock,
    VolumeConvFlatBlock,
    VolumeConvUpBlock,
)
from .hybrid_conv3d_blocks import (
    HybridConvDownBlock,
    HybridConvFlatBlock,
    HybridConvUpBlock,
)
from .sst_blocks import (
    BasicShiftBlockV2
)
from .kpconv_blocks import (
    SimpleBlock,
    KPDualBlock,
    FPBlockUp
)
from .pointnet2repsurf_blocks import (
    PointNetSetAbstractionCN2Nor,
    PointNetFeaturePropagationCN2,
    batch_index_to_offset
)

from .pointgroupnet_blocks import (
    PointGroupNetDownBlock,
    PointGroupNetUpBlock,
)

from .basic_blocks import *
from .basic_block_2d import *
from .attention_blocks import *
from .spconv_blocks import *
from .assigners import ASSIGNERS

from .edge_conv import EdgeConv
from .grid_conv import GridConv
from pcdet.utils import common_utils

CONVS = dict(
    EdgeConv=EdgeConv,
    GridConv=GridConv,
)

def build_conv(conv_cfg, cur_channel):
    conv_type = conv_cfg["TYPE"]
    conv_cfg = EasyDict(conv_cfg.copy())
    if conv_type == "EdgeConv":
        if "INPUT_CHANNEL" not in conv_cfg:
            conv_cfg.INPUT_CHANNEL = cur_channel
        edge_conv = EdgeConv(conv_cfg)
        return edge_conv, edge_conv.num_point_features
    elif conv_type == "GridConv":
        if "INPUT_CHANNEL" not in conv_cfg:
            conv_cfg.INPUT_CHANNEL = cur_channel
        if "num_convs" in conv_cfg:
            num_convs = conv_cfg.num_convs
            grid_convs = nn.ModuleList()
            for i in range(num_convs):
                conv_cfg_i = EasyDict(common_utils.indexing_list_elements(conv_cfg, i))
                grid_conv = GridConv(conv_cfg_i.assigner, conv_cfg_i)
                grid_convs.append(grid_conv)
                conv_cfg.INPUT_CHANNEL = conv_cfg_i.OUTPUT_CHANNEL
        else:
            grid_convs = GridConv(conv_cfg.assigner, conv_cfg)
        return grid_convs, conv_cfg.OUTPUT_CHANNEL
    else:
        raise ValueError(f"conv type {conv_type} not recognized")


