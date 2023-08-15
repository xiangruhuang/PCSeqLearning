import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch_scatter import scatter

from ...utils.polar_utils import xyz2sphere, xyz2cylind, xyz2sphere_aug
from ...ops.pointops.functions import pointops, pointops_utils

from pcdet.models.model_utils.sampler_utils import SAMPLERS
from pcdet.models.model_utils.grouper_utils import GROUPERS
from pcdet.models.blocks import (
    DOWNBLOCKS,
    UPBLOCKS
)

class PointNet2Down(nn.Module):
    def __init__(self, in_channel, sampler_cfg, grouper_cfg, block_cfg):
        if sampler_cfg is not None:
            sampler = SAMPLERS[sampler_cfg.pop("TYPE")]
            self.sampler = sampler(
                               runtime_cfg=None,
                               model_cfg=sampler_cfg,
                           )
        
        if grouper_cfg is not None:
            grouper = GROUPERS[grouper_cfg.pop("TYPE")]
            self.grouper = grouper(
                               runtime_cfg=None,
                               model_cfg=grouper_cfg,
                           )
