import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils

class ReconstructionHeadTemplate(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_key = runtime_cfg.get("input_key", None)

    def forward(self, **kwargs):
        raise NotImplementedError
