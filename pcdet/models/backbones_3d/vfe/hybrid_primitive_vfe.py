import numpy as np
import torch
from torch import nn
from torch_scatter import scatter
from collections import defaultdict
from easydict import EasyDict

#from pcdet.models.model_utils.grid_sampling import GridSampling3D
from .vfe_template import VFETemplate
from ....ops.torch_hash import RadiusGraph
from ...blocks import MLP
from pcdet.models.model_utils.partition_utils import PARTITIONERS
from pcdet.models.model_utils.primitive_utils import pca_fitting
from pcdet.utils import common_utils


class HybridPrimitiveVFE(VFETemplate):
    def __init__(self, model_cfg, runtime_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = runtime_cfg["num_point_features"]
        self.grid_size = model_cfg.get("GRID_SIZE", None)
        partition_cfg = model_cfg.get("PARTITION_CFG", None)
        self.partitioner = PARTITIONERS[partition_cfg['TYPE']](
                               runtime_cfg=None,
                               model_cfg=partition_cfg,
                           )
        self.pca_cfg = model_cfg.get("PCA_CFG", None)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """

        ref = EasyDict(dict(
                  bxyz=batch_dict['point_bxyz'],
                  feat=batch_dict['point_feat'],
              ))
        runtime_dict = {}

        ref = self.partitioner(ref, runtime_dict)
        pointwise, planewise = pca_fitting(ref, ref.partition_id, self.pca_cfg)
        pointwise = common_utils.transform_name(pointwise, lambda name: 'point_'+name)
        planewise = common_utils.transform_name(planewise, lambda name: 'plane_'+name)
        batch_dict.update(planewise)
        batch_dict.update(pointwise)
        
        return batch_dict
