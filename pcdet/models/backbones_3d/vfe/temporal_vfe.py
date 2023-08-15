import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils, loss_utils, polar_utils
from torch_scatter import scatter

from .vfe_template import VFETemplate
from pcdet.models.blocks import MLP
from pcdet.ops.torch_hash import RadiusGraph


class TemporalVFE(VFETemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg)
        self.input_channels = runtime_cfg["num_point_features"]

        self.model_cfg = model_cfg

        self.radius_graph = RadiusGraph(ndim=3)
        self.radius_graph.qmin[0] = 1
        self.radius_graph.qmax[0] = 1
        self.radius = (1+.5**2)**0.5

        self.forward_dict = {}

    def get_output_feature_dim(self):
        return self.input_channels

    def forward(self, batch_dict):
        num_sweeps = batch_dict['batch_size']
        lidar_bxyz = batch_dict['point_bxyz']
        lidar_batch_idx = lidar_bxyz[:, 0].round().long()
        cur_indices = torch.where(lidar_batch_idx == 0)[0]
        cur_bxyz = lidar_bxyz[lidar_batch_idx == 0]
        edge_indices = []
        for i in range(1, num_sweeps):
            next_bxyz = lidar_bxyz[lidar_batch_idx == i]
            next_indices = torch.where(lidar_batch_idx == i)[0]
            e_next, e_cur = self.radius_graph(next_bxyz, cur_bxyz, self.radius, 1, sort_by_dist=True)
            e_cur, e_next = cur_indices[e_cur], next_indices[e_next]
            edge_index = torch.stack([e_cur, e_next], dim=-1) # [E, 2]
            edge_indices.append(edge_index)

            del cur_bxyz
            del cur_indices
            cur_bxyz = next_bxyz
            cur_indices = next_indices

        
        edge_indices = torch.cat(edge_indices, dim=0) # [E, 2]
        
        batch_dict['sequence_edges'] = edge_indices.T
        point_xyz = lidar_bxyz.clone()
        point_xyz[:, 0] = 0
        batch_dict['point_xyz'] = point_xyz
            
        return batch_dict
