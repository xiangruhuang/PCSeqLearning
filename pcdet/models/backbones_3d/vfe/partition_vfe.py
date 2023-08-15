import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_scatter import scatter

from .vfe_template import VFETemplate
from pcdet.models.blocks import MLPBlock
from pcdet.ops.voxel import VoxelAggregation

class NNPartitionAggregation(object):
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, point_wise_mean_dict, ):
        

class NNPartitionVFE(VFETemplate):
    """Partition the point set and extract feature for each partition

    """
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg)
        self.model_cfg = model_cfg
        self.runtime_cfg = runtime_cfg

        self.point_feature_cfg = self.model_cfg.get("POINT_FEATURE_CFG", [])
        
        num_point_features = runtime_cfg.get("num_point_features", None)
        num_point_features += 3
        for key, size in self.point_feature_cfg.items():
            num_point_features += size
        self.scale = runtime_cfg.get("scale", 1.0)

        self.mlp_channels = self.model_cfg.get("MLP_CHANNELS", None)
        self.mlp_channels = [int(self.scale*c) for c in self.mlp_channels]
        assert len(self.mlp_channels) > 0
        mlp_channels = [num_point_features] + list(self.mlp_channels)

        self.norm_cfg = self.model_cfg.get("NORM_CFG", None)

        self.vfe_layers = nn.ModuleList()
        for i in range(len(mlp_channels) - 1):
            in_channel = mlp_channels[i]

            if i > 0:
                in_channel *= 2
            out_channel = mlp_channels[i + 1]
            self.vfe_layers.append(
                MLPBlock(in_channel, out_channel, self.norm_cfg, activation=nn.ReLU(), bias=False)
            )

        self.partition_graph_cfg = model_cfg.get("PARTITION_GRAPH_CFG", None)
        self.runtime_cfg.update(self.partition_graph_cfg)
        self.partition_graph = PartitionAggregation(model_cfg=self.partition_graph_cfg,
                                                runtime_cfg=self.runtime_cfg)

        self.num_point_features = self.mlp_channels[-1]
        runtime_cfg['input_channels'] = self.mlp_channels[-1]
        self.output_key = 'voxel'

    def get_output_feature_dim(self):
        return self.num_point_features

    def process_point_features(self, partition_wise_dict, batch_dict, partition_index, out_of_boundary_mask):
        """
        Args:
            partition_wise_dict: attributes that has shape [V, ?]
            batch_dict: input data
            partition_index [N] : the partition index of each point
        Returns:
            point_features [N, C_out]
        """
        point_xyz = batch_dict['point_bxyz'][~out_of_boundary_mask, 1:4].contiguous()
        point_feat = batch_dict['point_feat'][~out_of_boundary_mask]
        
        feature_list = [point_xyz, point_feat]
        if 'offset_to_partition_xyz' in self.point_feature_cfg:
            feature_list.append(point_xyz-partition_wise_dict['partition_xyz'][partition_index])
        if 'offset_to_partition_center' in self.point_feature_cfg:
            feature_list.append(point_xyz-partition_wise_dict['partition_center'][partition_index])

        return torch.cat(feature_list, dim=-1)

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            point_bxyz [N, 4] input point coordinates
            point_feat [N, C] input point features
        Returns:
            voxel_features [V, C] output feature per voxel
            voxel_coords [V, 4] integer coordinate of each voxel
        """
        point_bxyz = batch_dict['point_bxyz'] # (batch_idx, x, y, z)
        point_feat = batch_dict['point_feat'] # (i, e)
        point_wise_mean_dict=dict(
            point_bxyz=point_bxyz,
            point_feat=point_feat,
        )

        voxel_wise_dict, point_wise_dict, num_voxels, out_of_boundary_mask = \
                self.partition_graph(point_wise_mean_dict,
                                     point_wise_median_dict=dict(
                                         segmentation_label = batch_dict['segmentation_label']
                                     ))
        voxel_index = point_wise_dict['voxel_index']

        if self.use_volume:
            point_xyz = point_bxyz[:, 1:]
            voxel_volume = scatter(point_xyz.new_ones(point_xyz.shape[0]), voxel_index,
                                   dim=0, dim_size=num_voxels, reduce='sum')
            assert (voxel_volume > 0.5).all()
            voxel_xyz = scatter(point_xyz, voxel_index, dim=0,
                                dim_size=num_voxels, reduce='mean')
            point_d = point_xyz - voxel_xyz[voxel_index]
            point_ddT = point_d.unsqueeze(-1) * point_d.unsqueeze(-2)
            voxel_ddT = scatter(point_ddT, voxel_index, dim=0,
                                dim_size=num_voxels, reduce='mean')

            #voxel_eigvals, voxel_eigvecs = np.linalg.eigh(voxel_ddT.detach().cpu().numpy())
            #voxel_eigvals = torch.from_numpy(voxel_eigvals).to(voxel_ddT)
            #voxel_eigvecs = torch.from_numpy(voxel_eigvecs).to(voxel_ddT)
            voxel_eigvals, voxel_eigvecs = torch.linalg.eigh(voxel_ddT) # eigvals in ascending order
            voxel_wise_dict['voxel_eigvals'] = voxel_eigvals
            voxel_wise_dict['voxel_eigvecs'] = voxel_eigvecs
                                 
        point_features = self.process_point_features(voxel_wise_dict, batch_dict,
                                                     voxel_index, out_of_boundary_mask)

        for i, vfe_layer in enumerate(self.vfe_layers):
            point_features = vfe_layer(point_features)
            voxel_features = scatter(point_features, voxel_index, dim_size=num_voxels, dim=0, reduce='mean')
            if i != len(self.vfe_layers) - 1:
                point_features = torch.cat([point_features, voxel_features[voxel_index]], dim=-1)

        voxel_wise_dict['voxel_feat'] = voxel_features
        batch_dict.update(voxel_wise_dict)
        batch_dict['point_bcoords'] = point_wise_dict['point_bcoords']

        return batch_dict
