import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils, loss_utils, polar_utils
from torch_scatter import scatter

from .vfe_template import VFETemplate
from pcdet.models.model_utils.basic_blocks import MLP
from pcdet.ops.torch_hash import RadiusGraph
from pcdet.ops.pointops.functions import pointops, pointops_utils
from functools import partial
from ..pointnet2_utils import PointNetSetAbstractionCN2Nor

class MaskEmbeddingVFE(VFETemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg)
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.input_channels = runtime_cfg["num_point_features"]
        self.model_cfg = model_cfg

        self.stride = model_cfg.get("STRIDE", 4)
        self.radius = model_cfg.get("RADIUS", 0.5)
        self.radius_graph = RadiusGraph(ndim=3)
        self.num_sectors = model_cfg.get("NUM_SECTORS", 6)
        self.num_neighbors = model_cfg.get("NUM_NEIGHBORS", 16)
        self.ignore_batch_index = model_cfg.get("IGNORE_BATCH_INDEX", False)
        self.mlp_channels = model_cfg.get("MLP_CHANNELS", None)
        self.mask_ratio = model_cfg.get("MASK_RATIO", None)

        pe_config = model_cfg.get("POS_EMBEDDING", None)
        self.position_embedding = nn.ModuleList()
        self.position_embedding_keys = []
        if pe_config is not None:
            for name, channels in pe_config.items():
                self.position_embedding_keys.append(name)
                pe_module = MLP([3] + channels)
                self.position_embedding.append(pe_module)


        self.sa_layer = PointNetSetAbstractionCN2Nor(
                            self.stride, self.num_neighbors, self.input_channels,
                            self.mlp_channels, return_polar=False,
                            num_sectors=self.num_sectors,
                        )

        self.num_point_features = self.mlp_channels[-1]
        self.forward_dict = {}

    def get_output_feature_dim(self):
        return self.num_point_features
    
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

    def generate_mask(self, num_patches, mask_ratio):
        return torch.rand(num_patches) < mask_ratio

    def forward(self, batch_dict):
        if self.ignore_batch_index:
            # treat all batch as one
            batch_dict['batch_size'] = 1
            batch_dict['point_bxyz'][:, 0] = 0

        num_sweeps = batch_dict['batch_size']
        point_bxyz = batch_dict['point_bxyz']
        point_feat = batch_dict['point_feat']
        batch_index = point_bxyz[:, 0].round().long()

        num_points = []
        for i in range(batch_dict['batch_size']):
            num_points.append((batch_index == i).sum().int())
        num_points = torch.tensor(num_points).int().cuda()
        offset = num_points.cumsum(dim=0).int()
        point_xyz = point_bxyz[:, 1:4].contiguous()

        pos_feat_off = (point_xyz, point_feat, offset)
        
        pos_feat_off_out = self.sa_layer(pos_feat_off)

        #new_xyz, new_feat, new_offset, fps_idx, group_idx = \
        #        pointops_utils.sample_and_group(
        #            self.stride, self.num_neighbors, point_xyz,
        #            point_feat, offset, num_sectors=self.num_sectors,
        #            return_idx=True
        #        )
        #edge_index = torch.stack([fps_idx[:, None].repeat(1, self.num_neighbors),
        #                          group_idx], dim=0).reshape(2, -1) # [2, N*num_neighbors]

        token_bxyz = self.convert_to_bxyz(pos_feat_off_out)

        token_feat = pos_feat_off_out[1]
        mask = self.generate_mask(token_bxyz.shape[0], self.mask_ratio).to(token_bxyz.device)
        batch_dict['mask'] = mask
        masked_bxyz = token_bxyz[mask]
        visible_bxyz = token_bxyz[~mask]
        masked_feat = token_feat[mask]
        visible_feat = token_feat[~mask]
        if self.output_key is not None:
            batch_dict[f'{self.output_key}_visible_bxyz'] = visible_bxyz
            batch_dict[f'{self.output_key}_masked_bxyz'] = masked_bxyz
            batch_dict[f'{self.output_key}_visible_feat'] = visible_feat
            batch_dict[f'{self.output_key}_masked_feat'] = masked_feat

        # position embedding
        for pe_key, pe_module in zip(self.position_embedding_keys, self.position_embedding):
            batch_dict[pe_key] = pe_module(point_xyz)
            
        return batch_dict
