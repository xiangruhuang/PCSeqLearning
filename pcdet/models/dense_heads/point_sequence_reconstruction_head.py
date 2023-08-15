import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils, polar_utils
from torch_scatter import scatter

from .reconstruction_head_template import ReconstructionHeadTemplate
from ...models.blocks import MLP
from pcdet.ops.torch_hash import RadiusGraph, ChamferDistance

class PointSequenceReconstructionHead(ReconstructionHeadTemplate):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg)
        input_channels = runtime_cfg["num_point_features"]
        self.max_num_points = runtime_cfg.get("max_num_points", 200000)

        self.model_cfg = model_cfg

        channels = self.model_cfg.get("CHANNELS", None)
        self.latent_dim = channels[-1]
        self.mlp = MLP(channels)
        self.point_bxyz_key = self.input_key + '_bxyz'
        self.point_feature_key = self.input_key + '_feat'

        self.num_predicted_points = model_cfg.get("NUM_PREDICTED_POINTS", 0)
        self.max_num_points *= self.num_predicted_points
        
        self.predictor = nn.Sequential(
                            nn.Linear(channels[-1], self.num_predicted_points*3)
                         )

        self.radius_graph = RadiusGraph(max_num_points=self.max_num_points, ndim=3)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.radius = model_cfg.get("RADIUS", None)
        self.forward_dict = {}

    def build_losses(self, losses_cfg):
        loss_type = losses_cfg.get("LOSS_REC", None)
        if loss_type == 'Chamfer':
            self.loss_func = ChamferDistance(self.max_num_points, ndim=3)
        else:
            raise NotImplementedError(f"Loss Type {loss_type}")

    def compute_gt_neighborhood(self, batch_dict, ignore_batch=False):
        """
        Args:
            ignore_batch: if True, merge all point clouds into one, (used in point cloud sequences)

        Returns:
            edge_indices: [2, E] (idx_of_ref, idx_of_query)
        """
        points = batch_dict[self.point_bxyz_key] # query from them [N, 4]
        point_bxyz = batch_dict['point_bxyz'] # query to them
        if ignore_batch:
            points = points.clone()
            point_bxyz = point_bxyz.clone()
            points[:, 0] = 0
            point_bxyz[:, 0] = 0
        e_ref, e_query = self.radius_graph(point_bxyz, points, self.radius,
                                           self.num_predicted_points, sort_by_dist=False)


        gt_points = torch.cat([e_query[:, None], point_bxyz[e_ref, 1:4]], dim=-1) # [E, 4]
        gt_nbrhood = torch.cat([e_query[:, None], point_bxyz[e_ref, 1:4]-points[e_query, 1:4]], dim=-1) # [E, 4]
        return gt_points, gt_nbrhood

    def get_loss(self, tb_dict=None):
        if tb_dict is None:
            tb_dict = {}
        gt_nbrhood = self.forward_dict['gt_nbrhood']
        pred_nbrhood = self.forward_dict['pred_nbrhood']
        max_radius = pred_nbrhood.norm(p=2, dim=-1).max().item()
        loss = self.loss_func(pred_nbrhood, gt_nbrhood, max_radius*2)

        return loss, tb_dict

    def forward(self, batch_dict):
        """For each point in batch_dict[self.point_feature_key], predict a fixed number of points
        compare it against corresponding ground truth neighborhood points with matching loss (e.g. chamfer)
            
        """
        ref = batch_dict['point_bxyz']
        query = batch_dict[self.point_bxyz_key]
        # [M, 4], the first index represents neighborhood index in range [Q]
        gt_points, gt_nbrhood = self.compute_gt_neighborhood(batch_dict)
        
        point_feat = batch_dict[self.point_feature_key] # query features [Q, D]
        point_feat = self.mlp(point_feat) # processed features [Q, D2]
        pred_nbrhood = self.predictor(point_feat).reshape(-1, self.num_predicted_points, 3) # [Q, num_pred, 3]
        nbrhood_idx = torch.arange(pred_nbrhood.shape[0])[:, None, None].expand(-1, self.num_predicted_points, -1).to(pred_nbrhood)
        pred_nbrhood = torch.cat([nbrhood_idx, pred_nbrhood], dim=-1).view(-1, 4) # [Q, num_pred, 4].view(-1, 4)
        
        self.forward_dict['gt_nbrhood'] = gt_nbrhood

        self.forward_dict['pred_nbrhood'] = pred_nbrhood
        batch_dict['gt_point_bxyz'] = gt_points
        batch_dict.update(self.forward_dict)

        return batch_dict
