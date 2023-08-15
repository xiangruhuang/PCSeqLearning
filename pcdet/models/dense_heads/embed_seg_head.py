import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from ...utils import box_utils, loss_utils
from .point_head_template import PointHeadTemplate
from ...ops.torch_hash import RadiusGraph
from pcdet.models.model_utils import graph_utils
from torch_scatter import scatter
from torch_cluster import knn


class EmbedSegHead(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        num_intrinsic_dims = model_cfg.get("NUM_INTRINSIC_DIMS", 128)
        input_channels = runtime_cfg['input_channels']
        num_classes = runtime_cfg['num_seg_classes']
        super().__init__(model_cfg=model_cfg,
                         num_class=num_classes)
        self.ignore_index = model_cfg.get("IGNORE_INDEX", None)
        self.scale = runtime_cfg.get('scale', 1.0)
        self.assign_to_point = model_cfg.get("ASSIGN_TO_POINT", False)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=[int(c*self.scale) for c in self.model_cfg.CLS_FC],
            input_channels=input_channels,
            output_channels=num_intrinsic_dims,
            dropout=self.dropout
        )
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.target_assigner_cfg = self.model_cfg.get("TARGET_ASSIGNER", None)
        if self.target_assigner_cfg is not None:
            max_num_points = self.target_assigner_cfg.get("MAX_NUM_POINTS", None)
            self.radius_graph = RadiusGraph(max_num_points=max_num_points, ndim=3)
        self.graph = graph_utils.KNNGraph({}, dict(NUM_NEIGHBORS=1))
        self.corres_graph = partial(knn, k=1)
    
    def build_losses(self, losses_cfg):
        if not isinstance(losses_cfg['LOSS'], list):
            losses_cfg['LOSS'] = [losses_cfg['LOSS']]
        if not isinstance(losses_cfg['WEIGHT'], list):
            losses_cfg['WEIGHT'] = [losses_cfg['WEIGHT']]
        self.loss_names = losses_cfg['LOSS']
        self.losses = nn.ModuleList()
        self.loss_weight = []
        for loss, weight in zip(losses_cfg['LOSS'], losses_cfg['WEIGHT']):
            self.losses.append(
                loss_utils.LOSSES[loss](loss_cfg=losses_cfg)
            )
            self.loss_weight.append(weight)

        #if losses_cfg['LOSS'] == 'cross-entropy-with-logits':
        #    self.cls_loss_func = loss_utils.CrossEntropyWithLogits() 
        #elif losses_cfg['LOSS'] == 'focal-loss':
        #    alpha = losses_cfg.get("ALPHA", 0.5)
        #    gamma = losses_cfg.get("GAMMA", 2.0)
        #    self.add_module(
        #        'cls_loss_func',
        #        loss_utils.FocalLoss(alpha = alpha, gamma = gamma, reduction='mean')
        #    )
        #elif losses_cfg['LOSS'] == 'ohem':
        #    self.add_module(
        #        'cls_loss_func',
        #        loss_utils.OHEMLoss(ignore_index=0, thresh=0.7, min_kept=0.001)
        #    )
        #self.loss_weight = losses_cfg.get('WEIGHT', 1.0)
    
    def get_cls_layer_loss(self, tb_dict=None, prefix=None):
        gt_corres = self.forward_ret_dict[self.gt_seg_cls_label_key].view(-1).long()
        corres = self.forward_ret_dict['correspondence'].view(-1).long()

        template_xyz = self.forward_ret_dict['template_xyz']
        gt_embedding = self.forward_ret_dict['template_embedding'][gt_corres]
        pred_embedding = self.forward_ret_dict['pred_embedding']

        gap = (template_xyz[gt_corres] - template_xyz[corres]).norm(p=2, dim=-1)

        reg_loss = self.losses[0](pred_embedding, gt_embedding, gap) * self.loss_weight[0]
        if tb_dict is None:
            tb_dict = {}
        
        if prefix is not None:
            for thresh in [2, 5, 10]:
                error_rate = (gap > thresh / 100.0).float().mean()
                tb_dict.update({f'{prefix}/error_rate_{thresh}cm': error_rate.item()})
            tb_dict.update({f'{prefix}/average_geodesic': gap.mean().item()})
        else:
            for thresh in [2, 5, 10]:
                error_rate = (gap > thresh / 100.0).float().mean()
                tb_dict.update({f'error_rate_{thresh}cm': error_rate.item()})
            tb_dict.update({f'average_geodesic': gap.mean().item()})

        #point_loss_cls = 0.0
        #for loss_module, loss_name, loss_weight in \
        #        zip(self.losses, self.loss_names, self.loss_weight):
        #    loss_this = loss_module(point_cls_preds, point_cls_labels)*loss_weight
        #    if prefix is None:
        #        tb_dict[loss_name] = loss_this.item()
        #    else:
        #        tb_dict[f'{prefix}/{loss_name}'] = loss_this.item()
        #    point_loss_cls += loss_this

        return reg_loss, tb_dict
    
    def get_loss(self, tb_dict=None, prefix=None):
        tb_dict = {} if tb_dict is None else tb_dict
        loss, tb_dict_1 = self.get_cls_layer_loss(prefix=prefix)

        point_loss = loss
        tb_dict.update(tb_dict_1)
        if prefix is None:
            tb_dict.update({f'loss': point_loss.item()})
        else:
            tb_dict.update({f'{prefix}/loss': point_loss.item()})

        return point_loss, tb_dict

    def get_iou_statistics(self):
        pred_dicts = self.get_evaluation_results()
        iou_dicts = []
        iou_dict = dict(
            ups=None,
            downs=None,
        )
        for pred_dict in pred_dicts:
            iou_dicts.append(dict())

        return iou_dicts, iou_dict

    def get_evaluation_results(self):
        corres = self.forward_ret_dict['correspondence']
        gt_corres = self.forward_ret_dict[self.gt_seg_cls_label_key]

        batch_idx = self.forward_ret_dict['batch_idx']
        pred_dicts = []
        point_bxyz = self.forward_ret_dict['point_bxyz']
        
        for i in range(self.forward_ret_dict['batch_size']):
            bs_mask = batch_idx == i
            point_xyz = point_bxyz[bs_mask, 1:4]

            gt_corres = self.forward_ret_dict[self.gt_seg_cls_label_key][bs_mask]
            corres_this = corres[bs_mask]

            record_dict = dict(
                point_wise=dict(
                    gt_corres=gt_corres,
                    corres=corres_this,
                    point_xyz=point_xyz,
                ),
                object_wise=dict(),
                scene_wise=dict(
                    num_seg_class=self.num_class,
                ),
            )
            pred_dicts.append(record_dict)
        return pred_dicts

    def assign_targets(self, target_assigner_cfg, batch_dict):
        ref_label = batch_dict[target_assigner_cfg["REF_SEGMENTATION_LABEL"]]
        ref_bxyz = batch_dict[target_assigner_cfg["REF_POINT_BXYZ"]]
        query_bxyz = batch_dict[target_assigner_cfg["QUERY_POINT_BXYZ"]]
        query_label_key = target_assigner_cfg["QUERY_SEGMENTATION_LABEL"]

        radius = target_assigner_cfg["RADIUS"]
        er, eq = self.radius_graph(ref_bxyz, query_bxyz, radius, 1, sort_by_dist=True)

        query_label = ref_label.new_full(query_bxyz.shape[:1], 0) # by default, assuming class 0 is ignored
        query_label[eq] = ref_label[er]
        
        batch_dict[query_label_key] = query_label

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict[self.point_feature_key]
        pred_embedding = self.cls_layers(point_features).sigmoid()  # (total_points, num_intrinsic_dims)
        
        if self.target_assigner_cfg is not None:
            self.assign_targets(self.target_assigner_cfg, batch_dict)

        if self.target_assigner_cfg is not None:
            self.assign_targets(self.target_assigner_cfg, batch_dict)

        template_xyz = batch_dict['template_xyz']
        template_embedding = batch_dict['template_embedding']

        # compute point to template map
        _, correspondence = self.corres_graph(template_embedding, pred_embedding)
        gt_corres = batch_dict[self.gt_seg_cls_label_key].view(-1).long()
        if self.ignore_index is not None:
            mask = gt_corres != self.ignore_index
            gt_corres = gt_corres[mask]
            correspondence = correspondence[mask]
            pred_embedding = pred_embedding[mask]
        
        ret_dict = {
            'pred_embedding': pred_embedding,
            'correspondence': correspondence,
            'corres_error': (template_xyz[correspondence] - template_xyz[gt_corres]).norm(p=2, dim=-1),
            'embedding_error': (pred_embedding-template_embedding[gt_corres]).norm(p=2, dim=-1),
        }

        batch_dict.update(ret_dict)
        
        if self.gt_seg_cls_label_key in batch_dict:
            ret_dict[self.gt_seg_cls_label_key] = gt_corres

        ret_dict['batch_idx'] = batch_dict[self.batch_key][:, 0].round().long()
        ret_dict['point_bxyz'] = batch_dict[self.batch_key]
        if self.assign_to_point and (not self.training):
            # assign pred_seg_cls_labels to points
            ref_bxyz = batch_dict[self.batch_key]
            ref_labels = ret_dict['correspondence']
            query_bxyz = batch_dict['point_bxyz']
            
            e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
            new_ret_dict = {}
            for key in ret_dict.keys():
                new_ret_dict[key] = scatter(ret_dict[key][e_ref], e_query, dim=0,
                                            dim_size=query_bxyz.shape[0], reduce='max')
            new_ret_dict['point_bxyz'] = batch_dict['point_bxyz']
            ret_dict = new_ret_dict
        
        ret_dict['template_embedding'] = template_embedding
        ret_dict['template_xyz'] = template_xyz
        
        ret_dict['batch_size'] = batch_dict['batch_size']
        self.forward_ret_dict = ret_dict

        return batch_dict
