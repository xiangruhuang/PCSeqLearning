import torch
from torch import nn
import torch.nn.functional as F

from ...utils import box_utils, loss_utils
from .point_head_template import PointHeadTemplate
from ...ops.torch_hash import RadiusGraph
from pcdet.models.model_utils import graph_utils
from torch_scatter import scatter


class PointSegHead(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        num_class = runtime_cfg['num_seg_classes']
        input_channels = runtime_cfg['input_channels']
        super().__init__(model_cfg=model_cfg,
                         num_class=num_class)
        self.scale = runtime_cfg.get('scale', 1.0)
        self.assign_to_point = model_cfg.get("ASSIGN_TO_POINT", False)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=[int(c*self.scale) for c in self.model_cfg.CLS_FC],
            input_channels=input_channels,
            output_channels=num_class,
            dropout=self.dropout
        )
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.target_assigner_cfg = self.model_cfg.get("TARGET_ASSIGNER", None)
        if self.target_assigner_cfg is not None:
            max_num_points = self.target_assigner_cfg.get("MAX_NUM_POINTS", None)
            self.radius_graph = RadiusGraph(max_num_points=max_num_points, ndim=3)
        if self.assign_to_point:
            self.graph = graph_utils.KNNGraph({}, dict(NUM_NEIGHBORS=1))
    
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
        point_cls_labels = self.forward_ret_dict[self.gt_seg_cls_label_key].view(-1).long()
        point_cls_preds = self.forward_ret_dict['pred_seg_cls_logits'].view(-1, self.num_class)

        if tb_dict is None:
            tb_dict = {}
        
        cls_count = point_cls_preds.new_zeros(self.num_class)
        for i in range(self.num_class):
            cls_count[i] = (point_cls_labels == i).float().sum()
        for i in range(self.num_class):
            tb_dict.update({
                f'per_class/cls_count_{i}': cls_count[i].item(),
            })

        point_loss_cls = 0.0
        for loss_module, loss_name, loss_weight in \
                zip(self.losses, self.loss_names, self.loss_weight):
            loss_this = loss_module(point_cls_preds, point_cls_labels)*loss_weight
            if prefix is None:
                tb_dict[loss_name] = loss_this.item()
            else:
                tb_dict[f'{prefix}/{loss_name}'] = loss_this.item()
            point_loss_cls += loss_this

        return point_loss_cls, tb_dict
    
    def get_loss(self, tb_dict=None, prefix=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss(prefix=prefix)

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        iou_stats, _ = self.get_iou_statistics()
        ups, downs = iou_stats[0]['ups'], iou_stats[0]['downs']
        for iou_stat in iou_stats[1:]:
            ups += iou_stat['ups']
            downs += iou_stat['downs']
        ious = ups / torch.clamp(downs, min=1.0)
        for i in range(self.num_class):
            if downs[i] > 0:
                if prefix is None:
                    tb_dict.update({f'per_class/IoU_{i}': ious[i]})
                else:
                    tb_dict.update({f'{prefix}/per_class/IoU_{i}': ious[i]})
        if prefix is None:
            tb_dict.update({f'IoU_FG': ups[1:5].sum()/torch.clamp(downs[1:5].sum(), min=1.0),
                            f'IoU_BG': ups[5:].sum()/torch.clamp(downs[5:].sum(), min=1.0),
                            })
            tb_dict.update({f'mIoU': ious.mean()})
            tb_dict.update({f'loss': point_loss.item()})
        else:
            tb_dict.update({f'{prefix}/IoU_FG': ups[1:5].sum()/torch.clamp(downs[1:5].sum(), min=1.0),
                            f'{prefix}/IoU_BG': ups[5:].sum()/torch.clamp(downs[5:].sum(), min=1.0),
                            })
            tb_dict.update({f'{prefix}/mIoU': ious.mean()})
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
            pred_labels = pred_dict['point_wise']['pred_segmentation_label']
            gt_labels = pred_dict['point_wise']['gt_segmentation_label']
            ups = pred_labels.new_zeros(self.num_class)
            downs = pred_labels.new_zeros(self.num_class)
            pred_labels[gt_labels == 0] = 0
            for cls in range(self.num_class):
                pred_mask = pred_labels == cls
                gt_mask = gt_labels == cls
                ups[cls] = (pred_mask & gt_mask).sum()
                downs[cls] = (pred_mask | gt_mask).sum()
            
            iou_dict['ups'] = ups if iou_dict['ups'] is None else iou_dict['ups'] + ups
            iou_dict['downs'] = downs if iou_dict['downs'] is None else iou_dict['downs'] + downs
            iou_dicts.append(
                dict(
                    ups = ups,
                    downs = downs
                )
            )
        return iou_dicts, iou_dict

    def get_evaluation_results(self):
        pred_logits = self.forward_ret_dict['pred_seg_cls_logits']
        pred_scores = torch.sigmoid(pred_logits)
        batch_idx = self.forward_ret_dict['batch_idx']
        pred_dicts = []
        point_bxyz = self.forward_ret_dict['point_bxyz']
        
        for i in range(self.forward_ret_dict['batch_size']):
            bs_mask = batch_idx == i
            point_xyz = point_bxyz[bs_mask, 1:4]
            pred_confidences, pred_labels = pred_scores[bs_mask].max(-1)
            gt_labels = self.forward_ret_dict[self.gt_seg_cls_label_key][bs_mask]
            valid_mask = (gt_labels >= 0)
            pred_labels = pred_labels[valid_mask]
            gt_labels = gt_labels[valid_mask]
            record_dict = dict(
                point_wise=dict(
                    gt_segmentation_label=gt_labels,
                    pred_segmentation_label=pred_labels,
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
        point_pred_logits = self.cls_layers(point_features)  # (total_points, num_class)

        if self.target_assigner_cfg is not None:
            self.assign_targets(self.target_assigner_cfg, batch_dict)
        
        ret_dict = {
            'pred_seg_cls_logits': point_pred_logits,
        }

        point_pred_scores = torch.sigmoid(point_pred_logits)
        ret_dict['pred_seg_cls_confidences'], ret_dict['pred_seg_cls_labels'] = point_pred_scores.max(dim=-1)
        
        batch_dict.update(ret_dict)
        
        if self.gt_seg_cls_label_key in batch_dict:
            ret_dict[self.gt_seg_cls_label_key] = batch_dict[self.gt_seg_cls_label_key]

        ret_dict['batch_idx'] = batch_dict[self.batch_key][:, 0].round().long()
        ret_dict['point_bxyz'] = batch_dict[self.batch_key]
        if self.assign_to_point and (not self.training):
            # assign pred_seg_cls_labels to points
            ref_bxyz = batch_dict[self.batch_key]
            ref_labels = ret_dict['pred_seg_cls_labels']
            query_bxyz = batch_dict['point_bxyz']
            e_ref, e_query = self.graph(ref_bxyz, query_bxyz)
            new_ret_dict = {}
            for key in ret_dict.keys():
                new_ret_dict[key] = scatter(ret_dict[key][e_ref], e_query, dim=0,
                                            dim_size=query_bxyz.shape[0], reduce='max')
            new_ret_dict['point_bxyz'] = batch_dict['point_bxyz']
            ret_dict = new_ret_dict
        
        ret_dict['batch_size'] = batch_dict['batch_size']
        self.forward_ret_dict = ret_dict

        return batch_dict
