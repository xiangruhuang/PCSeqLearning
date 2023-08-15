import torch
import torch.nn.functional as F

from ...utils import box_utils, loss_utils
from .point_head_template import PointHeadTemplate


class VoxelSegHead(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg,
                         num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.build_losses(self.model_cfg.LOSS_CONFIG)
    
    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss
    
    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['voxel_seg_gt_labels'].view(-1).long()
        point_cls_preds = self.forward_ret_dict['voxel_seg_pred_logits'].view(-1, self.num_class)

        cls_count = point_cls_preds.new_zeros(self.num_class)
        for i in range(self.num_class):
            cls_count[i] = (point_cls_labels == i).float().sum()
        positives = (point_cls_labels >= 0)
        positive_labels = point_cls_labels[positives]
        cls_weights = (1.0 * positives).float()
        pos_normalizer = torch.zeros_like(positives.float())
        pos_normalizer[positives] = cls_count[positive_labels]
        cls_weights /= torch.clamp(pos_normalizer, min=20.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        cls_loss_src = self.cls_loss_func(point_cls_preds.unsqueeze(0),
                                          one_hot_targets.unsqueeze(0),
                                          weights=cls_weights).squeeze(0)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['voxel_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'voxel_seg_loss_cls': point_loss_cls.item(),
        })

        for i in range(self.num_class):
            tb_dict.update({
                f'point_seg_cls{i}_num': cls_count[i].item(),
            })
        return point_loss_cls, tb_dict
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict
    
    def get_per_class_iou(self, batch_dict):
        pred_dicts = self.get_evaluation_results(batch_dict)
        ups, downs = pred_dicts[0]['ups'], pred_dicts[0]['downs']
        for p in pred_dicts[1:]:
            ups += p['ups']
            downs += p['downs']
        iou = ups / downs.clamp(min=1.0)
        return iou

    def get_evaluation_results(self, batch_dict):
        pred_logits = self.forward_ret_dict['voxel_seg_pred_logits']
        pred_scores = torch.sigmoid(pred_logits)
        point_coords = batch_dict['point_coords']
        pred_dicts = []
        for i in range(batch_dict['batch_size']):
            bs_mask = point_coords[:, 0] == i
            pred_confidences, pred_labels = pred_scores[bs_mask].max(-1)
            gt_labels = batch_dict['voxel_seg_labels'][bs_mask]
            valid_mask = (gt_labels >= 0)
            pred_labels = pred_labels[valid_mask]
            gt_labels = gt_labels[valid_mask]
            ups = pred_labels.new_zeros(self.num_class)
            downs = pred_labels.new_zeros(self.num_class)
            for cls in range(self.num_class):
                pred_mask = pred_labels == cls
                gt_mask = gt_labels == cls
                ups[cls] = (pred_mask & gt_mask).sum()
                downs[cls] = (pred_mask | gt_mask).sum()
            record_dict = dict(ups=ups, downs=downs)
            pred_dicts.append(record_dict)
        return pred_dicts

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

        ret_dict = {
            'voxel_seg_pred_logits': point_pred_logits,
        }

        point_pred_scores = torch.sigmoid(point_pred_logits)
        batch_dict['voxel_seg_pred_confidences'], batch_dict['voxel_seg_pred_labels'] = point_pred_scores.max(dim=-1)

        if self.training:
            ret_dict['voxel_seg_gt_labels'] = batch_dict['voxel_seg_labels']
        batch_dict.update(ret_dict)
        self.forward_ret_dict = ret_dict

        return batch_dict
