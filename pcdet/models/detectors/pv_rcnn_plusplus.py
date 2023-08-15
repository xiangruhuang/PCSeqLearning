from .detector3d_template import Detector3DTemplate

import torch

class PVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0

    def forward(self, batch_dict):
        batch_dict['points'] = torch.cat([batch_dict['point_bxyz'][:, 1:], batch_dict['point_feat']], dim=-1)
        batch_dict['voxel_points'] = torch.cat([batch_dict['voxel_point_xyz'], batch_dict['voxel_point_feat']], dim=-1)
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        if self.roi_head:
            batch_dict = self.roi_head.proposal_layer(
                batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.roi_head.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                if 'roi_valid_num' in batch_dict:
                    batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        if self.roi_head:
            batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            disp_dict.update({'num_pos': (batch_dict['gt_boxes'][:, :, 3] > 0.5).sum() / batch_dict['batch_size']})

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}

        loss = 0
        if 'dense_head' not in self.loss_disabled:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss = loss + loss_rpn
        if (self.point_head is not None) and ('point_head' not in self.loss_disabled):
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss = loss + loss_point
        if 'roi_head' not in self.loss_disabled:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss + loss_rcnn

        return loss, tb_dict, disp_dict
