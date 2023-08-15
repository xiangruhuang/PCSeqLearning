from .detector3d_template import Detector3DTemplate


class PVRCNNPlusPlusCoTrain(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0
        self.visualize = model_cfg.get('VISUALIZE', False)
        if self.visualize:
            from pcdet.utils import Visualizer
            self.vis = Visualizer()

    def forward(self, batch_dict):
        import ipdb; ipdb.set_trace()
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

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
        batch_dict = self.roi_head(batch_dict)
        if self.seg_head is not None:
            batch_dict = self.pfe_seg(batch_dict)
            batch_dict = self.seg_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            disp_dict.update({'num_pos': (batch_dict['gt_boxes'][:, :, 3] > 0.5).sum() / batch_dict['batch_size']})

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.seg_head is not None:
                point_seg_cls_preds = self.seg_head.forward_ret_dict['point_seg_cls_preds']
                point_coords = batch_dict['point_coords_for_seg']
                for i in range(batch_dict['batch_size']):
                    bs_mask = point_coords[:, 0] == i
                    pred_seg_scores, pred_seg_labels = point_seg_cls_preds[bs_mask].max(-1)
                    pred_dicts[i]['pred_seg_scores'] = pred_seg_scores
                    pred_dicts[i]['pred_seg_labels'] = pred_seg_labels
                    pred_dicts[i]['point_coords_for_seg'] = point_coords[bs_mask, 1:4]
                
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        if self.seg_head is not None:
            loss_seg, tb_dict = self.seg_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn + loss_seg
        return loss, tb_dict, disp_dict
