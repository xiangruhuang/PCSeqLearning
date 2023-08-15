import torch

from .detector3d_template import Detector3DTemplate

from pcdet.utils import common_utils

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['points'] = torch.cat([batch_dict['point_bxyz'][:, 1:], batch_dict['point_feat']], dim=-1)
        batch_dict['voxel_points'] = torch.cat([batch_dict['voxel_point_xyz'], batch_dict['voxel_point_feat']], dim=-1)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if self.visualizer:
                self.visualizer(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            for pred_dict, frame_id in zip(pred_dicts, batch_dict['frame_id'].reshape(-1)):
                pred_dict['scene_wise']['frame_id'] = frame_id
            if self.visualizer:
                max_num_objects = max([pred_dict['object_wise']['pred_box_attr'].shape[0] for pred_dict in pred_dicts])
                pred_dicts_stack = common_utils.stack_dicts_torch([p['object_wise'] for p in pred_dicts], pad_to_size=max_num_objects)
                batch_dict.update(pred_dicts_stack)
                batch_dict['pred_boxes'] = torch.cat([pred_dicts_stack['pred_box_attr'],
                                                      pred_dicts_stack['pred_box_cls_label'].unsqueeze(-1)],
                                                     dim=-1)

                self.visualizer(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dicts = batch_dict['final_box_dicts']
        pred_dicts = []
        for i, final_pred_dict in enumerate(final_pred_dicts):
            pred_dict = dict(
                point_wise=dict(),
                object_wise=dict(
                    pred_box_attr=final_pred_dict['pred_box_attr'],
                    pred_box_scores=final_pred_dict['pred_box_scores'],
                    pred_box_cls_label=final_pred_dict['pred_box_cls_label'],
                ),
                scene_wise=dict(
                    frame_id=batch_dict['frame_id'].reshape(-1)[i],
                )
            )
            pred_dicts.append(pred_dict)
                
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dicts[index]['pred_box_attr']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return pred_dicts, recall_dict
