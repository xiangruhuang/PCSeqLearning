import numpy as np
import torch
import pickle
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2
import argparse
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils

def eval_feature_leakage(gt_infos, pred_infos, class_names):
    assert len(gt_infos) == len(pred_infos), "infos should have same length"

    frame2gt = {}
    for gt_info in gt_infos:
        frame_id = gt_info['frame_id']
        frame2gt[frame_id] = gt_info
    
    frame2pred = {}
    for pred_info in pred_infos:
        frame_id = pred_info['frame_id']
        frame2pred[frame_id] = pred_info

    for frame_id in frame2gt.keys():
        gt_info = frame2gt[frame_id]
        pred_info = frame2pred[frame_id]
        gt_names = gt_info['name']
        gt_boxes = gt_info['gt_boxes_lidar'].astype(np.float32)
        import ipdb; ipdb.set_trace()
        tracking_difficulty = gt_info['tracking_difficulty']
        
        pred_scores = pred_info['score']
        pred_names = pred_info['name']
        pred_boxes = pred_info['boxes_lidar'].astype(np.float32)

        gt_boxes = torch.tensor(gt_boxes)
        pred_boxes = torch.tensor(pred_boxes)
        for class_name in class_names:
            gt_boxes_cls = gt_boxes[gt_names == class_name]
            pred_boxes_cls = pred_boxes[pred_names == class_name]
            if gt_boxes_cls.shape[0] == 0:
                continue
            if pred_boxes_cls.shape[0] == 0:
                iou1 = np.zeros(gt_boxes_cls.shape[0])
            else:
                iou = iou3d_nms_utils.boxes_iou3d_gpu(gt_boxes.cuda(), pred_boxes.cuda()).cpu().numpy()
                iou1 = iou.max(1)
        
        pass

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='')
    parser.add_argument('--sampled_interval', type=int, default=1, help='sampled interval for GT sequences')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the waymo format results via Feature Leakage Metric')

    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]['annos']
        cur_info['frame_id'] = gt_infos[idx]['frame_id']
        gt_infos_dst.append(cur_info)

    eval_feature_leakage(gt_infos_dst, pred_infos, args.class_names)

if __name__ == '__main__':
    main()
