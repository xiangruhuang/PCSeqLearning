import numpy as np
import pickle
import argparse
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infos', type=str)
    parser.add_argument('result', type=str)
    
    args = parser.parse_args()

    return args

def compute_coverage_by_ii(infos, results):
    frame_id_pool = [info['frame_id'] for info in infos]
    results = [r for r in results if r['frame_id'] in frame_id_pool]
    for info, result in zip(infos, results):
        assert info['frame_id'] == result['frame_id']
        pred_boxes = result['boxes_lidar']
        pred_names = result['name']
        pred_scores = result['score']

        gt_boxes = info['annos']['gt_boxes_lidar']
        gt_ii = info['annos']['interaction_index']
        gt_tracking_difficulty = info['annos']['tracking_difficulty']
        gt_difficulty = info['annos']['difficulty']
        gt_names = info['annos']['name']
        
        for name in ['Vehicle', 'Pedestrian', 'Cyclist']:
            pred_mask = pred_names == name
            gt_mask = gt_names == name
            pred_box = torch.from_numpy(pred_boxes[pred_mask]).float().cuda()
            gt_box = torch.from_numpy(gt_boxes[gt_mask]).float().cuda()
            
            pred_score = pred_scores[pred_mask]

            import ipdb; ipdb.set_trace()
            if (pred_box.shape[0] > 0) and (gt_box.shape[0] > 0):
                iou = boxes_iou3d_gpu(pred_box, gt_box)
                

    
    pass


def main():
    args = parse_args()
    print(args)
    with open(args.infos, 'rb') as fin:
        infos = pickle.load(fin)
    
    with open(args.result, 'rb') as fin:
        results = pickle.load(fin)
    
    compute_coverage_by_ii(infos, results)

if __name__ == '__main__':
    main()
