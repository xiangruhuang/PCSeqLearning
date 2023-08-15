# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import numpy as np
import pickle
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2
import argparse


tf.get_logger().setLevel('INFO')


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

#    0: [1, 2, 3, 4], # Vehicle, npoint=(2000/4096)
#    1: [5, 6], # Cyclist, (6/4096)
#    2: [7], # Pedestrian (136/4096)
#    3: [8], # sign (36/4096)
#    4: [9], # traffic light (3/4096)
#    5: [10], # pole (73/4096)
#    6: [11], # construction cone (3/4096)
#    7: [12, 13], # bicycle, motorcycle (3/4096)
#    8: [14], # building (1293/4096)
#    9: [15, 16], # vegetation, tree trunk (1076/4096)
#    10: [17, 18, 19, 20], # curb, road, lane marker, other ground (1540/4096)
#    11: [21, 22], # walkable, sidewalk (670/4096)

class OpenPCDetWaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Truck', 'Cyclist']

    def waymo_visualization(self, prediction_infos, gt_infos, class_name):
        print('Start the waymo visualization...')
        assert len(prediction_infos) == len(gt_infos), '%d vs %d' % (prediction_infos.__len__(), gt_infos.__len__())

        from pcdet.utils import Visualizer
        vis = Visualizer()
        colors = []
        for i in range(12):
            color = [i % 2, i//2%2, (i//4)/2.0]
            colors.append(color)
        colors = np.array(colors).astype(np.float32)
        for info in prediction_infos:
            points = info['point_coords_for_seg']
            pred_seg_labels = info['pred_seg_labels']
            pred_seg_scores = info['pred_seg_scores']
            ps_p = vis.pointcloud('points', points[:, :3])
            ps_p.add_scalar_quantity('scores', pred_seg_scores)
            ps_p.add_color_quantity('seg-colors', colors[pred_seg_labels], enabled=True)
            ps_p.add_scalar_quantity('seg-labels', pred_seg_labels)
            import ipdb; ipdb.set_trace()
            vis.show()

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='')
    parser.add_argument('--sampled_interval', type=int, default=1, help='sampled interval for GT sequences')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the waymo format results...')
    eval = OpenPCDetWaymoDetectionMetricsEstimator()

    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]['annos']
        cur_info['frame_id'] = gt_infos[idx]['frame_id']
        gt_infos_dst.append(cur_info)

    waymo_AP = eval.waymo_visualization(
        pred_infos, gt_infos_dst, class_name=args.class_names
    )

    print(waymo_AP)


if __name__ == '__main__':
    main()
