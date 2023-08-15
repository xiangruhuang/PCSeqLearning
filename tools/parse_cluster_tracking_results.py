import _init_path
import numpy as np
import torch
import glob, os
from collections import defaultdict
from tqdm import tqdm
#import matplotlib.pyplot as plt
import argparse
from pcdet.utils import box_utils
from easydict import EasyDict

name_map = {
    #'l1_opt_r025': r'$\lambda=0.25$',
    #'l1_opt_r05': r'$\lambda=0.5$',
    #'l1_opt_r1': r'$\lambda=1$',
    #'l1_opt_r0125': r'$\lambda=0.125$',
    'TLS_multiradius_every8': r'$\lambda=0.06125$',
}

def estimate_velo(seq_boxes):
    velo = torch.zeros_like(seq_boxes.attr[:, 0])
    for trace_id in seq_boxes.trace_id.unique().tolist():
        trace_mask = (seq_boxes.trace_id == trace_id).reshape(-1)
        trace_frame = seq_boxes.frame[trace_mask]
        sorted_idx = torch.argsort(trace_frame)
        trace_attr = seq_boxes.attr[trace_mask][sorted_idx]
        trace_corners = box_utils.boxes_to_corners_3d(trace_attr)

        trace_velo = torch.zeros_like(trace_attr[:, 0])
        if trace_velo.numel() > 1:
            trace_velo[1:] = (trace_corners[1:] - trace_corners[:-1]).norm(p=2, dim=-1).mean(dim=-1)
            trace_velo[0] = trace_velo[1]

        velo[trace_mask.nonzero()[:, 0][sorted_idx]] = trace_velo
    seq_boxes.velo = velo
    seq_boxes.moving = velo > 5e-2
    return seq_boxes

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--result_dir', type=str, default='../output/waymo_sequence_registration/cluster_tracking/')
  parser.add_argument('--min_iou', dest='iou_threshold', type=float, default=0.7)
  parser.add_argument('--output_dir', type=str, default='cluster_tracking_stats')

  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  algorithms = glob.glob(f'{args.result_dir}/*')
  seq_dict = defaultdict(list)
  moving_seq_dict = defaultdict(list)
  for _algorithm in algorithms:
    algorithm = _algorithm.split('/')[-1]
    if not algorithm in ['TLS_multiradius_every8']:
      continue
    with open(f'{args.output_dir}/{algorithm}.txt', 'w') as fout:
      fout.write(f'algorithm={algorithm}\n')
      pth_files = glob.glob(f'{args.result_dir}/{algorithm}/*/all.pth')
      sequence_list = []
      for pth_file in tqdm(pth_files):
        sequence_id = pth_file.split('/')[-2]
        data = torch.load(pth_file, map_location='cpu')
        data = estimate_velo(EasyDict(data))
            
        # coverage, mIoU for all
        mask = data['best_iou'] > args.iou_threshold
        num_boxes = mask.shape[0]
        if (num_boxes == 0) or (not data['moving'].any()):
            continue
        mIoU = data['best_iou'].mean()
        coverage = mask.float().mean()

        moving_mIoU = data['best_iou'][data['moving']].mean()
        num_moving_boxes = data['moving'].long().sum()
        moving_coverage= mask[data['moving']].float().mean()

        sequence_list.append([sequence_id, coverage, num_boxes, mIoU, moving_coverage, num_moving_boxes, moving_mIoU])
        seq_dict[sequence_id].append([algorithm, coverage, moving_coverage])

      sequence_list = sorted(sequence_list, key = lambda x: x[4])
      for sequence_id, coverage, num_boxes, mIoU, moving_coverage, num_moving_boxes, moving_mIoU in sequence_list:
        fout.write(f'{sequence_id} num_boxes={num_boxes} coverage(all)={coverage:.4f} mIoU(all)={mIoU:.4f} coverage(moving)={moving_coverage:.4f} num_moving_boxes={num_moving_boxes} mIoU(moving)={moving_mIoU}\n')

  with open(f'{args.output_dir}/compare.txt', 'w') as fout:
    key_list = sorted(seq_dict.keys(), key = lambda k: seq_dict[k][0][2] - seq_dict[k][-1][2])
    for key in key_list:
      val = seq_dict[key]
      if len(val) > 1:
        min_v2 = min([v[2] for v in val])
        max_v2 = max([v[2] for v in val])
        if min_v2 == max_v2:
          continue
        fout.write(f'{key}\n')
        for v in val:
          fout.write(f'\t{v[0]}: all={v[1]:.4f}, moving={v[2]:.4f}\n')
