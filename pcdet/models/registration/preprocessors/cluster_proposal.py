import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
import os
from torch_scatter import scatter
from torch_cluster import knn
from collections import defaultdict

from .preprocessor_utils import ground_plane_removal
from pcdet.utils import common_utils
from pcdet.utils.timer import Timer
from pcdet.models.model_utils import graph_utils
from pcdet.ops.voxel import VoxelAggregation
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

class ClusterProposal(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.forward_dict = EasyDict()
        self.fake_params = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        component_keys = model_cfg["COMPONENT_KEYS"]

        self.component_keys = component_keys
        for i, component_key in enumerate(component_keys):
            graph_cfg = {}
            graph_cfg.update(model_cfg["GRAPH"])
            graph_cfg['RADIUS'] = graph_cfg["RADIUS"][i]
            graph = graph_utils.build_graph(graph_cfg, runtime_cfg=runtime_cfg)
            self.register_module(f'graph_{component_key}', graph)

    def propose_cluster(self, seq_dict):
        """Propose Object Clusters from a point cloud sequence.
        
        Args:
            point_fxyz ([N, 4], float32): the point cloud sequence after ground removal, first dimension is frame id.
            point_sweep ([N, 1], int32): point cloud frame id.
            segmentation_label ([N], int64): ground truth segmentation label (only used for debug).

        Returns:
            point_component ([N], int64): connected component id of each point

        """
        pcseq = EasyDict(
            fxyz=seq_dict['point_fxyz'],
            frame=seq_dict['point_sweep'],
        )

        num_points = pcseq.fxyz.shape[0]
        num_frames = pcseq.frame.max().long().item() + 1
        device = pcseq.fxyz.device
        sequence_id = seq_dict['frame_id'][0][:-4]

        os.makedirs(f'{self.model_cfg.DIR}', exist_ok=True)
        for comp_key in self.component_keys:
            with Timer(f'Propose Cluster for {comp_key}'):
                graph = getattr(self, f'graph_{comp_key}')
                torch.cuda.empty_cache()
                pcseq.component = torch.zeros(num_points, dtype=torch.int64, device=device)
                num_components_total = 0
                for frame_id in range(0, num_frames, 10):
                    frame_mask = (pcseq.frame.reshape(-1) >= frame_id) & (pcseq.frame.reshape(-1) < frame_id + 10)
                    if not frame_mask.any():
                        continue
                    pcframe = EasyDict(common_utils.filter_dict(pcseq, frame_mask))

                    # compute connected components via radius connectivity defined in self.graph
                    # Vars: 
                    #   e0, e1 ([num_edges], torch.int32): representing edges in the graph
                    e0, e1, _ = graph(pcframe, pcframe)

                    #with Timer(f'Connected Components for graph size {pcframe.frame.shape[0]}'):
                    num_components, component = graph_utils.connected_components(
                                                    torch.stack([e0, e1], dim=0),
                                                    pcframe.fxyz.shape[0])
                    
                    # update pcseq.component
                    pcseq.component[frame_mask] = component.to(pcseq.component) + num_components_total
                    num_components_total += num_components

                seq_dict[f'point_{comp_key}'] = pcseq.component
                print(f'Cluster Proposal {comp_key}: num_components={num_components}')
                #print(f'saving proposal results to {self.model_cfg.DIR}/{sequence_id}_{comp_key}.pth, num_components={num_components}')
                #torch.save(pcseq.component, f'{self.model_cfg.DIR}/{sequence_id}_{comp_key}.pth')

        return seq_dict

    def assign_instances_to_boxes(self, point_instance_label, bp_mask):
        """Assign (predicted or GT) object instances to boxes.

        Args:
            point_instance_label ([N], int64): instance id of each point
            bp_mask ([B, N], bool): whether point i is in box j

        Returns:
            instance2box (dictionary): assigned box id of each instance
            point_box_id ([N], int64): assgiend box id of each point
        """
        instance_list = point_instance_label.unique().detach().cpu().long().numpy().tolist()
        instance2box = defaultdict(lambda: -1)
        point_box_id = torch.zeros_like(point_instance_label) - 1

        for instance_label in instance_list:
            instance_mask = (point_instance_label == instance_label).reshape(-1)
            bi_mask = bp_mask[:, instance_mask] # box-instance mask
            if not bi_mask.any():
                continue
            box_id = bi_mask.long().sum(-1).argmax()
            instance2box[instance_label] = box_id
            point_box_id[instance_mask] = box_id

        return instance2box, point_box_id

    def format_boxes(self, seq_dict, num_frames):
        """

        Args:
            
        Returns:
            seq_boxes (dictionary): per-box attributes
                seq_boxes.attr ([B, 7], float32): each box's attributes
                seq_boxes.cls_label ([B, 7], float32): each box's class label {1, 2, 3}
                seq_boxes.trace_id ([B], int32): each box's trace id
                seq_boxes.frame ([B], int32): each box's frame id
        """
        # extract GT boxes attributes
        seq_box_attr = seq_dict['gt_box_attr'].reshape(-1, 7)
        seq_box_cls_label = seq_dict['gt_box_cls_label'].reshape(-1)
        seq_box_track_label = seq_dict['gt_box_track_label'].reshape(-1)
        seq_box_frame_id = seq_dict['gt_box_frame'].reshape(-1)
        seq_boxes = EasyDict(dict(
                       attr=seq_box_attr,
                       cls_label=seq_box_cls_label,
                       frame=seq_box_frame_id,
                       trace_id=seq_box_track_label,
                   ))

        return seq_boxes

    def evaluate_proposal(self, seq_dict):
        """Evaluate the proposed clusters by IoU to the corresponding GT cluster.

        Returns:
            instance_IoU: for each GT box, the maximum IoU (point-wise).
            trace_IoU: for each GT trace, the maximum IoU (point-wise).
        
        """
        num_frames = seq_dict['point_sweep'].max().long().item() + 1

        # extract and format GT boxes.
        # assign placeholder for the best coverage IoU of each box
        seq_boxes = self.format_boxes(seq_dict, num_frames)
        num_boxes = seq_boxes.attr.shape[0]
        if num_boxes == 0:
            num_points = seq_dict[f'point_{self.component_keys[0]}'].numel()
            for key in ['gt_box_id', 'gt_trace_id', 'pred_trace_id', 'pred_box_id']:
                seq_dict[f'point_{key}'] = seq_dict['segmentation_label'].new_zeros(num_points)-1
            return seq_dict
        seq_boxes.best_iou = torch.zeros_like(seq_boxes.attr[:, 0])
        
        # initialize trace-wise information (placeholders)
        num_traces = seq_boxes.trace_id.max().long().item() + 1
        traces = EasyDict(dict(
            best_iou=seq_boxes.attr.new_zeros(num_traces),
            cls_label=seq_boxes.trace_id.new_zeros(num_traces),
            min_frame=seq_boxes.trace_id.new_zeros(num_traces),
            max_frame=seq_boxes.trace_id.new_zeros(num_traces),
        ))
        for trace_id in range(num_traces):
            trace_mask = (seq_boxes.trace_id == trace_id).reshape(-1)
            traces.cls_label[trace_id] = seq_boxes.cls_label[trace_mask].median().long()
            traces.min_frame[trace_id] = seq_boxes.frame[trace_mask].min().long()
            traces.max_frame[trace_id] = seq_boxes.frame[trace_mask].max().long()

        sequence_id = seq_dict['frame_id'][0][:-4]

        for comp_key in self.component_keys:
            seq_points = EasyDict(
                fxyz=seq_dict['point_fxyz'],
                frame=seq_dict['point_sweep'],
                segmentation_label=seq_dict['segmentation_label'],
            )
            if 'instance_label' in seq_dict:
                seq_points['instance_label'] = seq_dict['instance_label']

            seq_points.component = seq_dict[f'point_{comp_key}']

            num_frames = seq_points.frame.max().long().item() + 1
            num_points = seq_points.frame.shape[0]
            device = seq_points.fxyz.device

            torch.cuda.empty_cache()
            
            # placeholder for assigned box indices and trace indices
            seq_points.gt_box_id = torch.zeros_like(seq_points.component) - 1
            seq_points.pred_box_id = torch.zeros_like(seq_points.component) - 1
            seq_points.gt_trace_id = torch.zeros_like(seq_points.component) - 1
            seq_points.pred_trace_id = torch.zeros_like(seq_points.component) - 1

            # evaluate the quality of proposed connected components
            for frame_id in range(num_frames):
                # extract a point cloud frame
                frame_mask = (seq_points.fxyz[:, 0] == frame_id).reshape(-1)
                if not frame_mask.any():
                    continue
                frame_points = EasyDict(common_utils.filter_dict(seq_points, frame_mask))
                
                # extract boxes in this frame
                frame_box_mask = (seq_boxes.frame == frame_id).reshape(-1)
                frame_boxes = EasyDict(common_utils.filter_dict(seq_boxes, frame_box_mask))
                num_frame_boxes = frame_boxes.attr.shape[0]
                if num_frame_boxes == 0:
                    continue
                
                # for this frame, compute point-in-box mask of shape [num_boxes, num_points]
                bp_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
                              frame_points.fxyz[:, 1:].detach().cpu(),
                              frame_boxes.attr.detach().cpu(),
                              ).long().to(frame_boxes.attr.device)

                # assign points to GT boxes. Translate box id to trace id.
                in_box_mask = (bp_mask == 1).any(0)
                frame_points.gt_box_id[in_box_mask] = bp_mask[:, in_box_mask].argmax(0)
                frame_points.gt_trace_id[in_box_mask] = frame_boxes.trace_id[frame_points.gt_box_id[in_box_mask]]
                
                # assign connected components to GT boxes
                component2box, frame_points.pred_box_id = self.assign_instances_to_boxes(
                                                              frame_points.component,
                                                              bp_mask)
                valid_mask = (frame_points.pred_box_id >= 0).reshape(-1)
                frame_points.pred_trace_id[valid_mask] = frame_boxes.trace_id[frame_points.pred_box_id[valid_mask]]

                # go through each (component, box) pair, compute the point-wise IoU
                # if iou > box.best_iou, update
                component_list = frame_points.component.unique().detach().tolist()
                for c in component_list:
                    corres_box_id = component2box[c]
                    if corres_box_id < 0:
                        continue

                    # update the best iou of the corresponding box
                    mask1 = (frame_points.gt_box_id == corres_box_id).reshape(-1)
                    mask2 = (frame_points.component == c).reshape(-1)
                    iou = (mask1 & mask2).float().sum().item() / ((mask1 | mask2).float().sum().item() + 1e-6)
                    if iou > frame_boxes.best_iou[corres_box_id]:
                        #print(f'updating box {corres_box_id}\'s iou to {iou}')
                        frame_boxes.best_iou[corres_box_id] = iou
                    
                    # update the best iou of a trace (multiple boxes across the entire sequence)
                    trace_id = frame_boxes.trace_id[corres_box_id]
                    if iou > traces.best_iou[trace_id]:
                        #print(f'updating trace {trace_id}\'s iou to {iou}')
                        traces.best_iou[trace_id] = iou

                # propagate frame point attributes to sequence points
                for key in ['gt_box_id', 'gt_trace_id', 'pred_trace_id', 'pred_box_id']:
                    seq_points[key][frame_mask] = frame_points[key]
                
                # propagate frame box attributes to sequence boxes
                for key in [f'best_iou']:
                    seq_boxes[key][frame_box_mask] = frame_boxes[key]

            seq_boxes[f'best_iou_after_{comp_key}'] = seq_boxes['best_iou'].clone()
            #torch.save(seq_boxes[f'best_iou_after_{comp_key}'], f'{self.model_cfg.DIR}/{sequence_id}.after_{comp_key}.box')
            num_frames_by_trace = traces.max_frame - traces.min_frame + 1
            trace_miou = ((traces.best_iou * num_frames_by_trace).sum() / (num_frames_by_trace.sum() + 1e-6)).item()
            box_miou = seq_boxes[f'best_iou_after_{comp_key}'].mean().item()
            print(f'mIoU({comp_key})={box_miou:.6f}, Trace-propagated mIoU({comp_key})={trace_miou:.6f}')
        
        # dump evaluation results
        os.makedirs(f'{self.model_cfg.DIR}', exist_ok=True)
        #torch.save(seq_boxes, f'{self.model_cfg.DIR}/{sequence_id}.box')

        for key in ['best_iou']:
            seq_dict[f'gt_box_{key}'] = seq_boxes.best_iou
        
        for key in ['best_iou']:
            seq_dict[f'gt_trace_{key}'] = traces.best_iou
            
        for key in ['gt_box_id', 'gt_trace_id', 'pred_trace_id', 'pred_box_id']:
            seq_dict[f'point_{key}'] = seq_points[key]

        return seq_dict

    def forward(self, seq_dict):
        seq_dict = self.propose_cluster(seq_dict)
        
        with Timer('Evaluate Proposal'):
            seq_dict = self.evaluate_proposal(seq_dict)

        return seq_dict

    def get_output_feature_dim(self):
        return 0
