import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
import os
from torch_scatter import scatter
from torch_cluster import knn

from .preprocessor_utils import ground_plane_removal
from .registration_utils import (
    register_to_next_frame,
    robust_mean,
    efficient_robust_sum,
    robust_median
)
from pcdet.utils import common_utils
from pcdet.utils.timer import Timer
from pcdet.models.model_utils import graph_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.models.visualizers import GeometryVisualizer
from pcdet.models.model_utils.grid_sampling import GridSampling3D

def random_rotation():
    U0, S0, V0T = torch.linalg.svd(torch.randn(3, 3))
    V0 = V0T.T
    sign = torch.ones_like(S0)
    sign[-1] = (V0 @ U0.T).det()
    return V0 @ torch.diag_embed(sign) @ U0.T

def visualize_cluster(sxyz, T):
    import polyscope as ps; ps.set_up_dir('z_up'); ps.init()
    for i in range(T.shape[0]):
        Ti = T[i]
        xyz = sxyz[:, 1:].clone()
        xyz = xyz @ Ti[:3, :3].T + Ti[:3, 3]
        ps_comp = ps.register_point_cloud(f'component-{i}', xyz.detach().cpu(), radius=5e-4)

def sample_frame(grid_sampler, frame):
    _, inv = grid_sampler(frame.fxyz, return_inverse=True)
    new_frame = EasyDict(dict())
    num_grids = inv.max().long().item()+1
    for key in ['fxyz', 'stationary']:
        new_frame[key] = scatter(frame[key].float(), inv, dim=0, dim_size=num_grids,
                                 reduce='mean')
    new_frame['stationary'] = new_frame['stationary'] > 0.5
    
    for key in ['component', 'frame']:
        new_frame[key] = robust_median(frame[key].long().reshape(-1), inv, num_grids)

    return new_frame

def rotation_angle(R):
    """
    Args:
        R [..., 3, 3]
    """
    R_diag = R.reshape(*R.shape[:-2], 9)[..., [0, 4, 8]]
    theta = ((R_diag.sum(-1) - 1) / 2).clamp(min=-1, max=1).arccos()
    return theta

def random_color(labels):
    num_labels = max(labels.max().item() + 1, 1)
    labels = labels.clamp(min=0)
    return common_utils.separate_colors(num_labels)[labels.detach().cpu().numpy()]

def pose_inv(T):
    """
    Args:
        T [..., 4, 4]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    RT = R.transpose(-1, -2)
    Tinv = T.clone()
    Tinv[..., :3, :3] = RT
    Tinv[..., :3, 3:] = -RT @ t.unsqueeze(-1)
    return Tinv

def dist_compensate(comp_deg):
    thresholds = [0, 10, 40, 100, 200, 400, 10000000]
    comp_dist =  [  1, 0.5, 0.3, 0.2, 0.1, 0.0]
    compensate = torch.zeros_like(comp_deg).float()
    for i in range(1, len(thresholds)):
        mask = (comp_deg >= thresholds[i-1]) & (comp_deg < thresholds[i])
        compensate[mask] = comp_dist[i-1]
    return compensate

def component_diameter(frame_points):
    point_xyz = frame_points.fxyz[:, 1:]
    num_components = frame_points.component.max().long().item() + 1

    # remove empty component
    comp_deg = efficient_robust_sum(torch.ones_like(point_xyz[:, 0]), frame_points.component, num_components)
    valid_comp_mask = (comp_deg > 0.5).reshape(-1)

    # compute component geometric center
    comp_center = efficient_robust_sum(point_xyz, frame_points.component, num_components)
    comp_center[valid_comp_mask] = comp_center[valid_comp_mask] / comp_deg[valid_comp_mask, None]

    # compute component diameter
    comp_diameter = point_xyz.new_zeros(num_components)
    point_dist = (point_xyz - comp_center[frame_points.component]).norm(p=2, dim=-1)
    comp_diameter = scatter(point_dist, frame_points.component,
                            reduce='max', dim=0, dim_size=num_components)

    return comp_diameter * 2

def component_center(frame_points):
    point_xyz = frame_points.fxyz[:, 1:]
    num_components = frame_points.component.max().long().item() + 1

    # remove empty component
    comp_deg = efficient_robust_sum(torch.ones_like(point_xyz[:, 0]), frame_points.component, num_components)
    valid_comp_mask = (comp_deg > 0.5).reshape(-1)

    # compute component geometric center
    comp_center = efficient_robust_sum(point_xyz, frame_points.component, num_components)
    comp_center[valid_comp_mask] = comp_center[valid_comp_mask] / comp_deg[valid_comp_mask, None]

    return comp_center

def filter_components(frame_points, max_diameter=12.5):
    """Remove components that are too large and reorder.
    Args:
        frame_points (dictionary):
            frame_points.component
            frame_points.fxyz
    Returns:
        valid_comp_mask [C]
            
    """
    point_xyz = frame_points.fxyz[:, 1:]
    num_components = frame_points.component.max().long().item() + 1

    # remove empty component
    comp_deg = scatter(torch.ones_like(point_xyz[:, 0]), frame_points.component,
                       reduce='sum', dim=0, dim_size=num_components)
    valid_comp_mask = (comp_deg > 0.5).reshape(-1)

    if max_diameter > 0:
        # compute component diameter
        comp_diameter = component_diameter(frame_points)
        
        # update valid component mask by diameter limit
        valid_comp_mask = valid_comp_mask & (comp_diameter < max_diameter)

    return valid_comp_mask

    ## remove unwanted components
    #valid_point_mask = valid_comp_mask[frame_points.component]
    #filtered_points = EasyDict(common_utils.filter_dict(frame_points, valid_point_mask))

    ## compute new component id for each component
    #new_comp_id = frame_points.component.new_zeros(num_components)
    #num_new_components = valid_comp_mask.long().sum().item()
    #new_comp_id[valid_comp_mask] = torch.arange(num_new_components).to(frame_points.component)
    #filtered_points.component = new_comp_id[filtered_points.component]

    #return filtered_points

def smooth_velo(_comp_velos, comp_center_diffs, frame_id, next_frame_id, weight0 = 1, weight=10, num_itr=300, stopping=1e-3):
    """[frame_id, next_frame_id]

    """
    if frame_id == next_frame_id:
        return _comp_velos

    if frame_id < next_frame_id:
        track_dir = 1
    else:
        track_dir = -1
        temp = frame_id
        frame_id = next_frame_id
        next_frame_id = temp
        
    comp_velos = nn.Parameter(_comp_velos, requires_grad=True)
    opt = torch.optim.AdamW([comp_velos], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [100, 200, 300])
    last_loss = 1e10
    countdown = 3
    for itr in range(num_itr):
        opt.zero_grad()
        loss_fit = (comp_velos[:, frame_id:(next_frame_id+1), :2] - comp_center_diffs[:, frame_id:(next_frame_id+1), :2]).square().mean()
        loss_smooth = (comp_velos[:, frame_id:next_frame_id, :2] - comp_velos[:, (frame_id+1):(next_frame_id+1), :2]).abs().mean()
        loss = loss_fit * weight0 + loss_smooth * weight
        loss.backward()
        opt.step()
        scheduler.step()
        if last_loss - loss.item() < stopping:
            countdown -= 1
        else:
            countdown = 3
        if countdown <= 0:
            break
        last_loss = loss.item()
        #print(f'iter={itr}, loss={loss:.6f}, loss_fit={loss_fit}, loss_smooth={loss_smooth}')

    return comp_velos.data

class ClusterTracking(nn.Module):
    """
    Build a graph that between adjacent frames.
    
    """
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.forward_dict = EasyDict()
        
        registration_cfg = model_cfg.REGISTRATION
        self.stopping_delta = registration_cfg["STOPPING_DELTA"]
        graph_cfg = registration_cfg['GRAPH']
        self.radius_list = graph_cfg['RADIUS']
        self.voxel_size_list = registration_cfg["VOXEL_SIZE"]
        for i, (radius, voxel_size) in enumerate(zip(self.radius_list, self.voxel_size_list)):
            _graph_cfg = {}
            _graph_cfg.update(graph_cfg)
            _graph_cfg['RADIUS'] = radius
            graph = graph_utils.build_graph(_graph_cfg, {})
            self.register_module(f"registration_graph_{i}", graph)
            grid_sampler = GridSampling3D(voxel_size)
            self.register_module(f"sampler_{i}", grid_sampler)

        self.nn_graph = graph_utils.build_graph(model_cfg["NN_GRAPH"], runtime_cfg=runtime_cfg)
        self.visualize = runtime_cfg.get("visualize", False)
        if self.visualize:
            vis_model_cfg = dict(
                                 ENABLED=True,
                                 UP_DIR='z_up',
                            )
            self.visualizer = GeometryVisualizer(vis_model_cfg)

        
        params = model_cfg.get("TRACKING_PARAMS", {})
        self.reg_error_coeff = params.get("REGISTRATION_ERROR_COEFFICIENT", 0.13)
        self.track_interval = params.get("TRACK_INTERVAL", 10)
        self.angle_threshold = params.get("ANGLE_THRESHOLD", 45)
        self.min_move_frame = params.get("MIN_MOVE_FRAME", 6)

        self.component_keys = model_cfg["COMPONENT_KEYS"]
    
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
        seq_box_gt_velo = seq_dict['gt_box_velo'].reshape(-1)
        seq_box_moving = seq_dict['moving'].reshape(-1)
        seq_boxes = EasyDict(dict(
                       attr=seq_box_attr,
                       cls_label=seq_box_cls_label,
                       frame=seq_box_frame_id,
                       trace_id=seq_box_track_label,
                       velo=seq_box_gt_velo,
                       moving=seq_box_moving,
                   ))

        return seq_boxes

    def visualize_frame(self, seq_points, frame, extracted):

        import polyscope as ps; ps.set_up_dir('z_up'); ps.init()
        num_components = frame.component.max().long().item() + 1
        frame_id = frame.fxyz[0, 0].round().long().item()
        ps_ex = ps.register_point_cloud(f'extracted-from-{frame_id}', extracted.fxyz[:, 1:].detach().cpu().numpy(), radius=4e-4, enabled=False)
        ps_ex.add_scalar_quantity('scalar/frame', extracted.fxyz[:, 0].detach().cpu().numpy())
        colors = common_utils.separate_colors(num_components)
        ps_ex.add_scalar_quantity('scalar/component', extracted.component.detach().cpu().numpy())
        ps_ex.add_scalar_quantity('scalar/moving', extracted.moving.detach().cpu().numpy())
        ps_ex.add_color_quantity('labels/component', colors[extracted.component.detach().cpu().numpy()])

        pass

    def extract_traces_and_update_boxes(self, all_points, extracted, seq_boxes):
        full_extracted = EasyDict(dict(
            fxyz=[],
            component=[],
            segmentation_label=[],
            instance_label=[],
            original_indices=[],
            frame_indices=[],
            moving=[],
        ))
        transforms = extracted.pop('transforms')
        device = all_points.fxyz.device
        if self.visualize:
            min_frame_id = extracted.fxyz[:, 0].round().long().min().item()
            max_frame_id = extracted.fxyz[:, 0].round().long().max().item()
            frame_id_mask = ((all_points.frame >= min_frame_id) & (all_points.frame <= max_frame_id)).reshape(-1)

            segment = EasyDict(dict(
                            type='point_cloud',
                            name=f'all_points',
                            xyz=all_points.fxyz[frame_id_mask, 1:],
                            radius=0.06,
                            kwargs=dict(enabled=True),
                            scalars=EasyDict(dict(
                                        frame=dict(values=all_points.frame[frame_id_mask].reshape(-1).detach().cpu(), enabled=True),
                                    )),
                            colors=EasyDict(dict(
                                   )),
                            vectors=dict(),
                        ))
            self.visualizer.register_point_cloud(segment)

        try:
            num_components = extracted.component.max().long().item() + 1
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(e)
        component_hit = extracted.component.new_zeros(num_components)
        component_size_min = scatter(extracted.fxyz[:, 0], extracted.component, dim=0,
                                     dim_size=num_components, reduce='min').round().long()
        component_size_max = scatter(extracted.fxyz[:, 0], extracted.component, dim=0,
                                     dim_size=num_components, reduce='max').round().long()
        component_size = component_size_max - component_size_min + 1

        # update box-wise information
        for fid in extracted.fxyz[:, 0].round().long().unique().detach().tolist():
            frame_box_mask = (seq_boxes.frame == fid).reshape(-1)
            frame_boxes = EasyDict(common_utils.filter_dict(seq_boxes, frame_box_mask))

            ref_frame_mask = (all_points.frame == fid).reshape(-1)
            ref_frame_points = EasyDict(common_utils.filter_dict(all_points, ref_frame_mask))
            num_ref_frame_points = ref_frame_mask.sum().long().item()
            ref_frame_points.frame_indices = torch.arange(num_ref_frame_points).to(extracted.component)

            if frame_box_mask.any():
                ref_bp_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
                                  ref_frame_points.fxyz[:, 1:].detach().cpu(),
                                  frame_boxes.attr.detach().cpu(),
                                  ).long().to(frame_boxes.attr.device)
                ref_frame_points.gt_box_id = ref_bp_mask.argmax(0)
                ref_frame_points.gt_box_id[ref_bp_mask.max(0)[0] == 0] = -1
            else:
                ref_frame_points.gt_box_id = torch.zeros(num_ref_frame_points).to(
                                                         ref_frame_points.frame
                                                        ) - 1

            extracted_frame_mask = (extracted.fxyz[:, 0].round().long() == fid).reshape(-1)
            _one_extracted_frame = EasyDict(common_utils.filter_dict(extracted, extracted_frame_mask))
            
            self.nn_graph.radius *= 1.732
            e_ext, e_ref, _ = self.nn_graph(_one_extracted_frame, ref_frame_points)
            self.nn_graph.radius /= 1.732

            component_center = robust_mean(_one_extracted_frame.fxyz[:, 1:3], _one_extracted_frame.component, num_components)
            component_diameter = (_one_extracted_frame.fxyz[:, 1:3] - component_center[_one_extracted_frame.component]).norm(p=2, dim=-1)
            component_diameter = scatter(component_diameter, _one_extracted_frame.component,
                                         dim=0, dim_size=num_components, reduce='max')
            valid_edge_mask = (_one_extracted_frame.fxyz[e_ext, -1] - ref_frame_points.fxyz[e_ref, -1]) < 0.5
            dist = (ref_frame_points.fxyz[e_ref, 1:3] - component_center[_one_extracted_frame.component[e_ext]]).norm(p=2, dim=-1)
            valid_edge_mask &= (dist < component_diameter[_one_extracted_frame.component[e_ext]] + 0.05)
            valid_edge_mask &= (_one_extracted_frame.fxyz[e_ext, -1] - ref_frame_points.fxyz[e_ref, -1]) > -0.05
            e_ext, e_ref = e_ext[valid_edge_mask], e_ref[valid_edge_mask]
            one_extracted_frame = EasyDict(dict(
                fxyz=ref_frame_points.fxyz[e_ref],
                component=_one_extracted_frame.component[e_ext],
                segmentation_label=ref_frame_points.full_segmentation_label[e_ref],
                instance_label=ref_frame_points.full_instance_label[e_ref],
                frame_indices=ref_frame_points.frame_indices[e_ref],
                original_indices=ref_frame_mask.nonzero()[e_ref],
                moving=_one_extracted_frame.moving[e_ext],
            ))

            for key in one_extracted_frame.keys():
                full_extracted[key].append(one_extracted_frame[key])

            if frame_box_mask.any():
                bp_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
                              one_extracted_frame.fxyz[:, 1:].detach().cpu(),
                              frame_boxes.attr.detach().cpu(),
                              ).long().to(frame_boxes.attr.device)
            
                for idx, c in enumerate(one_extracted_frame.component.unique().detach().tolist()):
                    component_mask = (one_extracted_frame.component == c).reshape(-1)

                    if not bp_mask[:, component_mask].any():
                        continue

                    assigned_box_id = bp_mask[:, component_mask].float().sum(-1).argmax()

                    m1 = (ref_frame_points.gt_box_id == assigned_box_id)
                    mask = torch.zeros(ref_frame_points.fxyz.shape[0], device=device).bool()
                    mask[one_extracted_frame.frame_indices[component_mask]] = True

                    iou = (mask & m1).long().sum() / ((mask | m1).long().sum() + 1e-6)
                    if iou > 0.7:
                        component_hit[c] += 1
                    #if self.visualize:
                    #    import polyscope as ps; ps.set_up_dir('z_up'); ps.init()
                    #    ps_c = ps.register_point_cloud('component', one_extracted_frame.fxyz[component_mask, 1:].detach().cpu(), radius=8e-4)
                    #    ps_b = ps.register_point_cloud('ref_frame_points in box', ref_frame_points.fxyz[m1, 1:].detach().cpu(), radius=8e-4)
                    #    print(f'IoU={iou}')
                    #    ps.show()

                    if iou > frame_boxes.best_iou[assigned_box_id]:
                        frame_boxes.best_iou[assigned_box_id] = iou

            # propagate to box sequence
            seq_boxes.best_iou[frame_box_mask] = frame_boxes.best_iou
        
        for key in full_extracted.keys():
            if len(full_extracted[key]) == 0:
                import ipdb; ipdb.set_trace()
                print(key)
                print(all_points.fxyz.shape)
                print(extracted.fxyz.shape)
            full_extracted[key] = torch.cat(full_extracted[key], dim=0)

        full_extracted.component_hit = component_hit
        full_extracted.component_size = component_size
        full_extracted.transforms = transforms

        return full_extracted, seq_boxes

    def track_frame(self, seq_points, frame, seq_boxes):
        """Track connected components in a frame `frame` along a point cloud sequence `seq_points`.
        Args:
            seq_points (dictionary): point cloud sequence.
            frame (dictionary): point cloud frame to be tracked.

        Returns:
            transforms ([num_components, num_frames, 4, 4], float32): component wise transformations.
            reg_errors ([num_components, num_frames])
            comp_edge_ratios ([num_components, num_frames])
        """
        num_components = frame.component.max().long().item() + 1
        frame_id = frame.frame[0].long().item()
        frame_mask = (seq_points.frame == frame_id).reshape(-1)
        min_frame_id = max(seq_points.frame.min().long().item(), frame_id - self.track_interval)
        max_frame_id = min(seq_points.frame.max().long().item(), frame_id + self.track_interval)
        comp_deg = efficient_robust_sum(torch.ones_like(frame.component), frame.component, num_components)
        print(f'Tracking frame-{frame_id:03d} with {num_components:03d} components')

        comp_diameter = component_diameter(frame)

        # initialize cluster-wise transformations to identity.
        transforms = torch.diag_embed(
                         frame.fxyz.new_ones(num_components, max_frame_id-min_frame_id+1, 4).double()
                     )
        reg_errors = frame.fxyz.new_zeros(num_components, max_frame_id+1)
        comp_edge_ratios = frame.fxyz.new_zeros(num_components, max_frame_id+1)
        comp_min_frame_id = frame.component.new_zeros(num_components) + frame_id
        comp_max_frame_id = frame.component.new_zeros(num_components) + frame_id
        comp_velos = frame.fxyz.new_zeros(num_components, max_frame_id+1, 3)
        comp_centers = frame.fxyz.new_zeros(num_components, max_frame_id+1, 3)
        comp_centers[:, frame_id] = component_center(frame)
        comp_center_diffs = frame.fxyz.new_zeros(num_components, max_frame_id+1, 3)

        if self.visualize:
            cur_frame = EasyDict(dict(
                            type='point_cloud',
                            name=f'frame-{frame_id:03d}',
                            xyz=frame.fxyz[:, 1:],
                            radius=0.06,
                            kwargs=dict(enabled=False),
                            scalars=EasyDict(dict(
                                        scalars_component=dict(values=frame.component.detach().cpu()),
                                    )),
                            colors=EasyDict(dict(
                                        class_labels_component=dict(values=random_color(frame.component)),
                                   )),
                            vectors=dict(),
                        ))
            self.visualizer.register_point_cloud(cur_frame)
            
            num_components = frame.component.max().long().item() + 1

            #ps_f = ps.register_point_cloud(f'frame-{frame_id:03d}', frame.fxyz[:, 1:].detach().cpu(), radius=2e-4, enabled=False)
            #ps_f.add_scalar_quantity('component', frame.component.detach().cpu().reshape(-1))
        
            frame_id_mask = ((seq_points.frame >= frame_id - self.track_interval) & (seq_points.frame <= frame_id + self.track_interval)).reshape(-1)

            segment = EasyDict(dict(
                            type='point_cloud',
                            name=f'seq_points',
                            xyz=seq_points.fxyz[frame_id_mask, 1:],
                            radius=0.06,
                            kwargs=dict(enabled=True),
                            scalars=EasyDict(dict(
                                        frame=dict(values=seq_points.frame[frame_id_mask].reshape(-1).detach().cpu(), enabled=True),
                                        extracted=dict(values=seq_points.extracted[frame_id_mask].reshape(-1).detach().cpu(), enabled=True),
                                    )),
                            colors=EasyDict(dict(
                                       gt_box_id=dict(values=random_color(seq_points.gt_box_id[frame_id_mask])),
                                   )),
                            vectors=dict(),
                        ))
            self.visualizer.register_point_cloud(segment)

            #ps_s = ps.register_point_cloud('seq_points', seq_points.fxyz[frame_id_mask, 1:].detach().cpu(), radius=2e-4)
            #ps_s.add_scalar_quantity('frame', seq_points.frame[frame_id_mask].detach().cpu().reshape(-1), enabled=True)
            #ps_s.add_color_quantity('gt_box_id', torch.randn(10000, 3)[seq_points.gt_box_id[frame_id_mask].detach().cpu().reshape(-1)])
            #ps_s.add_scalar_quantity('extracted', seq_points.extracted[frame_id_mask].detach().cpu().reshape(-1), enabled=True)

            frame_box_mask = ((seq_boxes.frame >= frame_id - self.track_interval) & (seq_boxes.frame <= frame_id + self.track_interval)).reshape(-1)
            
            segment_boxes = EasyDict(common_utils.filter_dict(seq_boxes, frame_box_mask))

            _segment_boxes = EasyDict(dict(
                                 type='boxes',
                                 name=f'segment_boxes',
                                 attr=segment_boxes.attr[:, :7],
                                 kwargs=dict(),
                                 cls_label=segment_boxes.cls_label,
                                 scalars=EasyDict(dict(
                                             gt_box_frame=dict(values=segment_boxes.frame.reshape(-1).detach().cpu(), defined_on='cells', enabled=True),
                                             gt_box_best_iou=dict(values=segment_boxes.best_iou.reshape(-1).detach().cpu(), defined_on='cells', enabled=True),
                                             gt_box_velo=dict(values=segment_boxes.velo.reshape(-1).detach().cpu(), defined_on='cells', enabled=True),
                                             moving=dict(values=segment_boxes.moving.reshape(-1).detach().cpu(), defined_on='cells', enabled=True),
                                         )),
                                 colors=EasyDict(dict()),
                                 vectors=dict(),
                             ))
            self.visualizer.register_boxes(_segment_boxes)

        valid_comp_mask = filter_components(frame)
        valid_point_mask = valid_comp_mask[frame.component]

        extracted_fxyz = [frame.fxyz[valid_point_mask]]
        extracted_component = [frame.component[valid_point_mask]]
        extracted_segmentation_label = [frame.segmentation_label[valid_point_mask]]
        extracted_indices = [valid_point_mask.nonzero().reshape(-1)]
        extracted_original_indices = [frame_mask.nonzero()[valid_point_mask].reshape(-1)]

        # register from frame:<frame_id> to its nearby frames
        last_velo = None
        for track_dir in [-1, 1]:
            # backward, then forward
            next_frame_id = frame_id + track_dir
            stopped = torch.zeros(num_components).to(frame.component).bool()
            moving = torch.ones_like(stopped)
            stopped = stopped | ~valid_comp_mask
            moving = moving & valid_comp_mask
            last_xyz = frame.fxyz[:, 1:].clone()
            if (track_dir == 1) and (frame_id > 0):
                last_velo = comp_velos[:, frame_id]

            while (min_frame_id <= next_frame_id <= max_frame_id) and (~stopped).any():
                print(f'Tracking {frame_id} --> {next_frame_id}')
                # extract the next frame
                next_frame_mask = (seq_points.frame == next_frame_id).reshape(-1)
                next_frame = EasyDict(common_utils.filter_dict(seq_points, next_frame_mask))

                # Register source frame to target frame.
                # Args:
                #   frame: source frame.
                #   next_frame: target frame.
                # Returns:
                #   T: store component-wise transformation.
                #   frame: the transformed frame.
                #   l1_reg_error ([C], float64): component-wise registration error.
                frame.require_corres = ((~stopped)[frame.component]).reshape(-1)
                transforms[:, next_frame_id - min_frame_id] = transforms[:, next_frame_id - min_frame_id - track_dir]
                if last_velo is not None:
                    trans = last_velo.clone()
                    trans[stopped] = 0
                    frame.fxyz[:, 1:] += (trans[frame.component])*track_dir
                    transforms[:, next_frame_id - min_frame_id, :3, 3] += trans.double()*track_dir
                for i, radius in enumerate(self.radius_list):
                    grid_sampler = getattr(self, f"sampler_{i}")
                    graph = getattr(self, f"registration_graph_{i}")
                    subsampled_frame = sample_frame(grid_sampler, frame)
                    try:
                        subsampled_next_frame = sample_frame(grid_sampler, next_frame)
                    except Exception as e:
                        import ipdb; ipdb.set_trace()
                        print(e)
                    if self.visualize:
                        cur_sub_frame = EasyDict(dict(
                                            type='point_cloud',
                                            name=f'subframe{i}-{frame_id:03d}',
                                            xyz=subsampled_frame.fxyz[:, 1:],
                                            radius=0.06,
                                            kwargs=dict(enabled=False),
                                            scalars=EasyDict(dict(
                                                        scalars_component=dict(values=subsampled_frame.component.detach().cpu()),
                                                    )),
                                            colors=EasyDict(dict(
                                                        class_labels_component=dict(values=random_color(subsampled_frame.component)),
                                                   )),
                                            vectors=dict(),
                                        ))
                        self.visualizer.register_point_cloud(cur_sub_frame)
                        cur_next_frame = EasyDict(dict(
                                            type='point_cloud',
                                            name=f'nextframe{i}-{frame_id:03d}',
                                            xyz=subsampled_next_frame.fxyz[:, 1:],
                                            radius=0.06,
                                            kwargs=dict(enabled=False),
                                            scalars=EasyDict(dict(
                                                        scalars_component=dict(values=subsampled_next_frame.component.detach().cpu()),
                                                    )),
                                            colors=EasyDict(dict(
                                                        class_labels_component=dict(values=random_color(subsampled_next_frame.component)),
                                                   )),
                                            vectors=dict(),
                                        ))
                        self.visualizer.register_point_cloud(cur_next_frame)

                    with Timer(f"REGISTRATION at Level {i}"):
                        subsampled_frame, T, _l1_reg_error, _comp_edge_ratio = \
                                register_to_next_frame(
                                    graph, subsampled_frame, subsampled_next_frame, num_components,
                                    self.model_cfg.ANGLE_REGULARIZER,
                                    max_iter=80, stopping_delta=self.stopping_delta[i],
                                )
                        if i == 0:
                            comp_edge_ratio = _comp_edge_ratio
                        if i == len(self.radius_list) - 1:
                            l1_reg_error = _l1_reg_error
                        frame.fxyz[:, 1:] = (T[frame.component, :3, :3] @ frame.fxyz[:, 1:, None].double()).squeeze(-1).float() + T[frame.component, :3, 3].float()
                        transforms[:, next_frame_id - min_frame_id] = T @ transforms[:, next_frame_id - min_frame_id]

                comp_centers[:, next_frame_id] = component_center(frame)
                
                # estimate velocity
                point_velo = (frame.fxyz[:, 1:] - last_xyz)*track_dir
                comp_velo = robust_mean(point_velo, frame.component, num_components)
                comp_velo[:, 2] = 0
                comp_velos[:, next_frame_id] = comp_velo
                comp_center_diffs[:, next_frame_id] = (comp_centers[:, next_frame_id] - comp_centers[:, next_frame_id - track_dir])*track_dir
                comp_velos = smooth_velo(comp_velos, comp_center_diffs, frame_id+track_dir, next_frame_id)
                delta_velo = comp_velos[:, next_frame_id] - comp_velo
                comp_velo = comp_velos[:, next_frame_id]
                frame.fxyz[:, 1:] += delta_velo[frame.component] * track_dir
                transforms[:, next_frame_id - min_frame_id, :3, 3] += delta_velo * track_dir
                last_xyz = frame.fxyz[:, 1:].clone()

                if self.visualize:
                    # Visualize
                    point_reg_error = l1_reg_error[frame.component]

                    _next_frame = EasyDict(dict(
                                      type='point_cloud',
                                      name=f'frame-{next_frame_id:03d}',
                                      xyz=frame.fxyz[:, 1:],
                                      radius=0.06,
                                      kwargs=dict(enabled=False),
                                      scalars=EasyDict(dict(
                                                  reg_error=dict(values=point_reg_error.detach().cpu()),
                                                  comp_edge_ratio=dict(values=comp_edge_ratio[frame.component].detach().cpu()),
                                                  diam=dict(values=comp_diameter[frame.component].detach().cpu()),
                                              )),
                                      colors=EasyDict(dict(
                                             )),
                                      vectors=EasyDict(dict(
                                                  velo=dict(values=comp_velo[frame.component].detach().cpu(), vectortype='ambient'),
                                                  smooth_velo=dict(values=comp_velos[frame.component, next_frame_id].detach().cpu(), vectortype='ambient'),
                                              )),
                                  ))
                    ps_frame = self.visualizer.register_point_cloud(_next_frame)
                    #ps_frame = ps.register_point_cloud(f'frame-{next_frame_id:03d}',
                    #                                   frame.fxyz[:, 1:].detach().cpu(), radius=2e-4, enabled=False)
                    #ps_frame.add_scalar_quantity('reg_error', point_reg_error.detach().cpu())
                    #ps_frame.add_vector_quantity('velo', comp_velo[frame.component].detach().cpu(), vectortype='ambient')
                    #ps_frame.add_vector_quantity('smooth-velo', comp_velos[frame.component, next_frame_id].detach().cpu(), vectortype='ambient')
                    #ps_frame.add_scalar_quantity('comp_edge_ratio', comp_edge_ratio[frame.component].detach().cpu())
                    #ps_frame.add_scalar_quantity('diam', comp_diameter[frame.component].detach().cpu())
                
                # check stopping condition
                stopped = stopped | (l1_reg_error > self.reg_error_coeff * comp_diameter * (1 + dist_compensate(comp_deg)) )
                stopped = stopped | (comp_edge_ratio < 0.5)
                if (next_frame_id - frame_id)*track_dir == self.min_move_frame:
                    #stopped = stopped | ((comp_centers[:, next_frame_id] - comp_centers[:, frame_id]).norm(p=2, dim=-1) < 0.08 * comp_diameter) # 0.07 too large
                    moving = moving & ((comp_centers[:, next_frame_id] - comp_centers[:, frame_id]).norm(p=2, dim=-1) > 0.08 * comp_diameter)
                if last_velo is not None:
                    dev_velo_norm = (comp_velo - last_velo).norm(p=2, dim=-1)

                    if self.visualize:
                        ps_frame.add_scalar_quantity('dev_velo_norm', dev_velo_norm[frame.component].detach().cpu())

                    stopped = stopped | (dev_velo_norm > 0.24 * comp_diameter) # (0.08, 0.12)

                    norm = (comp_velo.norm(p=2, dim=-1) * comp_velos[:, next_frame_id-track_dir].norm(p=2, dim=-1)).clamp(min=1e-6)
                    comp_velo_angle = ((comp_velo * comp_velos[:, next_frame_id-track_dir]).sum(-1) / norm).clamp(min=-1, max=1).arccos() / np.pi * 180.0
                    stopped = stopped | (comp_velo_angle > self.angle_threshold) & (comp_velos[:, next_frame_id, :2].norm(p=2, dim=-1) > 0.01)

                    if self.visualize:
                        ps_frame.add_scalar_quantity('comp_velo_angle', comp_velo_angle[frame.component].detach().cpu())

                if self.visualize:
                    ps_frame.add_scalar_quantity('stopped', stopped[frame.component].detach().cpu(), enabled=True)
                    moved_dist = (comp_centers[:, next_frame_id] - comp_centers[:, frame_id]).norm(p=2, dim=-1)
                    ps_frame.add_scalar_quantity('moved dist', moved_dist[frame.component].detach().cpu(), enabled=False)

                last_velo = comp_velo
                if next_frame_id == frame_id - 1:
                    comp_velos[:, frame_id] = comp_velo

                if track_dir == -1:
                    comp_min_frame_id[~stopped] = next_frame_id
                else:
                    comp_max_frame_id[~stopped] = next_frame_id

                # extract point from this reference frame
                # record the reference frame id and the original tracking component
                frame.fxyz[:, 0] = next_frame_id
                f_this, f_next, _ = self.nn_graph(frame, next_frame)
                valid_mask = (~stopped)[frame.component[f_this]]
                f_this, f_next = f_this[valid_mask], f_next[valid_mask]
                extracted_fxyz.append(next_frame.fxyz[f_next])
                extracted_component.append(frame.component[f_this])
                extracted_segmentation_label.append(next_frame.segmentation_label[f_next])
                extracted_indices.append(f_next.reshape(-1))
                extracted_original_indices.append(next_frame_mask.nonzero()[f_next].reshape(-1))
                frame.fxyz[:, 0] = frame_id

                # update transformation based on track direction 
                reg_errors[:, next_frame_id] = l1_reg_error
                comp_edge_ratios[:, next_frame_id] = comp_edge_ratio

                next_frame_id += track_dir
                #if track_dir == 1:
                #    transforms.append(T @ transforms[-1])
                #    reg_errors.append(l1_reg_error)
                #else:
                #    transforms = [T @ transforms[0]] + transforms
                #    reg_errors = [l1_reg_error] + reg_errors
                
            frame.fxyz = seq_points.fxyz[frame_mask]

        extracted_fxyz = torch.cat(extracted_fxyz, dim=0)
        extracted_component = torch.cat(extracted_component, dim=0)
        extracted_segmentation_label = torch.cat(extracted_segmentation_label, dim=0)
        extracted_indices = torch.cat(extracted_indices, dim=0)
        extracted_original_indices = torch.cat(extracted_original_indices, dim=0)
        extracted = EasyDict(dict(
                        fxyz=extracted_fxyz,
                        component=extracted_component,
                        segmentation_label=extracted_segmentation_label,
                        frame_indices=extracted_indices,
                        original_indices=extracted_original_indices,
                        moving=moving[extracted_component],
                        valid_comp_mask=moving[extracted_component],
                        gt_box_label=torch.zeros_like(extracted_component),
                    ))

        valid_comp_mask = valid_comp_mask & ((comp_max_frame_id >= frame_id + self.min_move_frame) | (comp_min_frame_id <= frame_id - self.min_move_frame))
        #seq_points.extracted = seq_points.extracted & valid_comp_mask[extracted.component]
        valid_point_mask = valid_comp_mask[extracted.component]
        extracted = EasyDict(common_utils.filter_dict(extracted, valid_point_mask))
        extracted.transforms = transforms
        seq_points.extracted[extracted.original_indices] = True
        
        if self.visualize:
            num_components = extracted.component.max().item()+1
            colors = common_utils.separate_colors(num_components)
            _extracted = EasyDict(dict(
                             type='point_cloud',
                             name='extracted',
                             xyz=extracted.fxyz[:, 1:].detach().cpu(),
                             radius=0.06,
                             kwargs=dict(enabled=False),
                             scalars=EasyDict({
                                         'scalars/frame': dict(values=extracted.fxyz[:, 0].detach().cpu()),
                                         'scalars/component': dict(values=extracted.component.detach().cpu()),
                                         'scalars/moving': dict(values=extracted.moving.detach().cpu()),
                                     }),
                             colors=EasyDict({
                                        'labels/component': dict(values=random_color(extracted.component.detach().cpu())),
                                    }),
                             vectors=EasyDict(dict()),
                         ))
            
            self.visualizer.register_point_cloud(_extracted)
            #ps_ex = ps.register_point_cloud('extracted', extracted.fxyz[:, 1:].detach().cpu().numpy(), radius=2e-4, enabled=False)
            #ps_ex.add_scalar_quantity('scalar/frame', extracted.fxyz[:, 0].detach().cpu().numpy())
            #ps_ex.add_scalar_quantity('scalar/component', extracted.component.detach().cpu().numpy())
            #ps_ex.add_scalar_quantity('scalar/moving', extracted.moving.detach().cpu().numpy())
            #ps_ex.add_color_quantity('labels/component', colors[extracted.component.detach().cpu().numpy()])

        return extracted

    def forward(self, seq_dict):
        """Given proposed candidate clusters, track each of them.
        
        Args:
            point_component (N): proposed candidate object clusters.
              range in [0, C]
        """
        seq_points = EasyDict(
            fxyz=seq_dict['point_fxyz'],
            frame=seq_dict['point_sweep'],
            #component=seq_dict['point_component'],
            gt_box_id=seq_dict['point_gt_box_id'],
        )
        for key in ['instance_label', 'segmentation_label']:
            if key in seq_dict:
                seq_points[key] = seq_dict[key]
        
        all_points = EasyDict(
            fxyz=seq_dict['full_point_fxyz'],
            frame=seq_dict['full_point_sweep'],
            height=seq_dict['full_point_height'],
        )
        for key in ['full_instance_label', 'full_segmentation_label']:
            if key in seq_dict:
                all_points[key] = seq_dict[key]
        all_points = EasyDict(common_utils.filter_dict(
                                all_points,
                                seq_dict['full_point_height'] > 0.0))

        num_frames = seq_points.frame.max().long().item() + 1
        num_points = seq_points.frame.shape[0]
        device = seq_points.fxyz.device
        sequence_id = seq_dict['frame_id'][0][:-4]
        
        outfolder = f'{self.model_cfg.DIR}/{sequence_id}'
        outpath = f'{outfolder}/all.pth'
        if os.path.exists(outpath):
            print(f'{outpath} already exists. skipping...')
            return seq_dict

        os.makedirs(outfolder, exist_ok=True)

        # extract and format GT boxes.
        # assign placeholder for the best coverage IoU of each box
        seq_boxes = self.format_boxes(seq_dict, num_frames)
        num_boxes = seq_boxes.attr.shape[0]
        if num_boxes == 0:
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

        for comp_key in self.component_keys:
            print(f'Component Key = {comp_key}')
            torch.cuda.empty_cache()

            seq_points.component = seq_dict[f'point_{comp_key}']

            # set components to be stationary
            seq_points.component_diameter = component_diameter(seq_points)[seq_points.component]
            seq_points.stationary = (seq_points.component_diameter > 12.5)
            seq_points.extracted = torch.zeros_like(seq_points.fxyz[:, 0]).bool()
            
            for frame_id in range(0, num_frames, self.track_interval):
                # extract a point cloud frame
                frame_mask = (seq_points.fxyz[:, 0] == frame_id).reshape(-1)
                if not frame_mask.any():
                    continue
                frame_points = EasyDict(common_utils.filter_dict(seq_points, frame_mask))

                # shift the min of connected component id to zero.
                frame_points.component = frame_points.component - frame_points.component.min()

                # track this frame along the sequence
                ex_path = f'{outfolder}/{frame_id:03d}_{comp_key}.pth'
                with Timer(f'Tracking Frame {frame_id}'):
                    extracted = self.track_frame(seq_points, frame_points, seq_boxes)

                with Timer(f'Extract Traces from all points'):
                    if extracted.fxyz.shape[0] > 0:
                        extracted, seq_boxes = self.extract_traces_and_update_boxes(all_points, extracted, seq_boxes)

                print(f'saving extracted data to {ex_path}')
                torch.save(extracted, ex_path)
                extracted.pop('transforms')

                if self.visualize:
                    self.visualize_frame(seq_points, frame_points, extracted)

                segment_box_mask = ((seq_boxes.frame >= frame_id - self.track_interval) & (seq_boxes.frame <= frame_id + self.track_interval)).reshape(-1)
                segment_boxes = EasyDict(common_utils.filter_dict(seq_boxes, segment_box_mask))
                segment_coverage = (segment_boxes.best_iou > 0.7).float().mean()
                print(f'segment [{frame_id - self.track_interval}, {frame_id + self.track_interval}]: num_boxes={segment_boxes.attr.shape[0]}, coverage={segment_coverage:.6f}')

                if self.visualize:
                    import polyscope as ps;
                    ps_box = ps.get_volume_mesh('segment_boxes')
                    ps_box.add_scalar_quantity('best_iou', segment_boxes.best_iou.detach().cpu(), defined_on='cells')
                    ps.show()
            
            #box_path = f'{outfolder}/all_{comp_key}.pth'
            #torch.save(seq_boxes, box_path)

        if seq_boxes.moving.any():
            moving_mean = seq_boxes.best_iou[seq_boxes.moving]
        else:
            moving_mean = 'NA'

        print(f'All Box mIoU={seq_boxes.best_iou.mean()}')
        print(f'Moving Box mIoU={moving_mean}')
        os.makedirs(outfolder, exist_ok=True)
        print(f'saving extracted data to {outpath}')
        torch.save(seq_boxes, outpath)

        return seq_dict

    def get_output_feature_dim(self):
        return 0

    def extra_repr(self):
        return f"reg_error_coeff={self.reg_error_coeff}, min_move_frame={self.min_move_frame}, angle_threshold={self.angle_threshold}, track_interval={self.track_interval}"
