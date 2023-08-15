import torch
from torch import nn
import numpy as np
from collections import defaultdict
import polyscope as ps
from easydict import EasyDict
import os

def new_geometry(geometry_type):
    return EasyDict(dict(
                      type=geometry_type,
                      scalars=EasyDict(dict()),
                      vectors=EasyDict(dict()),
                      colors=EasyDict(dict()),
                      kwargs=EasyDict(dict()),
                    ))

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def filter_dict(data_dict, mask, ignore_keys = []):
    ret_data_dict = {}
    for key in data_dict.keys():
        if key in ignore_keys:
            ret_data_dict[key] = data_dict[key]
            continue
        if isinstance(mask, torch.Tensor) and (mask.dtype == torch.bool):
            assert mask.shape[0] == len(data_dict[key]), f"MisMatch for key={key}, mask.shape={mask.shape}, data.shape={len(data_dict[key])}"
        if isinstance(mask, np.ndarray) and (mask.dtype == bool):
            assert mask.shape[0] == len(data_dict[key]), f"MisMatch for key={key}, mask.shape={mask.shape}, data.shape={len(data_dict[key])}"
        ret_data_dict[key] = data_dict[key][mask]
    return ret_data_dict

def to_numpy_cpu(a, compress=False):
    if isinstance(a, torch.Tensor):
        data = a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        data = a
    else:
        raise ValueError("Requiring Numpy or torch.Tensor")

    if data.dtype == np.int64:
        data = data.astype(np.int16)
    if data.dtype == np.int32:
        data = data.astype(np.int16)
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if data.dtype == np.float32:
        data = data.astype(np.float32)

    return data

class GeometryVisualizer(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = model_cfg.get('ENABLED', False)
        self.save_path = model_cfg.get("SAVE_PATH", None)
        self.save_dir = model_cfg.get("SAVE_DIR", None)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.enabled:
            self.up_dir = model_cfg.get("UP_DIR", "z_up")
            self.point_cloud_vis = model_cfg.get("POINT_CLOUD", None)
            self.point_cloud_sequence_vis = model_cfg.get("POINT_CLOUD_SEQUENCE", None)
            self.box_vis = model_cfg.get("BOX", None)
            self.box_sequence_vis = model_cfg.get("BOX_SEQUENCE", None)
            self.lidar_origin_vis = model_cfg.get("LIDAR_ORIGIN", None)
            self.graph_vis = model_cfg.get("GRAPH", None)
            self.primitive_vis = model_cfg.get("PRIMITIVE", None)
            self.shared_color_dict = model_cfg.get("SHARED_COLOR", None)
            self.output = model_cfg.get("OUTPUT", None)
            self.voxel_size = model_cfg.get('voxel_size', None)
            self.pc_range = model_cfg.get('pc_range', None)
            self.size_factor = model_cfg.get('size_factor', None)
            self.radius = model_cfg.get('radius', 0.03)
            self.ground_plane = model_cfg.get("ground_plane", False)
            self.init()
    
    def color(self, color_name):
        if not hasattr(self, "_shared_color"):
            raise ValueError("Color Dictionary not initialized")
        return self._shared_color[color_name]

    def init(self):
        ps.set_up_dir(self.up_dir)
        ps.init()
        if not self.ground_plane:
            ps.set_ground_plane_mode('none')
        if self.shared_color_dict is not None:
            color_dict = {}
            for color_name, color in self.shared_color_dict.items():
                if isinstance(color, list) and len(color) == 1:
                    color_dict[color_name] = np.random.uniform(size=color)
                else:
                    color_dict[color_name] = np.array(color)
            self._shared_color = color_dict

    def process_scalars(self, geometry, scalars, batch_dict):
        for scalar_name, scalar_cfg in scalars.items():
            if scalar_name not in batch_dict:
                continue
            scalar = to_numpy_cpu(batch_dict[scalar_name]).reshape(-1)
            geometry.scalars['scalars/'+scalar_name] = dict(
                                                         name='scalars/'+scalar_name,
                                                         values=scalar,
                                                         **scalar_cfg,
                                                       )
        return geometry
    
    def process_class_labels(self, geometry, class_labels, batch_dict, batch_mask=None):
        for label_name, label_cfg in class_labels.items():
            if label_name not in batch_dict:
                continue

            if batch_mask is not None:
                label = to_numpy_cpu(batch_dict[label_name][batch_mask]).astype(np.int32)
            else:
                label = to_numpy_cpu(batch_dict[label_name]).astype(np.int32)

            if label.shape[0] == 0:
                continue
            cfg = EasyDict(dict(name='class_labels/'+label_name))
            for key, val in label_cfg.items():
                if (key == 'values') and isinstance(val, str):
                    color_this = self.color(val)[label]
                    invalid_mask = label < 0
                    color_this[invalid_mask] = np.array([75./255, 75./255, 75/255.])
                    cfg['values'] = to_numpy_cpu(color_this)
                else:
                    cfg[key] = val
            if cfg.get('values', None) is None:
                num_color = label.max() + 1
                ndim = int(num_color ** (1/3.0)) + 1
                u = np.linspace(0, 1, ndim)
                x, y, z = np.meshgrid(u, u, u)
                random_color = np.stack([x, y, z], axis=-1).reshape(-1, 3)
                color_this = random_color[label]
                cfg['values'] = to_numpy_cpu(color_this)
            geometry.colors['class_labels/'+label_name] = cfg
        return geometry

    def process_point_cloud_sequence(self, pc_key, vis_cfg, batch_dict):
        point_cloud = new_geometry('point_cloud')

        pointcloud = batch_dict[pc_key]
        point_cloud.frame = to_numpy_cpu(pointcloud[:, 0])
        point_cloud.xyz = to_numpy_cpu(pointcloud[:, 1:])
        point_cloud.name = pc_key
        batch_mask = torch.ones_like(pointcloud[:, 0]).bool()

        for key, val in vis_cfg.items():
            if key in ['color', 'radius', 'enabled', 'name']:
                point_cloud.kwargs[key] = val
            if key == 'scalars':
                point_cloud = self.process_scalars(point_cloud, val, batch_dict)
            if key == 'class_labels':
                point_cloud = self.process_class_labels(point_cloud, val, batch_dict, batch_mask)

        return point_cloud

    def process_point_cloud(self, pc_key, vis_cfg, batch_dict, batch_id):
        point_cloud = new_geometry('point_cloud')

        batch_mask = (batch_dict[pc_key][:, 0] == batch_id).reshape(-1)
        point_cloud.xyz = to_numpy_cpu(batch_dict[pc_key][batch_mask, 1:])
        point_cloud.name = pc_key

        for key, val in vis_cfg.items():
            if key in ['color', 'radius', 'enabled', 'name']:
                point_cloud.kwargs[key] = to_numpy_cpu(val)
            if key == 'scalars':
                point_cloud = self.process_scalars(point_cloud, val, batch_dict)
            if key == 'class_labels':
                point_cloud = self.process_class_labels(point_cloud, val, batch_dict, batch_mask)

        return point_cloud

    def visualize(self, monitor=None):
        if monitor is None:
            return
        if monitor == 'screen':
            self.show()
        elif monitor == 'pause':
            print('pausing')
            pause()
        elif isinstance(monitor, str):
            self.save(monitor)
        else:
            raise ValueError(f"Unrecognized Monitor Option {monitor}")

    def register_point_cloud(self, points):
        """
            points (dictionary):
                points.type: 'point_cloud'
                points.name: string
                points.xyz [N, 3]: point locations
                points.kwargs (dictionary)
                
                additional attributes:
                    points.scalars (dictionary): key -> kwargs
                    points.colors (dictionary): key -> kwargs
                    points.vectors (dictionary): key -> kwargs
        """
        import polyscope as ps
        ps_p = ps.register_point_cloud(points.name, to_numpy_cpu(points.xyz), **points.kwargs)
        if points.radius is not None:
            ps_p.set_radius(points.radius, relative=False)
        for scalar_name, scalar_kwargs in points.scalars.items():
            ps_p.add_scalar_quantity(name=scalar_name, **scalar_kwargs)
        for color_name, color_kwargs in points.colors.items():
            ps_p.add_color_quantity(name=color_name, **color_kwargs)
        for vector_name, vector_kwargs in points.vectors.items():
            ps_p.add_vector_quantity(name=vector_name, **vector_kwargs)

        return ps_p

    def register_boxes(self, boxes):
        """
            boxes (dictionary):
                boxes.type: 'boxes'
                boxes.name: string
                boxes.attr [B, 7]: box attributes, (x, y, z, dimx, dimy, dimz, heading)
                boxes.cls_label: [B]
                boxes.kwargs (dictionary)

                additional attributes:
                    boxes.scalars (dictionary): (key, Attribute) pairs
                    boxes.colors (dictionary): (key, Attribute) pairs
        """
        import polyscope as ps
        corners = boxes_to_corners_3d(boxes.attr)
        #ps_box = self.boxes_from_attr(name, corners, data_dict, batch_mask, data_mask, labels, **kwargs)
        corners = to_numpy_cpu(corners)
        corners = corners.reshape(-1, 3)
        ps_box = ps.register_volume_mesh(
                    boxes.name, corners,
                    hexes=np.arange(corners.shape[0]).reshape(-1, 8),
                    **boxes.kwargs
                 )
        ps_box.set_transparency(0.2)
        labels = to_numpy_cpu(boxes.cls_label).astype(np.int64)
        colors = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [1,1,0], [1,1,1], [0, 0, 0], [0.5, 0.5, 0.5]])
        ps_box.add_color_quantity('class', colors[labels], defined_on='cells', enabled=True)
        ps_box.add_scalar_quantity('scalars/class', labels, defined_on='cells')

        for scalar_name, scalar_kwargs in boxes.scalars.items():
            ps_box.add_scalar_quantity(scalar_name, **scalar_kwargs)
        for color_name, color_kwargs in boxes.colors.items():
            ps_box.add_color_quantity(color_name, **color_kwargs)
        for vector_name, vector_kwargs in boxes.vectors.items():
            ps_box.add_vector_quantity(vector_name, **vector_kwargs)

        return ps_box

    def register_geometries(self, geometries):
        for geometry in geometries:
            geometry_type = geometry.type
            getattr(self, f'register_{geometry_type}')(geometry)

    def forward(self, batch_dict):
        for i in range(batch_dict.get('batch_size', 1)):
            geometries = []
            name = batch_dict['frame_id'].reshape(-1)[i][:-4]
            if self.point_cloud_vis is not None:
                for pc_key, vis_cfg in self.point_cloud_vis.items():
                    if pc_key not in batch_dict:
                        continue
                    point_cloud = self.process_point_cloud(pc_key, vis_cfg, batch_dict, i)
                    geometries.append(point_cloud)
            
            if self.point_cloud_sequence_vis is not None:
                for pc_key, vis_cfg in self.point_cloud_sequence_vis.items():
                    if pc_key not in batch_dict:
                        continue
                    point_cloud_sequence = self.process_point_cloud_sequence(pc_key, vis_cfg, batch_dict)
                    geometries.append(point_cloud_sequence)
            
            geometries_np = np.array(geometries, dtype=object)
            if self.save_dir is not None:
                np.save(f'{self.save_dir}/{name}.npy', geometries_np)
            #if self.box_vis is not None:
            #    for box_key, vis_cfg_this in self.box_vis.items():
            #        if box_key not in batch_dict:
            #            continue
            #        vis_cfg = {}; vis_cfg.update(vis_cfg_this)
            #        boxes = to_numpy_cpu(batch_dict[box_key][i])
            #        mask = (boxes[:, 3:6] ** 2).sum(axis=-1) > 1e-1
            #        boxes = boxes[mask]
            #        if boxes.shape[1] > 7:
            #            labels = boxes[:, 7]
            #        else:
            #            labels = np.zeros(boxes.shape[0]).astype(np.int32)
            #        boxes = boxes[:, :7]
            #        if 'name' in vis_cfg:
            #            box_name = vis_cfg.pop('name')
            #        else:
            #            box_name = box_key
            #        self.boxes_from_attr(box_name, boxes, batch_dict, i, mask, labels, **vis_cfg)
            #
            #if self.box_sequence_vis is not None:
            #    for box_key, vis_cfg_this in self.box_sequence_vis.items():
            #        if box_key not in batch_dict:
            #            continue
            #        vis_cfg = {}; vis_cfg.update(vis_cfg_this)
            #        boxes = to_numpy_cpu(batch_dict[box_key])
            #        boxes = boxes.reshape(-1, boxes.shape[-1])
            #        mask = (boxes[:, 3:6] ** 2).sum(axis=-1) > 1e-1
            #        boxes = boxes[mask]
            #        if boxes.shape[1] > 7:
            #            labels = boxes[:, 7]
            #        else:
            #            labels = np.zeros(boxes.shape[0]).astype(np.int32)
            #        boxes = boxes[:, :7]
            #        if 'name' in vis_cfg:
            #            box_name = vis_cfg.pop('name')
            #        else:
            #            box_name = box_key
            #        self.boxes_from_attr(box_name, boxes, batch_dict, np.arange(boxes.shape[0]), mask, labels, **vis_cfg)
            #
            #if self.graph_vis is not None:
            #    for graph_key, vis_cfg_this in self.graph_vis.items():
            #        if graph_key not in batch_dict:
            #            continue
            #        vis_cfg = {}; vis_cfg.update(vis_cfg_this)
            #        e_ref, e_query = to_numpy_cpu(batch_dict[graph_key])
            #        query_key = vis_cfg.pop('query')
            #        query_points = to_numpy_cpu(batch_dict[query_key])
            #        ref_key = vis_cfg.pop('ref')
            #        ref_points = to_numpy_cpu(batch_dict[ref_key])
            #        scalars = vis_cfg.pop('scalars') if 'scalars' in vis_cfg else None

            #        try:
            #            valid_mask = (query_points[e_query, 0].round().astype(np.int32) == i) & (ref_points[e_ref, 0].round().astype(np.int32) == i)
            #        except Exception as e:
            #            print(graph_key)
            #            print(e)
            #        e_query, e_ref = e_query[valid_mask], e_ref[valid_mask]

            #        # take this batch
            #        query_batch_idx = np.where(query_points[:, 0].round().astype(np.int32) == i)[0]
            #        query_idx_map = np.zeros(query_points.shape[0]).round().astype(np.int32)
            #        query_idx_map[query_batch_idx] = np.arange(query_batch_idx.shape[0])
            #        query_points = to_numpy_cpu(query_points[query_batch_idx, 1:])
            #        e_query = query_idx_map[e_query]

            #        ref_batch_idx = np.where(ref_points[:, 0].round().astype(np.int32) == i)[0]
            #        ref_idx_map = np.zeros(ref_points.shape[0]).round().astype(np.int32)
            #        ref_idx_map[ref_batch_idx] = np.arange(ref_batch_idx.shape[0])
            #        ref_points = to_numpy_cpu(ref_points[ref_batch_idx, 1:])
            #        e_ref = ref_idx_map[e_ref]
            #    
            #        edge_indices = to_numpy_cpu(np.stack([e_query, e_ref+query_points.shape[0]], axis=-1))
            #        
            #        if 'name' in vis_cfg:
            #            graph_name = vis_cfg.pop('name')
            #        else:
            #            graph_name = graph_key
            #        all_points = np.concatenate([query_points[:, :3], ref_points[:, :3]], axis=0)
            #        ps_c = self.curvenetwork(graph_name, all_points, edge_indices, batch_dict, valid_mask, **vis_cfg)
            #        if scalars:
            #            for scalar_name, scalar_cfg in scalars.items():
            #                if scalar_name not in batch_dict:
            #                    continue
            #                try:
            #                    scalar = to_numpy_cpu(batch_dict[scalar_name][valid_mask])
            #                except Exception as e:
            #                    print(e)
            #                    print(scalar_name)
            #                    print(f"""Error in attaching {scalar_name} to graph {graph_name}, \
            #                              expect shape={valid_mask.shape[0]}, actual shape={batch_dict[scalar_name].shape}""")
            #                    assert False
            #                ps_c.add_scalar_quantity('scalars/'+scalar_name, scalar.reshape(-1), defined_on='edges', **scalar_cfg)

            #if self.primitive_vis is not None:
            #    for primitive_key, vis_cfg in self.primitive_vis.items():
            #        if primitive_key + '_bxyz' not in batch_dict:
            #            continue
            #            
            #        vis_cfg_this = {}; vis_cfg_this.update(vis_cfg)
            #        primitives = EasyDict(dict(
            #            eigvals = to_numpy_cpu(batch_dict[f'{primitive_key}_eigvals'].clamp(min=1e-4)),
            #            eigvecs = to_numpy_cpu(batch_dict[f'{primitive_key}_eigvecs']),
            #            bxyz = to_numpy_cpu(batch_dict[f'{primitive_key}_bxyz']),
            #            l1_proj_max = to_numpy_cpu(batch_dict[f'{primitive_key}_l1_proj_max']),
            #            l1_proj_min = to_numpy_cpu(batch_dict[f'{primitive_key}_l1_proj_min']),
            #        ))
            #        primitives.xyz = primitives.bxyz[:, 1:]
            #        primitives.batch_index = primitives.bxyz[:, 0].round().astype(np.int32)
            #        batch_mask = primitives.batch_index == i
            #        primitives = EasyDict(filter_dict(primitives, batch_mask))

            #        
            #        corners = []

            #        eps = vis_cfg_this.pop("epsilon") if "epsilon" in vis_cfg_this else 2e-5
            #        for dx in [primitives.l1_proj_min[:, 0:1]-eps, primitives.l1_proj_max[:, 0:1]+eps]:
            #            for dy, dz in [
            #                    (primitives.l1_proj_min[:, 1:2]-eps, primitives.l1_proj_min[:, 2:3]-eps),
            #                    (primitives.l1_proj_min[:, 1:2]-eps, primitives.l1_proj_max[:, 2:3]+eps),
            #                    (primitives.l1_proj_max[:, 1:2]+eps, primitives.l1_proj_max[:, 2:3]+eps),
            #                    (primitives.l1_proj_max[:, 1:2]+eps, primitives.l1_proj_min[:, 2:3]-eps),
            #                ]:
            #                dvec  = dx * primitives.eigvecs[:, :, 0]
            #                dvec += dy * primitives.eigvecs[:, :, 1]
            #                dvec += dz * primitives.eigvecs[:, :, 2]
            #                corner = primitives.xyz + dvec
            #                corners.append(corner)
            #        corners = np.stack(corners, axis=1)
            #        hexes = np.arange(corners.shape[0]*8).reshape(-1, 8)
            #        scalars = vis_cfg_this.pop("scalars") if "scalars" in vis_cfg else None
            #        class_labels = vis_cfg_this.pop("class_labels") if "class_labels" in vis_cfg_this else None
            #        ps_v = ps.register_volume_mesh(primitive_key, to_numpy_cpu(corners.reshape(-1, 3)), hexes=hexes, **vis_cfg_this)
            #        if scalars:
            #            for scalar_name, scalar_cfg in scalars.items():
            #                ps_v.add_scalar_quantity('scalars/'+scalar_name, to_numpy_cpu(batch_dict[scalar_name][batch_mask].view(-1)), defined_on='cells', **scalar_cfg)
            #        if class_labels:
            #            for label_name, label_cfg in class_labels.items():
            #                label = to_numpy_cpu(batch_dict[label_name][batch_mask]).astype(np.int32)
            #                label_cfg_this = {}
            #                for key, val in label_cfg.items():
            #                    if (key == 'values') and isinstance(val, str):
            #                        label_cfg_this[key] = self.color(val)[label]
            #                        invalid_mask = label < 0
            #                        label_cfg_this[key][invalid_mask] = np.array([75./255, 75./255, 75/255.])
            #                    else:
            #                        label_cfg_this[key] = val
            #                ps_v.add_color_quantity('class_labels/'+label_name, defined_on='cells', **label_cfg_this)
                        
            #self.visualize(monitor=self.output)
        if self.save_path is not None:
            torch.save(geometries, self.save_path)

    def clear(self):
        ps.remove_all_structures()
        self.logs = []

    def pc_scalar(self, pc_name, name, quantity, enabled=False):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.get_point_cloud(pc_name).add_scalar_quantity(name, quantity, enabled=enabled)
    
    def pc_color(self, pc_name, name, color, enabled=False):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.get_point_cloud(pc_name).add_color_quantity(name, color, enabled=enabled)

    def corres(self, name, src, tgt):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        points = np.concatenate([src, tgt], axis=0)
        edges = np.stack([np.arange(src.shape[0]),
                          np.arange(tgt.shape[0]) + src.shape[0]], axis=-1)
        return ps.register_curve_network(name, points, edges, radius=self.radius)

    def trace(self, name, points, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        num_points = points.shape[0]
        edges = np.stack([np.arange(num_points-1),
                          np.arange(num_points-1)+1], axis=-1)
        return ps.register_curve_network(name, points, edges, **kwargs)
   
    def curvenetwork(self, name, nodes, edges, data_dict, batch_mask, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")

        edge_scalars = kwargs.pop("edge_scalars") if "edge_scalars" in kwargs else None
        radius = kwargs.pop('radius', self.radius)
        ps_c = ps.register_curve_network(name, nodes, edges, **kwargs)
        ps_c.set_radius(radius, relative=False)

        if edge_scalars:
            for scalar_name, scalar_cfg in edge_scalars.items():
                scalar = to_numpy_cpu(data_dict[scalar_name][batch_mask])
                ps_c.add_scalar_quantity('edge-scalars/'+scalar_name, scalar, defined_on='edges', **scalar_cfg)
        return ps_c


    def pointcloud(self, name, pointcloud, data_dict, batch_mask, color=None, radius=None, **kwargs):
        """Visualize non-zero entries of heat map on 3D point cloud.
            point cloud (torch.Tensor, [N, 3])
        """
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        if radius is None:
            radius = self.radius
        scalars = kwargs.pop("scalars") if "scalars" in kwargs else None
        class_labels = kwargs.pop("class_labels") if "class_labels" in kwargs else None

        if color is None:
            ps_p = ps.register_point_cloud(name, pointcloud, **kwargs)
        else:
            ps_p = ps.register_point_cloud(
                name, pointcloud, radius=radius, color=tuple(color), **kwargs
                )
        ps_p.set_radius(radius, relative=False)

        if scalars:
            for scalar_name, scalar_cfg in scalars.items():
                if scalar_name not in data_dict:
                    continue
                try:
                    scalar = to_numpy_cpu(data_dict[scalar_name][batch_mask])
                    ps_p.add_scalar_quantity('scalars/'+scalar_name, scalar.reshape(-1), **scalar_cfg)
                except Exception as e:
                    print(e)
                    print(scalar_name)
                    print(f"""Error in attaching {scalar_name} to point cloud {name}, \
                              expect shape={pointcloud.shape[0]}, actual shape={data_dict[scalar_name].shape}""")
                    assert False

        if class_labels:
            for label_name, label_cfg in class_labels.items():
                if label_name not in data_dict:
                    continue
                label = to_numpy_cpu(data_dict[label_name][batch_mask]).astype(np.int32)
                if label.shape[0] == 0:
                    continue
                label_cfg_this = {}
                for key, val in label_cfg.items():
                    if (key == 'values') and isinstance(val, str):
                        label_cfg_this[key] = self.color(val)[label]
                        invalid_mask = label < 0
                        label_cfg_this[key][invalid_mask] = np.array([75./255, 75./255, 75/255.])
                    else:
                        label_cfg_this[key] = val
                if label_cfg_this.get('values', None) is None:
                    num_color = label.max() + 1
                    ndim = int(num_color ** (1/3.0)) + 1
                    u = np.linspace(0, 1, ndim)
                    x, y, z = np.meshgrid(u, u, u)
                    random_color = np.stack([x, y, z], axis=-1).reshape(-1, 3)
                    label_cfg_this['values'] = random_color[label]
                ps_p.add_color_quantity('class_labels/'+label_name, **label_cfg_this)

        return ps_p
    
    def get_meshes(self, centers, eigvals, eigvecs):
        """ Prepare corners and faces (for visualization only). """

        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        v1 = eigvecs[:, :3]
        v2 = eigvecs[:, 3:]
        e1 = np.sqrt(eigvals[:, 0:1])
        e2 = np.sqrt(eigvals[:, 1:2])
        corners = []
        for d1 in [-1, 1]:
            for d2 in [-1, 1]:
                corners.append(centers + d1*v1*e1 + d2*v2*e2)
        num_voxels = centers.shape[0]
        corners = np.stack(corners, axis=1) # [M, 4, 3]
        faces = [0, 1, 3, 2]
        faces = np.array(faces, dtype=np.int32)
        faces = np.repeat(faces[np.newaxis, np.newaxis, ...], num_voxels, axis=0)
        faces += np.arange(num_voxels)[..., np.newaxis, np.newaxis]*4
        return corners.reshape(-1, 3), faces.reshape(-1, 4)
    
    def planes(self, name, planes):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        corners, faces = self.get_meshes(planes[:, :3], planes[:, 6:8], planes[:, 8:14])
        return ps.register_surface_mesh(name, corners, faces)

    def boxes_from_attr(self, name, attr, data_dict=None, batch_mask=None, data_mask=None, labels=None, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        corners = boxes_to_corners_3d(attr)
        if 'with_ori' in kwargs:
            with_ori = kwargs.pop('with_ori')
        else:
            with_ori = False
        ps_box = self.boxes(name, corners, data_dict, batch_mask, data_mask, labels, **kwargs)
        #if with_ori:
        #    ori = attr[:, -1]
        #    sint, cost = np.sin(ori), np.cos(ori)
        #    arrow = np.stack([sint, cost, np.zeros_like(cost)], axis=-1)[:, np.newaxis, :].repeat(8, 1)
        #    ps_box.add_vector_quantity('orientation', arrow.reshape(-1, 3), enabled=True)
        

    def boxes(self, name, corners, data_dict=None, batch_mask=None, data_mask=None, labels=None, **kwargs):
        """
            corners (shape=[N, 8, 3]):
            labels (shape=[N])
        """
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        #  0    1
        #     3    2
        #  |    |
        #  4    5
        #     7    6
        #edges = [[0, 1], [0, 3], [0, 4], [1, 2],
        #         [1, 5], [2, 3], [2, 6], [3, 7],
        #         [4, 5], [4, 7], [5, 6], [6, 7]]
        N = corners.shape[0]
        #edges = np.array(edges) # [12, 2]
        #edges = np.repeat(edges[np.newaxis, ...], N, axis=0) # [N, 12, 2]
        #offset = np.arange(N)[..., np.newaxis, np.newaxis]*8 # [N, 1, 1]
        #edges = edges + offset
        #if kwargs.get('radius', None) is None:
        #    kwargs['radius'] = 2e-4
        scalars = kwargs.pop("scalars") if "scalars" in kwargs else None
        class_labels = kwargs.pop("class_labels") if "class_labels" in kwargs else None

        corners = to_numpy_cpu(corners)
        corners = corners.reshape(-1, 3)
        ps_box = ps.register_volume_mesh(
                    name, corners,
                    hexes=np.arange(corners.shape[0]).reshape(-1, 8),
                    **kwargs
                 )
        ps_box.set_transparency(0.2)
        if scalars:
            for scalar_name, scalar_cfg in scalars.items():
                if scalar_name not in data_dict:
                    continue
                scalar = to_numpy_cpu(data_dict[scalar_name][batch_mask][data_mask]).reshape(-1)
                ps_box.add_scalar_quantity('scalars/'+scalar_name, scalar, defined_on='cells', **scalar_cfg)

        if labels is not None:
            # R->Car, G->Ped, B->Cyc
            colors = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [1,1,0]])
            labels = to_numpy_cpu(labels).astype(np.int64) 
            #labels = np.repeat(labels[:, np.newaxis], 8, axis=-1).reshape(-1).astype(np.int64)
            ps_box.add_color_quantity('class', colors[labels], defined_on='cells', enabled=True)
            ps_box.add_scalar_quantity('scalars/class', labels, defined_on='cells')
        if class_labels is not None:
            for key, cfg in class_labels.items():
                label = to_numpy_cpu(data_dict[key]).reshape(-1)[data_mask]
                colors = np.random.randn(label.max()+1, 3)
                try:
                    ps_box.add_color_quantity(key, colors[label], defined_on='cells', **cfg)
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                    print(e)

        return ps_box

    def wireframe(self, name, heatmap):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        size_y, size_x = heatmap.shape
        x, y = np.meshgrid(heatmap)
        return x, y

    def heatmap(self, name, heatmap, color=True, threshold=0.1,
                **kwargs):
        """Visualize non-zero entries of heat map on 3D point cloud.
        `voxel_size`, `size_factor`, `pc_range` need to be specified.
        By default, the heatmap need to be transposed.

        Args:
            heatmap (torch.Tensor or np.ndarray, [W, H])

        """
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)

        if self.voxel_size is None:
            raise ValueError("self.voxel_size not specified")
        
        heatmap = heatmap.T
        size_x, size_y = heatmap.shape
        x, y = torch.meshgrid(torch.arange(size_x),
                              torch.arange(size_y),
                              indexing="ij")
        x, y = x.reshape(-1), y.reshape(-1)
        z = heatmap.reshape(-1)

        mask = torch.zeros(size_x+2, size_y+2, size_x+2, size_y+2, dtype=torch.bool)
        
        for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            mask[x+1, y+1, x+1+dx, y+1+dy] = True
        x0, y0, x1, y1 = torch.where(mask)
        x0, y0, x1, y1 = x0-1, y0-1, x1-1, y1-1
        is_inside = ((x1 >= size_x) | (x1 < 0) | (y1 >= size_y) | (y1 < 0)) == False
        e0 = (x0 * size_y + y0)[is_inside]
        e1 = (x1 * size_y + y1)[is_inside]
        
        edges = torch.stack([e0, e1], dim=-1)
        x = x * self.size_factor * self.voxel_size[0] + self.pc_range[0]
        y = y * self.size_factor * self.voxel_size[1] + self.pc_range[1]
        nodes = torch.stack([x, y, z], dim=-1)
        radius = kwargs.get("radius", self.radius*10)
        ps_c = self.curvenetwork(name, nodes, edges, radius=radius)
        
        if color:
            ps_c.add_scalar_quantity("height", z, enabled=True) 

        return ps_c

    def show(self):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.set_up_dir(self.up_dir)
        ps.init()
        ps.show()

    def look_at(self, center, distance=100, bev=True, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        if bev:
            camera_loc = center + np.array([0, 0, distance])
            # look down from bird eye view
            # with +y-axis being the up dir on the image
            ps.look_at_dir(camera_loc, center, (0,1,0), **kwargs)
        else:
            raise ValueError("Not Implemented Yet, please use bev=True")

    def screenshot(self, filename, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.screenshot(filename, **kwargs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    args = parser.parse_args()
    
    geometries = torch.load(args.data)

    gvis = GeometryVisualizer({'enabled': True, 'UP_DIR': 'z_up'})
    gvis.register_geometries(geometries)
    ps.show()
