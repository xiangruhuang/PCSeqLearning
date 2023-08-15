import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, runtime_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg)
        self.num_point_features = runtime_cfg.get("num_point_features")

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        
        voxel_features, voxel_num_points = batch_dict['voxel_point_feat'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()


        #voxel_features, voxel_num_points = batch_dict['voxel_points'], batch_dict['voxel_num_points'].long()
        #voxels_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        #normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        #voxels_mean = voxels_mean / normalizer
        ##voxels_mean = torch.cat([batch_dict['voxel_coords'][:, 0:1], voxels_mean], dim=-1) # [N, 1+C]
        #batch_dict['voxel_features'] = voxels_mean.contiguous()
        #if 'voxel_seg_cls_labels' in batch_dict:
        #    indices0 = torch.arange(voxel_features.shape[0]
        #                           ).unsqueeze(-1).to(voxel_num_points) # [N, 1]
        #    indices1 = torch.arange(voxel_features.shape[1]
        #                           ).unsqueeze(0).to(voxel_num_points) # [1, C]
        #    indices1 = indices1 % voxel_num_points.unsqueeze(-1) # [1, C] % [N, 1] = [N, C]
        #    #voxel_seg_inst_labels = batch_dict['voxel_point_seg_inst_labels'] # [N, C]
        #    voxel_seg_cls_labels = batch_dict['voxel_seg_cls_labels'] # [N, C]
        #    voxel_seg_cls_labels_median = voxel_seg_cls_labels[(indices0, indices1)].median(-1)[0]
        #    batch_dict['voxel_seg_cls_labels'] = voxel_seg_cls_labels_median
        #    #voxel_seg_inst_labels_median = voxel_seg_inst_labels[(indices0, indices1)].median(-1)[0]
        #    #`batch_dict['voxel_seg_inst_labels'] = voxel_seg_inst_labels_median

        #batch_dict['batch_idx'] = batch_dict['voxel_coords'][:, 0].long()
        return batch_dict
