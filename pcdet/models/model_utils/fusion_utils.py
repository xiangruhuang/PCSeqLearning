import torch
from torch import nn
from torch_scatter import scatter
from pcdet.utils.pca_utils import pca_by_group

class FusionTemplate(nn.Module):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, ref_bxyz, ref_feat, group_ids):
        raise NotImplementedError
        
class IFusion(FusionTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(runtime_cfg, model_cfg)

    def forward(self, ref_bxyz, ref_feat, group_ids):
        return ref_feat

class LineFusion(FusionTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(runtime_cfg, model_cfg)

    def forward(self, ref_bxyz, ref_feat, group_ids):
        return ref_feat

class MeanFusion(FusionTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(runtime_cfg, model_cfg)

    def forward(self, ref_bxyz, ref_feat, group_ids):
        """
        Args:
            ref_bxyz [N, 4]
            ref_feat [N, C]
            group_ids [N]
        Returns:
            fused_feat [N, C] feature fused within each group
        """
        num_groups = group_ids.max().item() + 1
        group_feat_mean = scatter(ref_feat, group_ids, dim=0,
                                  dim_size=num_groups, reduce='mean')
        
        return group_feat_mean[group_ids]

class ClusterFusion(FusionTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(runtime_cfg, model_cfg)

    def forward(self, ref_bxyz, ref_feat, group_ids):
        num_groups = group_ids.max().item() + 1
        cluster_center = scatter(ref_bxyz, group_ids, dim=0,
                                 dim_size=num_groups, reduce='mean')

        eigvals, eigvecs = pca_by_group(ref_bxyz[:, 1:4], group_ids)
        import ipdb; ipdb.set_trace()
        
        return ref_feat

class VoxelFusion(FusionTemplate):
    def __init__(self, runtime_cfg, model_cfg):
        super().__init__(runtime_cfg, model_cfg)

    def forward(self, ref_bxyz, ref_feat, group_ids):
        num_groups = group_ids.max().item() + 1
        cluster_center = scatter(ref_bxyz, group_ids, dim=0,
                                 dim_size=num_groups, reduce='mean')

        eigvals, eigvecs = pca_by_group(ref_bxyz[:, 1:4], group_ids)
        import ipdb; ipdb.set_trace()
        
        return ref_feat

FUSIONS = dict(
    IFusion=IFusion,
    LineFusion=LineFusion,
    ClusterFusion=ClusterFusion,
    VoxelFusion=VoxelFusion,
    MeanFusion=MeanFusion,
)
