import torch
from torch_scatter import scatter

def pca_by_group(point_xyz, group_ids):
    """
    Args:
        point_xyz [N, 3]
        group_ids [N] in range [G]
    Returns:
        eigvals [G, 3]
        eigvecs [G, 3, 3]
    """
    num_groups = group_ids.max().item() + 1

    group_center = scatter(point_xyz, group_ids, dim=0,
                           dim_size=num_groups, reduce='mean')
    
    point_d = point_xyz - group_center[group_ids] # [N, 3]

    point_ddT = point_d.unsqueeze(-1) @ point_d.unsqueeze(-2) # [N, 3, 3]
    group_cov = scatter(point_ddT, group_ids, dim=0,
                        dim_size=num_groups, reduce='mean') # [G, 3, 3]
    group_cov = group_cov + torch.diag_embed(torch.tensor([1, 1, 1]).float()).to(cov).repeat(num_groups, 1, 1) * self.theta1
    eigvecs, eigvals = torch.linalg.svd(group_cov)
    return eigvals, eigvecs
