import torch
from torch import nn
from tqdm import tqdm

from pcdet.utils import geop_utils

def transform(points, r, t):
    """
    Args:
        points [N, 3]
        r [N, 3]
        t [N, 3]
    """
    points_diff = points - points.mean(0)
    return points + r.cross(points_diff) + t

class GDSolver(object):
    def __init__(self, model_cfg, runtime_cfg):
        self.lr = model_cfg.get("LR", 1e-3)
        self.num_inner_iters = model_cfg.get("NUM_INNER_ITERS", 300)
        self.num_outer_iters = model_cfg.get("NUM_OUTER_ITERS", 50)
        self.stopping_delta = model_cfg.get("STOPPING_DELTA", 1e-2)

    def __call__(self, moving, frames, corres_edges, rigidity_edges):
        e_movings, e_refs, ref_sweeps = corres_edges
        import polyscope as ps; ps.set_up_dir('z_up'); ps.init()
        num_moving_points = moving.sxyz.shape[0]
        rt = moving.rt
        optimizer = torch.optim.AdamW([rt], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500])
        current_sweep = moving.sxyz[0, 0].round().long().item()

        for inner_itr in range(self.num_inner_iters):
            loss = 0.0
            optimizer.zero_grad()
            for e_moving, e_ref, ref_sweep in zip(e_movings, e_refs, ref_sweeps):
                s = ref_sweep - current_sweep
                r, t = rt[:, :3], rt[:, 3:]
                r_s = r * s
                t_s = t * s
                moved_xyz = transform(moving.sxyz[e_moving, 1:], r_s[e_moving], t_s[e_moving])
                loss += (moved_xyz - frames[ref_sweep].sxyz[e_ref, 1:]).square().mean()
                    
            l2 = (rt[rigidity_edges[0]] - rt[rigidity_edges[1]]).square().mean() * 10000
            l3 = rt[:, :2].square().mean()*10000 + rt[:, 2].square().mean()*100
            loss += l2 + l3*0.0000
            loss.backward()
            #rt.grad[last_sweep_mask] = 0.0
            optimizer.step()
            scheduler.step()
            if rt.grad.norm(p=2) < 1e-4:
                break
        moving_xyz = transform(moving.sxyz[:, 1:], rt[:, :3], rt[:, 3:])
        moving.velo = transform(moving.sxyz[:, 1:], rt[:, :3], rt[:, 3:]) - moving.sxyz[:, 1:]
        print(f'loss={loss.item():.6f}, rigidity={l2.item():.6f}, conservative={l3.item():.6f}, grad_norm={rt.grad.norm(p=2):.6f}, median_velo={moving.velo.norm(p=2, dim=-1).median()}')

        return moving

#def solve_gd(self, optimizer, scheduler, edges, rt, ref, e_ref_0, e_ref_1):
#    for inner_itr in range(100):
#        loss = 0.0
#
#def solve_q(self, optimizer, scheduler, edges, rt, ref, e_ref_0, e_ref_1):
#    import scipy.sparse as sp
#    from torch_geometric.utils import to_scipy_sparse_matrix
#    e0 = torch.cat([e0 for e0, e1 in edges], dim=0)
#    e1 = torch.cat([e1 for e0, e1 in edges], dim=0)
#    e0_cat = torch.cat([e0, e_ref_0], dim=0)
#    e1_cat = torch.cat([e1, e_ref_1], dim=0)
#    edges_cat = torch.stack([e_ref_0, e_ref_1], dim=0)
#    adj = to_scipy_sparse_matrix(edges_cat, num_nodes=ref.bxyz.shape[0])
#    num_components, component = sp.csgraph.connected_components(adj)
#    #_, count = np.unique(component, return_counts=True)
#    #dist = (ref.bxyz[e0_cat, 1:] - ref.bxyz[e1_cat, 1:]).norm(p=2, dim=-1) / (ref.bxyz[e0_cat, 0] - ref.bxyz[e1_cat, 0]).abs().clamp(min=1)
#    #dist = dist.detach().cpu().numpy()
#    #cut_thresh = dist.max()
#    #valid_mask = (count[component[e0_cat.detach().cpu().numpy()]] > 10000)
#    #while count.max() > 10000:
#    #    cut_thresh *= 0.95
#    #    remove_mask_this = (dist >= cut_thresh) & valid_mask
#    #    remove_mask_this = torch.from_numpy(remove_mask_this).cuda()
#    #    edges_cat_this = edges_cat[:, ~remove_mask_this]
#    #    adj = to_scipy_sparse_matrix(edges_cat_this, num_nodes=ref.bxyz.shape[0])
#    #    num_components, component = sp.csgraph.connected_components(adj)
#    #    _, count = np.unique(component, return_counts=True)
#    #    print(num_components, count.max(), edges_cat_this.shape)
#        
#    return component


SOLVERS = dict(
    GDSolver=GDSolver,
)
