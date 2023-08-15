# move the computation of position embeding and mask in middle_encoder_layer
import math
import numpy as np

import torch
from torch import nn
from torch._C import _infer_size, _add_docstr
from torch.nn import _reduction as _Reduction
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
from torch.nn import grad  # noqa: F401
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
# from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.functional import linear, softmax, dropout

# Modified from the MSA in PyTorch
import warnings

from pcdet.ops.sst.sst_ops import (
    flat2window_v2,
    window2flat_v2,
    get_inner_win_inds,
    make_continuous_inds,
    get_flat2win_inds_v2,
    get_window_coors
)
from pcdet.utils import common_utils

Tensor = torch.Tensor

class SSTInputLayerV2(nn.Module):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching. 
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing. 
    """

    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        self.drop_info = model_cfg.get("DROP_INFO", {})
        self.drop_info = self.drop_info[self.mode]
        self.sparse_shape = model_cfg.get("SPARSE_SHAPE", None)
        self.shuffle_voxels = model_cfg.get("SHUFFLE_VOXELS", True)
        self.debug = model_cfg.get("DEBUG", False)
        self.window_shape = model_cfg.get("WINDOW_SHAPE", None)
        self.normalize_pos = model_cfg.get("NORMALIZE_POS", False)
        self.pos_temperature = model_cfg.get("POS_TEMPERATURE", 10000)
        self.mute = model_cfg.get("MUTE", False)

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def forward(self, voxel_wise_dict):
        '''
        Args:
            voxel_feat: shape=[N, C], N is the voxel num in the batch.
            voxel_coords: shape=[N, 4], [b, z, y, x]
        Returns:
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
                {
                 feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
                 flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
                }
        '''
        self.set_drop_info()

        #if self.shuffle_voxels:
        #    # shuffle the voxels to make the drop process uniform.
        #    shuffle_inds = torch.randperm(len(voxel_wise_dict['voxel_feat']))
        #    common_utils.filter_dict(voxel_wise_dict, shuffle_inds)
        
        voxel_coords = voxel_wise_dict['voxel_coords'].long()
        voxel_feat = voxel_wise_dict['voxel_feat']

        #voxel_info = self.window_partition(voxel_wise_dict['voxel_coords'])
        #voxel_info.update(voxel_wise_dict)
        #voxel_info = self.drop_voxel(voxel_info, 2) # voxel_info is updated in this function

        #voxel_feat = voxel_info['voxel_feat']
        #voxel_coords = voxel_info['voxel_coords']
        
        voxel_wise_dict = self.window_partition(voxel_wise_dict)
        voxel_wise_dict = self.drop_voxel(voxel_wise_dict)
        
        for i in range(2):
            #voxel_wise_dicts = self.drop_voxel(voxel_wise_dicts[i], window_wise_dict[i])
            #self.drop_voxel(batch_win_inds)
            voxel_window_indices_si = voxel_wise_dict[f'voxel_window_indices_s{i}']
            num_voxels = voxel_window_indices_si.shape[0]
            voxel_drop_level_si = voxel_wise_dict[f'voxel_drop_level_s{i}']
            voxel_in_window_zyx_si = voxel_wise_dict[f'voxel_in_window_zyx_s{i}']
            voxel_pos_embed_si = self.get_pos_embed(voxel_in_window_zyx_si,
                                                    voxel_feat.size(1),
                                                    voxel_feat.dtype)

            voxel_wise_dict[f'voxel_pos_embed_s{i}'] = voxel_pos_embed_si
            #for dl in range(len(self.drop_info['range'])):
            #    dl_mask = voxel_drop_level == dl
            #    import ipdb; ipdb.set_trace()
            #    max_tokens = self.drop_info['num_sampled_tokens'][dl]
            #    voxel_window_indices_l = voxel_window_indices[dl_mask]
            #    unique_, voxel_windows_indices_l = voxel_windows_indices_l.unique(return_inverse=True)
            #    voxel_in_window_zyx_l = voxel_in_window_zyx[dl_mask]
            #    voxel_in_window_indices = get_inner_win_inds(voxel_window_indices_l)
            #    
            #    #voxel_indices = voxel_window_indices_l * max_tokens + voxel_in_window_indices
            #    #voxel_keep_indices = torch.where(dl_mask)[0]

            #    import ipdb; ipdb.set_trace()
            #    pass

            #    

            #flat2win_inds = get_flat2win_inds_v2(voxel_wise_dicts[i],
            #                                     window_wise_dicts[i],
            #                                     self.drop_info, debug=True)



            #key_mask = self.get_key_padding_mask(flat2win_inds)

        #if self.debug:
        #    coords_3d_dict_shift0 = flat2window_v2(voxel_coords, voxel_info['flat2win_inds_shift0'])
        #    coords_2d = window2flat_v2(coords_3d_dict_shift0, voxel_info['flat2win_inds_shift0'])
        #    assert (coords_2d == voxel_coords).all()

        #if self.shuffle_voxels:
        #    voxel_info['shuffle_inds'] = shuffle_inds
        
        return voxel_wise_dict
    
    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds] #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in range(len(drop_info['range'])):
            max_tokens = drop_info['num_sampled_tokens'][dl]
            lower = 0 if dl == 0 else drop_info['range'][dl-1]
            upper = drop_info['range'][dl]
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl
        
        if self.debug:
            assert (target_num_per_voxel > 0).all()
            assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, drop_lvl_per_voxel, inner_win_inds

    def drop_voxel(self, voxel_wise_dict):
        '''
        Args:
            voxel_wise_dict: {
                voxel_window_indices_s{i},
            }
        Returns:
            voxel_wise_dict: {
                voxel_drop_level_s{i}: sampled,
                voxel_used_indices: sampled and shared across shifts,
                voxel_window_indices_s{i}: sampled,
            }
        '''
        num_shifts = 2

        num_voxels = voxel_wise_dict['voxel_window_indices_s0'].shape[0]
        device = voxel_wise_dict['voxel_window_indices_s0'].device
        voxel_keep_mask = torch.ones(num_voxels, device=device, dtype=torch.bool)
        voxel_wise_dict[f'voxel_used_indices'] = torch.arange(num_voxels, 
                                                    device=device, dtype=torch.long)

        # augment with new attributes
        for i in range(num_shifts):
            voxel_drop_level = \
                    voxel_wise_dict[f'voxel_window_indices_s{i}'].new_zeros(num_voxels) - 1
            voxel_wise_dict[f'voxel_drop_level_s{i}'] = voxel_drop_level
            voxel_in_window_indices = \
                    voxel_wise_dict[f'voxel_window_indices_s{i}'].new_zeros(num_voxels) - 1
            voxel_wise_dict[f'voxel_in_window_indices_s{i}'] = voxel_in_window_indices

        # update attributes after dropping voxels
        for i in range(num_shifts):
            voxel_window_indices = voxel_wise_dict[f'voxel_window_indices_s{i}']
            voxel_drop_level = voxel_wise_dict[f'voxel_drop_level_s{i}']
            voxel_in_window_indices = voxel_wise_dict[f'voxel_in_window_indices_s{i}']
            voxel_keep_mask_i, drop_level_i, in_window_indices_i = \
                    self.drop_single_shift(voxel_window_indices[voxel_keep_mask])
            voxel_drop_level[voxel_keep_mask] = drop_level_i
            voxel_in_window_indices[voxel_keep_mask] = in_window_indices_i
            voxel_keep_mask[(voxel_keep_mask == True)] = voxel_keep_mask_i

            voxel_wise_dict[f'voxel_drop_level_s{i}'] = voxel_drop_level
            voxel_wise_dict[f'voxel_in_window_indices_s{i}'] = voxel_in_window_indices
        
        # drop voxels in all shifts
        voxel_wise_dict = common_utils.filter_dict(voxel_wise_dict, voxel_keep_mask)
        for i in range(num_shifts):
            assert (voxel_wise_dict[f'voxel_drop_level_s{i}'] >= 0).all()
            
        return voxel_wise_dict
        #batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        #num_voxels = voxel_window_indices.shape[0]

        #voxel_keep_indices = torch.arange(num_voxels, device=voxel_window_indices.device, dtype=torch.long)

        #voxel_keep_mask = torch.ones(num_voxels, device=voxel_window_indices.device, dtype=torch.bool)
        #keep_mask, drop_level = self.drop_single_shift(voxel_window_indices)
        #if self.debug:
        #    assert (drop_level >= 0).all()

        #drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        #voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        #batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        #if num_shifts == 1:
        #    voxel_info['voxel_keep_inds'] = voxel_keep_inds
        #    voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        #    voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        #    return voxel_info

        #batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        #batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        #keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)
        #if self.debug:
        #    assert (drop_lvl_s1 >= 0).all()

        ## drop data in first shift again
        #drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        #voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        #batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        #drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        #batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        #voxel_info['voxel_keep_inds'] = voxel_keep_inds
        #voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        #voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        #voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        #voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        #voxel_keep_inds = voxel_info['voxel_keep_inds']

        #voxel_num_before_drop = len(voxel_info['voxel_coords'])
        ##voxel_info['voxel_feat'] = voxel_info['voxel_feat'][voxel_keep_inds]
        ##voxel_info['voxel_coords'] = voxel_info['voxel_coords'][voxel_keep_inds]

        ## Some other variables need to be dropped.
        #for k, v in voxel_info.items():
        #    if k not in ['voxel_keep_inds',
        #                 'voxel_drop_level_shift0', 'voxel_drop_level_shift1',
        #                 'batch_win_inds_shift0', 'batch_win_inds_shift1']:
        #        voxel_info[k] = v[voxel_keep_inds]

        #### sanity check
        #if self.debug and self.training:
        #    for dl in self.drop_info:
        #        max_tokens = self.drop_info[dl]['max_tokens']

        #        mask_s0 = drop_lvl_s0 == dl
        #        if not mask_s0.any():
        #            if not self.mute:
        #                print(f'No voxel belongs to drop_level:{dl} in shift 0')
        #            continue
        #        real_max = torch.bincount(batch_win_inds_s0[mask_s0]).max()
        #        assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift0'

        #        mask_s1 = drop_lvl_s1 == dl
        #        if not mask_s1.any():
        #            if not self.mute:
        #                print(f'No voxel belongs to drop_level:{dl} in shift 1')
        #            continue
        #        real_max = torch.bincount(batch_win_inds_s1[mask_s1]).max()
        #        assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift1'
        ####
        #return voxel_info

    @torch.no_grad()
    def window_partition(self, voxel_wise_dict):
        """Hash voxel coordinates into windows of fixed size.
        Args:
            coors [V, 4]: coordinates (first dimension corresponds to batch?),
                          **required** to be non-negative
            self.sparse_shape [3]: three dimensional scene size (int)
            self.window_shape [3]: three dimensional window size (int)
            do_shift (bool): if True, shift coordinates by half the window size before hashing
        Returns:
            voxel_wise_dict: {
                voxel_window_indices_s{i} [V]: window indices of each voxel (in range [W])
                voxel_in_window_zyx_s{i} [V, 3]: coordinates relative to window
            }
        """
        for i in range(2):
            voxel_window_indices, voxel_in_window_zyx = \
                    get_window_coors(
                        voxel_wise_dict['voxel_coords'], self.sparse_shape,
                        self.window_shape, do_shift = (i % 2 == 1)
                    )

            voxel_wise_dict[f'voxel_window_indices_s{i}'] = voxel_window_indices
            voxel_wise_dict[f'voxel_in_window_zyx_s{i}'] = voxel_in_window_zyx

        return voxel_wise_dict

    @torch.no_grad()
    def get_pos_embed(self, coords_in_win, feat_dim, dtype):
        '''
        Args:
        coords_in_win: shape=[N, 3], order: z, y, x
        '''

        # [N,]
        window_shape = self.window_shape
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif  window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coords_in_win.size(1) == 3
        z, y, x = coords_in_win[:, 0] - win_z/2, coords_in_win[:, 1] - win_y/2, coords_in_win[:, 2] - win_x/2
        assert (x >= -win_x/2 - 1e-4).all()
        assert (x <= win_x/2-1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            torch.div(pos_length, 2, rounding_mode='trunc'), dtype=torch.float32, device=coords_in_win.device) * 2
        #inv_freq = self.pos_temperature ** (2 * torch.div(inv_freq, 2, rounding_mode='trunc') / pos_length)
        inv_freq = self.pos_temperature ** (inv_freq / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        if ndim == 3:
            embed_z = z[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x.sin(), embed_x.cos()], dim=-1).flatten(1)
        embed_y = torch.stack([embed_y.sin(), embed_y.cos()], dim=-1).flatten(1)
        if ndim == 3:
            embed_z = torch.stack([embed_z.sin(), embed_z.cos()], dim=-1).flatten(1)

        # [num_tokens, c]
        if ndim == 3:
            pos_embed_2d = torch.cat([embed_x, embed_y, embed_z], dim=-1).to(dtype)
        else:
            pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)
        
        gap = feat_dim - pos_embed_2d.size(1)
        assert gap >= 0
        if gap > 0:
            assert ndim == 3
            padding = torch.zeros((pos_embed_2d.size(0), gap), dtype=dtype, device=coords_in_win.device)
            pos_embed_2d = torch.cat([pos_embed_2d, padding], dim=1)
        else:
            assert ndim == 2

        #pos_embed_dict = flat2window_v2(
        #    pos_embed_2d, inds_dict)

        return pos_embed_2d

    @torch.no_grad()
    def get_key_padding_mask(self, ind_dict):
        num_all_voxel = len(ind_dict['voxel_drop_level'])
        key_padding = torch.ones((num_all_voxel, 1)).to(ind_dict['voxel_drop_level'].device).bool()

        window_key_padding_dict = flat2window_v2(key_padding, ind_dict)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        
        return window_key_padding_dict

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta
        print(f'drop_info is set to {self.drop_info}, in input_layer')


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_cosine_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    tau, 
    tau_min,
    num_heads,
    attn_mask: Optional[Tensor] = None,
    extra_attn: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    Ns = k.shape[1]
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if tau is not None:
        q = nn.functional.normalize(q, dim=2)
        k = nn.functional.normalize(k, dim=2)
        attn = torch.bmm(q, k.transpose(-2, -1))

        if tau.ndim == 4:
            assert tau.size(1) == num_heads and attn.size(-1) == Ns
            attn = attn.reshape(B // num_heads, num_heads, Nt, Ns)
            attn = attn / tau.clamp(min=tau_min)
            attn = attn.reshape(B, Nt, Ns)
        else:
            attn = attn / tau.clamp(min=tau_min)
    else:
        q = q / math.sqrt(E)
        attn = torch.bmm(q, k.transpose(-2, -1))

    if attn_mask is not None:
        attn += attn_mask
    if extra_attn is not None:
        assert extra_attn.shape == attn.shape, f'{extra_attn.shape} v.s. {attn.shape}'
        attn += extra_attn
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn

def cosine_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    extra_attn: Optional[Tensor] = None,
    tau = None,
    tau_min = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # The following commented lines requires higher version of torch
    # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    # if has_torch_function(tens_ops):
    #     raise NotImplementedError('This line should not be visited.')
    #     return handle_torch_function(
    #         multi_head_attention_forward,
    #         tens_ops,
    #         query,
    #         key,
    #         value,
    #         embed_dim_to_check,
    #         num_heads,
    #         in_proj_weight,
    #         in_proj_bias,
    #         bias_k,
    #         bias_v,
    #         add_zero_attn,
    #         dropout_p,
    #         out_proj_weight,
    #         out_proj_bias,
    #         training=training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         use_separate_proj_weight=use_separate_proj_weight,
    #         q_proj_weight=q_proj_weight,
    #         k_proj_weight=k_proj_weight,
    #         v_proj_weight=v_proj_weight,
    #         static_k=static_k,
    #         static_v=static_v,
    #     )

    # assert pos embedding

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_cosine_attention(q, k, v, tau, tau_min, num_heads, attn_mask, extra_attn, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class CosineMultiheadAttention(nn.MultiheadAttention):
    '''Inherit from standard multihead attention, call the customized multi_head_forward function in forward.
    '''

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, cosine=True, tau_min=0.01, non_shared_tau=False) -> None:
        super(CosineMultiheadAttention, self).__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.batch_first = batch_first

        self.tau_min = tau_min
        if cosine:
            if non_shared_tau:
                self.tau = torch.nn.Parameter(torch.ones(1, num_heads, 1, 1)) # shared between heads
            else:
                self.tau = torch.nn.Parameter(torch.ones(1, 1, 1)) # shared between heads
        else:
            self.tau = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                extra_attn=None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.
          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = cosine_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, extra_attn=extra_attn, tau=self.tau, tau_min=self.tau_min)
        else:
            attn_output, attn_output_weights = cosine_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, extra_attn=extra_attn, tau=self.tau, tau_min=self.tau_min)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
