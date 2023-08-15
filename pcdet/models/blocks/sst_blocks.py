# using flat2win_v2 without voxel_drop_level
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .basic_blocks import build_norm_layer

from pcdet.ops.sst.sst_ops import flat2window_v2, window2flat_v2

class WindowAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, batch_first=False, layer_id=None, layer_cfg=dict()):
        super().__init__()
        self.nhead = nhead

        if layer_cfg.get('cosine', False):
            from .sst_utils import CosineMultiheadAttention
            tau_min = layer_cfg.get('tau_min', 0.01)
            self.self_attn = CosineMultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=False, tau_min=tau_min,
                cosine=True,
                non_shared_tau=layer_cfg.get('non_shared_tau', False)
            )
        elif layer_cfg.get('linear', False):
            raise NotImplementedError
            from mmdet3d.models.sst.linear_msa import LinearMultiheadAttention
            self.self_attn = LinearMultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=False
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.exe_counter = 0

        self.layer_id = layer_id

    def forward(self, voxel_feat, voxel_wise_dict, **kwargs):
        '''Perform Self-attention within each window
        Args:
            feat_2d [N, D]: n nodes with feature dimension d
            pos_dict
            
        Out:
            shifted_feat_dict: the same type as window_feat_dict
        '''
        feat_dim = voxel_feat.shape[-1]
        dtype, device = voxel_feat.dtype, voxel_feat.device
        #out_feat_dict = {}
        shift = kwargs['shift']
        drop_info = kwargs['drop_info']

        #feat_3d_dict = flat2window_v2(feat_2d, ind_dict)

        #attn_map_dict = {}
        output_feat = torch.zeros_like(voxel_feat)
        for dl in range(len(drop_info['range'])):
            dl_mask = voxel_wise_dict[f'voxel_drop_level_s{shift}'] == dl
            if not dl_mask.any():
                continue

            max_tokens = drop_info['num_sampled_tokens'][dl]
            voxel_window_indices = voxel_wise_dict[f'voxel_window_indices_s{shift}'][dl_mask]
            voxel_in_window_indices = voxel_wise_dict[f'voxel_in_window_indices_s{shift}'][dl_mask]

            # compute unique window index
            unique_window_indices, inverse_map = \
                    voxel_window_indices.unique(return_inverse=True)
            num_windows = unique_window_indices.shape[0]
            valid_indices = inverse_map * max_tokens + voxel_in_window_indices

            # embed feature in a [num_windows, num_tokens, feat_dim] tensor
            feat_3d = torch.zeros((num_windows * max_tokens, feat_dim),
                                  dtype=dtype, device=device)
            feat_3d[valid_indices] = voxel_feat[dl_mask]
            feat_3d = feat_3d.view(num_windows, max_tokens, feat_dim).permute(1, 0, 2)

            # embed key padding mask in [num_windows, num_tokens] tensor
            key_padding_mask = torch.ones(num_windows * max_tokens,
                                          dtype=torch.bool, device=device) # True means masked
            key_padding_mask[valid_indices] = False
            key_padding_mask = key_padding_mask.view(num_windows, max_tokens)
            
            # get pos embedding [num_windows, num_tokens, feat_dim]
            pos = torch.ones(num_windows * max_tokens, feat_dim, device=device, dtype=dtype)
            pos[valid_indices] = voxel_wise_dict[f'voxel_pos_embed_s{shift}'][dl_mask]
            pos = pos.view(num_windows, max_tokens, feat_dim).permute(1, 0, 2)

            v = feat_3d
            
            assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape}, feat_shape:{feat_3d.shape}'
            q = k = feat_3d + pos

            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)

            output_feat[dl_mask] = out_feat_3d.view(num_windows*max_tokens, feat_dim)[valid_indices]
            
        return output_feat

        #for name in feat_3d_dict:
        #    #  [n, num_token, embed_dim]
        #    pos = pos_dict[name]

        #    feat_3d = feat_3d_dict[name]
        #    feat_3d = feat_3d.permute(1, 0, 2)

        #    v = feat_3d

        #    if pos is not None:
        #        pos = pos.permute(1, 0, 2)
        #        assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape}, feat_shape:{feat_3d.shape}'
        #        q = k = feat_3d + pos
        #    else:
        #        q = k = feat_3d

        #    key_padding_mask = key_padding_dict[name]
        #    out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
        #    out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)

        #results = window2flat_v2(out_feat_dict, ind_dict)
        #
        #return results

class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, layer_id=None, mlp_dropout=0, layer_cfg=dict()):
        super().__init__()
        assert not batch_first, 'Current version of PyTorch does not support batch_first in MultiheadAttention. After upgrading pytorch, do not forget to check the layout of MLP and layer norm to enable batch_first option.'
        self.batch_first = batch_first
        self.win_attn = WindowAttention(d_model, nhead, dropout, layer_id=layer_id, layer_cfg=layer_cfg)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        use_bn = layer_cfg.get('use_bn', False)
        if use_bn:
            self.norm1 = build_norm_layer(dict(type='naiveSyncBN1d', momentum=layer_cfg.get('mom', 0.1)), d_model)[1]
            self.norm2 = build_norm_layer(dict(type='naiveSyncBN1d', momentum=layer_cfg.get('mom', 0.1)), d_model)[1]
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)

        self.activation = _get_activation_fn(activation)
        self.post_norm = layer_cfg.get('post_norm', True)

    def forward(
        self,
        voxel_feat,
        voxel_wise_dict,
        **kwargs
        ):
        if self.post_norm:
            voxel_feat2 = self.win_attn(voxel_feat, voxel_wise_dict, **kwargs) #[N, d_model]
            voxel_feat = voxel_feat + self.dropout1(voxel_feat2)
            voxel_feat = self.norm1(voxel_feat)
            voxel_feat2 = self.linear2(self.dropout(self.activation(self.linear1(voxel_feat))))
            voxel_feat = voxel_feat + self.dropout2(voxel_feat2)
            voxel_feat = self.norm2(voxel_feat)
        else:
            assert False
            src2 = self.norm1(src)
            src2 = self.win_attn(src2, pos_dict, ind_dict, key_padding_mask_dict) #[N, d_model]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        return voxel_feat
    

class BasicShiftBlockV2(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, block_id=-100, layer_cfg=dict()):
        super().__init__()

        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 0, layer_cfg=layer_cfg)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 1, layer_cfg=layer_cfg)
        # BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i], dropout, activation, batch_first=False)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(self,
                voxel_feat,
                voxel_wise_dict,
                **kwargs
                ):
        """Perform Transformer computation within each window.

        Args:
            node_feat [N, D]: n nodes, d-dimensional feature
            pos_dict_list: 

        """
        num_shifts = 2

        for i in range(2):
            #this_id = i % num_shifts
            #pos_dict = pos_dict_list[this_id]
            #ind_dict = ind_dict_list[this_id]
            #key_mask_dict = key_mask_dict_list[this_id]

            layer = self.encoder_list[i]
            voxel_feat = layer(voxel_feat, voxel_wise_dict, shift=i, **kwargs)

        return voxel_feat

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
