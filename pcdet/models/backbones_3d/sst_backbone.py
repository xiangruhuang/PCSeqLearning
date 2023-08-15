import torch
import torch.nn as nn
import copy

from .sst_utils import SSTInputLayerV2
from pcdet.models.blocks.sst_blocks import BasicShiftBlockV2
from .post_processors import build_post_processor

"""SST (v2)
Adapted from original codebase.

"""
class SST(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(SST, self).__init__()
        self.in_channel = runtime_cfg.get("num_point_features", None)
        self.model_cfg = model_cfg
        self.scale = runtime_cfg.get("scale", 1.0)
        self.num_shifts = 2
        self.output_key = 'sst_out'

        self.tokenizer = SSTInputLayerV2(model_cfg.TOKENIZER_CFG, runtime_cfg)

        self.build_transformer_layers(model_cfg.TRANSFORMER_CFG)


        runtime_cfg['input_key'] = self.output_key 
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        #self.build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}), runtime_cfg)
        self.forward_dict = {}
            
        self._reset_parameters()

        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features
        else:
            self.num_point_features = self.d_model[-1]

    def build_transformer_layers(self, transformer_cfg):
        d_model = transformer_cfg.get("D_MODEL", [])
        nhead = transformer_cfg.get("NHEAD", [])
        dim_feedforward = transformer_cfg.get("DIM_FEEDFORWARD", [])
        self.num_blocks = num_blocks = transformer_cfg.get("NUM_BLOCKS", 6)
        self.debug = transformer_cfg.get("DEBUG", False)
        self.checkpoint_blocks = transformer_cfg.get("CHECKPOINT_BLOCKS", [])
        self.conv_shortcut = transformer_cfg.get("CONV_SHORTCUT", False)

        activation = transformer_cfg.get("ACTIVATION", "gelu")
        dropout = transformer_cfg.get("DROPOUT", 0.0)
        layer_cfg = transformer_cfg.get("LAYER_CFG", {})
        
        self.d_model = d_model = [int(d*self.scale) for d in d_model]
        self.dim_feedforward = dim_feedforward = [int(d*self.scale) for d in dim_feedforward]
        self.nhead = nhead = [int(d*self.scale) for d in nhead]

        if self.in_channel is not None:
            self.linear0 = nn.Linear(self.in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )
        self.transformer_layers = nn.ModuleList(block_list)

    def forward(self, batch_dict):
        '''
        Args:
            
        Returns:
            
        '''
        if 'memory' not in batch_dict:
            batch_dict['memory'] = {}
        batch_size = batch_dict['batch_size']

        # tokenization
        voxel_wise_dict = dict(
            voxel_feat=batch_dict['voxel_feat'],
            voxel_coords=batch_dict['voxel_coords'],
            #voxel_xyz=batch_dict['voxel_xyz'],
            #voxel_batch_index=batch_dict['voxel_batch_index'],
            voxel_bxyz=batch_dict['voxel_bxyz'],
            voxel_segmentation_label=batch_dict['voxel_segmentation_label']
        )

        voxel_wise_dict = self.tokenizer(voxel_wise_dict)

        voxel_coords = voxel_wise_dict['voxel_coords']
        voxel_feat = voxel_wise_dict['voxel_feat']

        #ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(self.num_shifts)]
        #padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(self.num_shifts)]
        #pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(self.num_shifts)]

        if hasattr(self, 'linear0'):
            voxel_feat = self.linear0(voxel_feat)

        for i, block in enumerate(self.transformer_layers):
            last_memory = torch.cuda.max_memory_allocated() / 2**30
            voxel_feat = block(voxel_feat, voxel_wise_dict, drop_info=self.tokenizer.drop_info)
            batch_dict['memory'][f'transformer_{i}_in_GB'] = torch.cuda.max_memory_allocated() / 2**30 - last_memory

        batch_dict['voxel_window_indices_s0'] = voxel_wise_dict['voxel_window_indices_s0']
        batch_dict['voxel_window_indices_s1'] = voxel_wise_dict['voxel_window_indices_s1']
        batch_dict['sst_out_feat'] = voxel_feat
        batch_dict['sst_out_bxyz'] = voxel_wise_dict['voxel_bxyz']
        batch_dict['sst_gt_segmentation_label'] = voxel_wise_dict['voxel_segmentation_label']

        if self.post_processor is not None:
            batch_dict = self.post_processor(batch_dict)
        
        return batch_dict
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name and 'tau' not in name:
                nn.init.xavier_uniform_(p)

