from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .post_processors import build_post_processor

from pcdet.models.blocks import SparseBasicBlock, post_act_block
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone

class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, model_cfg, runtime_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        grid_size = runtime_cfg['grid_size']
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = runtime_cfg['voxel_size']
        self.point_cloud_range = runtime_cfg['point_cloud_range']
        self.scale = runtime_cfg.get("scale", 1.0)
        input_channels = runtime_cfg['input_channels']

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        d1 = int(32*self.scale)
        #self.conv_input = spconv.SparseSequential(
        #    spconv.SubMConv3d(input_channels, d1, 3, padding=1, bias=False, indice_key='subm1'),
        #    norm_fn(d1),
        #    nn.ReLU(),
        #)
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(input_channels, d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(d1, d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(d1, d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(d1,   2*d1, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(2*d1, 2*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(2*d1, 2*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(2*d1, 4*d1, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(4*d1, 4*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(4*d1, 4*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(4*d1, 8*d1, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(8*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(8*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        if model_cfg.get("CONV5", False):
            self.conv5 = spconv.SparseSequential(
                block(8*d1, 8*d1, 3, norm_fn=norm_fn, stride=(2, 1, 1), padding=(0, 1, 1), indice_key='spconv5', conv_type='spconv'),
                block(8*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
                block(8*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
            )
            # decoder
            self.conv_up_t5 = SparseBasicBlock(8*d1, 8*d1, indice_key='subm5', norm_fn=norm_fn)
            self.conv_up_m5 = block(16*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
            self.inv_conv5 = block(8*d1, 8*d1, 3, norm_fn=norm_fn, indice_key='spconv5', conv_type='inverseconv')
        else:
            self.conv5 = None
        
        if model_cfg.get("CONV6", False):
            self.conv6 = spconv.SparseSequential(
                block(8*d1, 8*d1, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv6', conv_type='spconv'),
                block(8*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm6'),
                block(8*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm6'),
            )
            # decoder
            self.conv_up_t6 = SparseBasicBlock(8*d1, 8*d1, indice_key='subm6', norm_fn=norm_fn)
            self.conv_up_m6 = block(16*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm6')
            self.inv_conv6 = block(8*d1, 8*d1, 3, norm_fn=norm_fn, indice_key='spconv6', conv_type='inverseconv')
        else:
            self.conv6 = None
        
        global_cfg = model_cfg.get("GLOBAL", None)
        if global_cfg is not None:
            last_in_channels = runtime_cfg.get("in_channels", 16*d1)
            runtime_cfg['in_channels'] = 16*d1
            self.global_conv = BaseBEVBackbone(global_cfg, runtime_cfg)
            runtime_cfg['in_channels'] = last_in_channels
        else:
            self.global_conv = None

        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(8*d1, 8*d1, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(16*d1, 8*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(8*d1, 4*d1, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(4*d1, 4*d1, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(8*d1, 4*d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(4*d1, 2*d1, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(2*d1, 2*d1, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(4*d1, 2*d1, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(2*d1, d1, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(d1, d1, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(2*d1, d1, 3, norm_fn=norm_fn, indice_key='subm1')
        self.inv_conv1 = block(d1, d1, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        
        self.num_point_features = d1
        
        #for c in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, \
        #          ]:
        #    c[0][0].weight.data[:] = 0.01
        #    c[1][0].weight.data[:] = 0.01
        #    c[2][0].weight.data[:] = 0.01
        #    c[0][1].weight.data[:] = 1
        #    c[0][1].bias.data[:] = 0
        #    c[1][1].weight.data[:] = 1
        #    c[1][1].bias.data[:] = 0
        #    c[2][1].weight.data[:] = 1
        #    c[2][1].bias.data[:] = 0
        #for c in [self.conv_up_m5, self.inv_conv5,
        #          self.conv_up_m4, self.inv_conv4,
        #          self.conv_up_m3, self.inv_conv3,
        #          self.conv_up_m2, self.inv_conv2,
        #          self.conv_up_m1, self.inv_conv1]:
        #    c[0].weight.data[:] = 0.01
        #    c[1].weight.data[:] = 1
        #    c[1].bias.data[:] = 0

        #for c in [self.conv_up_t5,
        #          self.conv_up_t4,
        #          self.conv_up_t3,
        #          self.conv_up_t2,
        #          self.conv_up_t1,]:
        #    c.conv1.weight.data[:] = 0.01
        #    c.conv2.weight.data[:] = 0.01
        #    c.bn1.weight.data[:] = 1
        #    c.bn2.weight.data[:] = 1
        #    c.bn1.bias.data[:] = 0.0
        #    c.bn2.bias.data[:] = 0.0


        runtime_cfg['input_key'] = 'unet_voxel' 
        runtime_cfg['num_point_features'] = self.num_point_features
        self.post_processor = build_post_processor(model_cfg.get("POST_PROCESSING_CFG", {}),
                                                   runtime_cfg)
        if self.post_processor:
            self.num_point_features = self.post_processor.num_point_features

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_feat'], batch_dict['voxel_bcoords'][:, [0,3,2,1]]
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        if self.conv5:
            x_conv5 = self.conv5(x_conv4)
            if self.conv6:
                x_conv6 = self.conv6(x_conv5)
                x_up6 = self.UR_block_forward(x_conv6, x_conv6, self.conv_up_t6, self.conv_up_m6, self.inv_conv6)
            else:
                if self.global_conv:
                    dense_x = x_conv5.dense()
                    B, C, _, W, H = dense_x.shape
                    dense_x = dense_x.reshape(B, C*2, W, H)
                    batch_dict['spatial_features'] = dense_x
                    batch_dict = self.global_conv(batch_dict)
                    dense_out = batch_dict['spatial_features_2d']

                    dense_out = dense_out.reshape(B, C, 2, W, H)
                    indices = x_conv5.indices.long()
                    x_up6 = x_conv5.replace_feature(dense_out[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]])
                else:
                    x_up6 = x_conv5
        
            # for segmentation head
            x_up5 = self.UR_block_forward(x_conv5, x_up6, self.conv_up_t5, self.conv_up_m5, self.inv_conv5)
        else:
            x_up5 = x_conv4

        x_convs = [x_conv1, x_conv2, x_conv3, x_conv4]
        if self.conv5:
            x_convs.append(x_conv5)
        for i, x_conv in enumerate(x_convs):
            downsample_times = [1, 2, 4, 8, [8, 8, 16]][i]
            downsample_times = torch.tensor(downsample_times).to(x_conv.features)
            point_corners = common_utils.get_voxel_corners(
                x_conv.indices[:, 1:], downsample_times=downsample_times,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            voxel_size = torch.tensor(self.voxel_size, device=point_corners.device).float()
            point_corners += voxel_size * 0.5
            batch_dict[f'spconv_unet_bcenter{5-i}'] = torch.cat([x_conv.indices[:, 0:1], point_corners], dim=-1)
            batch_dict[f'spconv_unet_feat{5-i}'] = x_conv.features

        #for key in x_conv5.indice_dict.keys():
        #    print(key, x_conv5.indice_dict[key].pair_bwd.shape, (x_conv5.indice_dict[key].pair_bwd != -1).sum() / x_conv5.indice_dict[key].pair_bwd.shape[-1])
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_up5, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.inv_conv1)

        batch_dict['unet_voxel_feat'] = x_up1.features
        point_coords = common_utils.get_voxel_centers(
            x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['unet_voxel_bxyz'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        if self.post_processor:
            batch_dict = self.post_processor(batch_dict)
        return batch_dict
