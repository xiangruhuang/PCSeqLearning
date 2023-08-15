import torch
from torch import nn

def build_norm_layer(num_features, norm_cfg):
    norm_type = norm_cfg['type']
    norm_layer = getattr(nn, norm_type)
    
    norm_cfg_clone = {}
    norm_cfg_clone.update(norm_cfg)
    norm_cfg_clone.pop('type')
    norm_cfg_clone['num_features'] = num_features
    return norm_layer(**norm_cfg_clone)

def build_conv_layer(in_channels, out_channels, conv_cfg, **conv_kwargs):
    conv_type = conv_cfg['type']
    conv_layer = getattr(nn, conv_type)
    
    conv_cfg_clone = {}
    conv_cfg_clone.update(conv_cfg)
    conv_cfg_clone.pop('type')

    return conv_layer(in_channels=in_channels,
                      out_channels=out_channels,
                      **conv_cfg_clone,
                      **conv_kwargs)

def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True, eps=1e-5):
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                nn.BatchNorm1d(channels[i], momentum=bn_momentum, eps=eps),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )

def MLPBlock(in_channel, out_channel, norm_cfg,
             activation=nn.LeakyReLU(0.2), bias=True):
    norm_layer = build_norm_layer(out_channel, norm_cfg)
    return nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                norm_layer,
                activation
           )

