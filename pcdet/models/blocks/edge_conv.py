import torch
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F
from functools import partial

class EdgeConv(nn.Module):
    def __init__(self, conv_cfg):
        super().__init__()
        self.pos_channel = 3
        self.in_channel = in_channel = conv_cfg["INPUT_CHANNEL"]
        mlp_channels = conv_cfg["MLP_CHANNELS"]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.aggr = 'max'

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.mlp_l0 = nn.Linear(self.pos_channel, mlp_channels[0], bias=False)
        self.norm_l0 = norm_fn(mlp_channels[0])
        if in_channel > 0:
            self.mlp_f0 = nn.Linear(in_channel, mlp_channels[0], bias=False)
            self.norm_f0 = norm_fn(mlp_channels[0])

        last_channel = mlp_channels[0]
        for out_channel in mlp_channels[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(norm_fn(out_channel))
            last_channel = out_channel
        self.num_point_features = last_channel

    def forward(self, ref, query, graph):
        e_ref, e_query, e_weight = graph

        pos_diff = (ref.bxyz[e_ref] - query.bxyz[e_query])[:, 1:4] # [E, 3]
        
        pos_feat = self.norm_l0(self.mlp_l0(pos_diff)) # [E, 3] -> [E, D]
        if self.in_channel > 0:
            ref_feat2 = self.norm_f0(self.mlp_f0(ref.feat))
            edge_feat = F.relu(pos_feat + ref_feat2[e_ref], inplace=False)
        else:
            edge_feat = F.relu(pos_feat, inplace=False)

        ## mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_feat = conv(edge_feat)
            edge_feat = F.relu(bn(edge_feat), inplace=False)
        query_feat = scatter(edge_feat, e_query, dim=0,
                             dim_size=query.bxyz.shape[0], reduce=self.aggr)
        #if query_feat.shape[-1] == ref.feat.shape[-1]:
        #    query_feat = ref.feat + query_feat

        return query_feat

    def extra_repr(self):
        return f"aggr={self.aggr}"

class EdgeConvUp(nn.Module):
    def __init__(self, conv_cfg):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        skip_channel = conv_cfg.get("SKIP_CHANNEL", None)
        prev_channel = conv_cfg["PREV_CHANNEL"]
        mlp_channels = conv_cfg["MLP_CHANNELS"]
        self.skip = skip_channel is not None

        self.mlp_f0 = nn.Linear(prev_channel, mlp_channels[0], bias=False)
        self.norm_f0 = nn.BatchNorm1d(mlp_channels[0])
        if skip_channel is not None:
            self.mlp_s0 = nn.Linear(skip_channel, mlp_channels[0], bias=False)
            self.norm_s0 = nn.BatchNorm1d(mlp_channels[0])

        last_channel = mlp_channels[0]
        for out_channel in mlp_channels[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.num_point_features = last_channel

    def forward(self, ref, query, graph):
        e_ref, e_query, _ = self.graph(ref, query)

        pos_dist = (ref.bxyz[e_ref, 1:4] - query.bxyz[e_query, 1:4]).norm(p=2, dim=-1) # [E]
        pos_dist = 1.0 / (pos_dist + 1e-8)

        weight_sum = scatter(pos_dist, e_query, dim=0,
                             dim_size=query.bxyz.shape[0], reduce='sum')
        weight = pos_dist / weight_sum[e_query] # [E]

        ref_feat2 = self.norm_f0(self.mlp_f0(ref.feat))[e_ref]
        query_feat = scatter(ref_feat2*weight[:, None], e_query, dim=0,
                             dim_size=query.bxyz.shape[0], reduce='sum')

        if self.skip:
            query_skip_feat = self.norm_s0(self.mlp_s0(query.feat))
            query_feat = F.relu(query_feat + query_skip_feat, inplace=False)
        else:
            query_feat = F.relu(query_feat, inplace=False)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            query_feat = F.relu(bn(conv(query_feat)), inplace=False)

        return query_feat
