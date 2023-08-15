import torch
from torch import nn
from pcdet.models.model_utils.sampler_utils import SAMPLERS
from pcdet.models.model_utils.graph_utils import GRAPHS
from pcdet.models.model_utils.grouper_utils import GROUPERS
from pcdet.models.model_utils.fusion_utils import FUSIONS
from pcdet.models import blocks
from .assigners import ASSIGNERS

class DownBlockTemplate(nn.Module):
    def __init__(self,
                 block_cfg,
                 sampler_cfg,
                 graph_cfg,
                 assigner_cfg=None,
                 grouper_cfg=None,
                 fusion_cfg=None):
        super().__init__()
        if sampler_cfg is not None:
            sampler = SAMPLERS[sampler_cfg.pop("TYPE")]
            self.sampler = sampler(
                               runtime_cfg=None,
                               model_cfg=sampler_cfg,
                           )
        
        if graph_cfg is not None:
            graph = GRAPHS[graph_cfg["TYPE"]]
            self.graph = graph(
                             runtime_cfg=None,
                             model_cfg=graph_cfg,
                         )
        
        if grouper_cfg is not None:
            grouper = GROUPERS[grouper_cfg.pop("TYPE")]
            self.grouper = grouper(
                               runtime_cfg=None,
                               model_cfg=grouper_cfg,
                           )

        if assigner_cfg is not None:
            assigner = ASSIGNERS[assigner_cfg["TYPE"]]
            self.kernel_assigner = assigner(
                                       assigner_cfg=assigner_cfg,
                                   )
        
        if fusion_cfg is not None:
            fusion = FUSIONS[fusion_cfg.pop("TYPE")]
            self.fusion = fusion(
                              runtime_cfg=None,
                              model_cfg=fusion_cfg,
                          )

        norm_cfg = block_cfg.get("NORM_CFG", None)
        if norm_cfg is not None:
            if 'OUTPUT_CHANNEL' in block_cfg:
                output_channel = block_cfg["OUTPUT_CHANNEL"]
            elif 'MLP_CHANNELS' in block_cfg:
                output_channel = block_cfg["MLP_CHANNELS"][-1]
            self.norm = nn.BatchNorm1d(output_channel, **norm_cfg)
        else:
            self.norm = None

        act_cfg = block_cfg.get("ACTIVATION", None)
        if act_cfg is not None:
            if act_cfg == 'ReLU':
                self.act = nn.ReLU()
            else:
                raise ValueError("Unrecognized Activation {act_cfg}")
        else:
            self.act = None

    def forward(self, ref_bxyz, ref_feat):
        assert NotImplementedError

class UpBlockTemplate(nn.Module):
    def __init__(self,
                 block_cfg,
                 graph_cfg=dict(
                     TYPE="KNNGraph",
                     NUM_NEIGHBORS=3,
                 ),
                 assigner_cfg=None,
                ):
        super().__init__()
        
        if graph_cfg is not None:
            graph = GRAPHS[graph_cfg["TYPE"]]
            self.graph = graph(
                             runtime_cfg=None,
                             model_cfg=graph_cfg,
                         )
        
        if assigner_cfg is not None:
            assigner = ASSIGNERS[assigner_cfg["TYPE"]]
            self.kernel_assigner = assigner(
                                       assigner_cfg=assigner_cfg,
                                   )
            
        norm_cfg = block_cfg.get("NORM_CFG", None)
        if norm_cfg is not None:
            if 'OUTPUT_CHANNEL' in block_cfg:
                output_channel = block_cfg["OUTPUT_CHANNEL"]
            elif 'MLP_CHANNELS' in block_cfg:
                output_channel = block_cfg["MLP_CHANNELS"][-1]
            self.norm = nn.BatchNorm1d(output_channel, **norm_cfg)
        else:
            self.norm = None

        act_cfg = block_cfg.get("ACTIVATION", None)
        if act_cfg is not None:
            if act_cfg == 'ReLU':
                self.act = nn.ReLU()
            else:
                raise ValueError("Unrecognized Activation {act_cfg}")
        else:
            self.act = None
    
    def forward(self, ref_bxyz, ref_feat,
                query_bxyz, query_feat,
                e_ref, e_query):
        assert NotImplementedError

#class MessagePassingBlockTemplate(nn.Module):
#    def __init__(self, block_cfg, sampler_cfg, graph_cfg, kernel_assigner_cfg=None):
#        super().__init__()
#        if sampler_cfg is not None:
#            sampler = SAMPLERS[sampler_cfg.pop("TYPE")]
#            self.sampler = sampler(
#                               runtime_cfg=None,
#                               model_cfg=sampler_cfg,
#                           )
#        
#        assert graph_cfg is not None
#        graph = GRAPHS[graph_cfg["TYPE"]]
#        self.graph = graph(
#                         runtime_cfg=None,
#                         model_cfg=graph_cfg,
#                     )
#        
#    def forward(self, ref_bxyz, ref_feat):
#        assert NotImplementedError
