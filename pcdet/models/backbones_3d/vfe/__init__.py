from .mean_vfe import MeanVFE
#from .mask_embedding_vfe import MaskEmbeddingVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_vfe import DynamicVFE
from .repsurf_dynamic_vfe import RepsurfDynamicVFE
from .dynamic_pillar_vfe import DynamicPillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .hybrid_vfe import HybridVFE
from .hybrid_primitive_vfe import HybridPrimitiveVFE
from .temporal_vfe import TemporalVFE
from .plane_fitting import PlaneFitting

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
#    'MaskEmbeddingVFE': MaskEmbeddingVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynamicVFE': DynamicVFE,
    'RepsurfDynamicVFE': RepsurfDynamicVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'HybridVFE': HybridVFE,
    'HybridPrimitiveVFE': HybridPrimitiveVFE,
    'TemporalVFE': TemporalVFE,
    'PlaneFitting': PlaneFitting,
}
