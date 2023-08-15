from .polyscope_visualizer import PolyScopeVisualizer, to_numpy_cpu
from .plotly_visualizer import PlotlyVisualizer
from .geometry_visualizer import GeometryVisualizer

__all__ = {
    'PolyScopeVisualizer': PolyScopeVisualizer,
    'PlotlyVisualizer': PlotlyVisualizer,
    'GeometryVisualizer': GeometryVisualizer,
}

def build_visualizer(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
