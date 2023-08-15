import plotly
import plotly.graph_objs as go
import torch
from torch import nn
import numpy as np

def to_numpy_cpu(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        return a
    else:
        raise ValueError("Requiring Numpy or torch.Tensor")

class PlotlyVisualizer(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.range = model_cfg.get("RANGE", [-100, -100, -100, 100, 100, 100])
        self.range = np.array(self.range)

    def forward(self, batch_dict):
        plotly.offline.init_notebook_mode()
        point_bxyz = batch_dict['point_bxyz']
        for b in range(batch_dict['batch_size']):
            mask = point_bxyz[:, 0] == b
            point_xyz = point_bxyz[mask, 1:4]
            self.pointcloud('points', to_numpy_cpu(point_xyz))

    def pointcloud(self, name, point_xyz):
        trace = go.Scatter3d(
            x=point_xyz[:, 0],
            y=point_xyz[:, 1],
            z=point_xyz[:, 2],
            mode='markers',
            marker={
                'size': 1,
                'opacity': 0.8,
            }
        )
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
            scene=dict(
                xaxis=dict(title="x", range = [-100, 100]), #self.range[[0, 3]]),
                yaxis=dict(title="y", range = [-100, 100]), #self.range[[1, 4]]),
                zaxis=dict(title="z", range = [-100, 100]), #self.range[[2, 5]]),
                aspectmode='cube',
            ),
        )
        data = [trace]
        plot_figure = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(plot_figure)

