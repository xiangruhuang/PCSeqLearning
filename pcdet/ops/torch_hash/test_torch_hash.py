import torch
from .torch_hash_cuda import (
    track_graphs_cpu
)

num_graphs=5
points = torch.randn(50, 3)
graph_idx = torch.randint(0, num_graphs, [50]).long()
print(graph_idx)
track_graphs_cpu(points, graph_idx, num_graphs)
import ipdb; ipdb.set_trace()
