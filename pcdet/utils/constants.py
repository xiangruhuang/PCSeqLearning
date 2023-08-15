import numpy as np
seg_class_colors= np.array([
        [0.3,0.3,0.3], # 0
        [1,0,0],
        [1,0,0],
        [0.6, 0.1, 0.8], # 3
        [0.2, 0.1, 0.9],
        [0.5, 1, 0.5],
        [0,1,0], # 6
        [0.8,0.8,0.8],
        [0.0, 0.8, 0.8],
        [0.05, 0.05, 0.3],
        [0.8, 0.6, 0.2], # 10 
        [0.5, 1, 0.5],
        [0.5, 1, 0.5], # 12
        [0.2, 0.5, 0.8],
        [0.0, 0.8, 0],
        [0.0, 0.0, 0.0],
        [1, 1, 1], # 16
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 1], # 18
        [0., 1, 0.3],
        [0.9, 0.35, 0.2],
        [0.9, 0.6, 0.2], # 21
    ]).astype(np.float32)

CONSTANTS = dict(
    seg_class_colors=seg_class_colors
)
