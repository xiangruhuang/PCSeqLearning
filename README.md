

# PCSeqLearning

This is the code repository for the paper:

Representation Learning for Object Detection from Unlabeled Point Cloud Sequences.

Xiangru Huang, Yue Wang, Vitor Guizilini, Rares Ambrus, Adrien Gaidon and Justin Solomon.

Conference on Robotic Learning (CoRL) 2022. 

The code is adapted from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). The code will be updated by September 5th to include full functionality. Currently, it can extract object cluster sequences from LiDAR point cloud sequences for Waymo Open Dataset.

# System Requirements

The code has been tested with the following (major) environment dependencies:
```
waymo-open-dataset-tf-2-5-0==1.4.3
torch==1.13.1+cu117
```

# Installation

Please check `Install.md` for instructions of setting up this repo. 

# Data preparation
Please follow the instructions [here](https://github.com/xiangruhuang/PCSeqLearning/blob/main/docs/GETTING_STARTED.md) for preparing Waymo Open Dataset.

# Demo
```
pip install polyscope
bash scripts/dist_train_multi.sh 0 cfgs/waymo_models/PCsequence/registration/cluster_tracking_TLS_multiradius_every8.yaml cfgs/dataset_configs/waymo/PCsequence/registration/all_sequence.yaml cfgs/optimizers/registration.yaml --vis_cfg_file cfgs/visualizers/waymo/PCsequence/registration/voxel_visualizer.yaml
```

The visualizer will show the extracted object clusters given an input Waymo point cloud sequence.
