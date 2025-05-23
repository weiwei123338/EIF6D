# EIF6D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for "EIF-6D: Dual-Stream Transformation Network with Implicit-Explicit Fusion for Category-Level 6D Pose Estimation"
## Overview

EIF6D introduces a novel approach to category-level 6D pose estimation,a dual-stream transformation network that integrates explicit and implicit collaborative deformation mechanisms, which achieves high-precision alignment between shape priors and observed objects at both point cloud level and feature level..

## Installation

### Environment Setup

We provide a comprehensive conda environment configuration to ensure all dependencies are properly installed:

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate eif6d
```

The environment includes:
- PyTorch 1.10.1 with CUDA 11.3 support
- Python 3.6.13

### Compilation

Compile the custom PointNet++ operations:

```bash
# Compile pointnet2 modules
cd model/pointnet2
python setup.py install
```

## Dataset Preparation

### REAL275 and CAMERA25 Datasets

For the REAL275 and CAMERA25 datasets, please follow:
1. [DPDN dataset instructions](https://github.com/JiehongLin/Self-DPDN)
2. [SPD dataset instructions](https://github.com/mentian/object-deformnet)

The datasets should be organized in the following structure:
```
data
├── CAMERA
│   ├── train
│   └── val
├── camera_full_depths
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
├── obj_models
│   ├── train
│   ├── val
│   ├── real_train
│   └── real_test
├── segmentation_results
│   ├── train_trainedwoMask
│   ├── test_trainedwoMask
│   └── test_trainedwithMask
└── mean_shapes.npy

```

## Training

Train the model from scratch using:

```bash
# For single GPU training
python trainPT2Net.py --gpus 0 --config config/PT2Net.yaml

# For multi-GPU training
python trainPT2Net.py --gpus 0,1 --config config/PT2Net.yaml
```

Training configurations can be adjusted in `config/PT2Net.yaml`.

## Evaluation

Evaluate the trained model on test datasets:

```bash
python test_PT2.py --config config/PT2Net.yaml
```

## Results

Our method achieves state-of-the-art performance on the REAL275 and CAMERA25 benchmarks:

| Method | 5° 2cm | 5° 5cm | 10° 2cm | 10° 5cm | IoU75 | 
|--------|--------|--------|---------|---------|-------|
| NOCS   | 7.2    | 10.0   | 13.8    | 25.2    | 30.1  |
|RBP-Pose| 38.2   | 48.1   | 63.1    | 79.2    | 67.8  | 
|  DPDN  | 46.0   | 50.7   | 70.4    | 78.4    | 76.0  |
|IST-Net | 47.5   | 53.4   | 72.1    | 80.5    | 76.6  |
|  Ours  | 50.6   | 57.4   | 72.7    | 81.5    | 77.1  |

## Citation



## Acknowledgements

- Our code is developed upon [DPDN](https://github.com/JiehongLin/Self-DPDN) and [IST-Net](https://github.com/CVMI-Lab/IST-Net).
- The dataset is provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019).


