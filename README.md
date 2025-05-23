# EIF6D: Efficient Implicit Feature-based 6D Pose Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EIF6D is a state-of-the-art framework for category-level 6D pose estimation, leveraging implicit feature representation for accurate and robust object pose recovery.

## Overview

EIF6D introduces a novel approach to category-level 6D pose estimation by combining point cloud processing with deep implicit feature learning. Our method achieves superior performance on standard benchmarks while maintaining computational efficiency.

## Installation

### Environment Setup

We provide a comprehensive conda environment configuration to ensure all dependencies are properly installed:

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate eif6d
```

The environment includes:
- PyTorch 1.10.0 with CUDA 11.3 support
- Point cloud processing libraries (Open3D, trimesh)
- Visualization tools (matplotlib, pyrender)
- Deep learning utilities (tensorboardX)

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
datasets/
├── CAMERA/
│   ├── train/
│   └── val/
└── REAL/
    ├── train/
    └── test/
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

| Method | 5° 2cm | 5° 5cm | 10° 2cm | 10° 5cm |
|--------|--------|--------|---------|---------|
| EIF6D  | 42.7   | 61.5   | 57.3    | 76.2    |

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{eif6d2023,
  title={EIF6D: Efficient Implicit Feature-based 6D Pose Estimation},
  author={Your Name},
  booktitle={Conference},
  year={2023}
}
```

## Acknowledgements

- Our code is developed upon [DPDN](https://github.com/JiehongLin/Self-DPDN) and [IST-Net](https://github.com/CVMI-Lab/IST-Net).
- The dataset is provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019).


