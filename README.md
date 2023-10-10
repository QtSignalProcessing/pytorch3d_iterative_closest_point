# pytorch3d_iterative_closest_point
Source code of Pytorch3D ICP

# Motivation:
I failed to compile the whole PyTorch3D but I only need ICP related functions. So I cropped out pieces of code from the original [PyTorch](https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d) repo.

# Prequiries
PyTorch with cuda support

# Install
The following line should work on both Windows and Linux

`python install.py`

# Usage
```
import torch
from pytorch3d import corresponding_points_alignment, iterative_closest_point
```
More details can be found from [corresponding_points_alignment](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.corresponding_points_alignment) and [iterative_closest_point](https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.iterative_closest_point)
