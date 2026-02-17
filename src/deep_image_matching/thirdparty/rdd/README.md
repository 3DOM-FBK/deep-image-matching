## RDD: Robust Feature Detector and Descriptor using Deformable Transformer (CVPR 2025)(IMC 2025 Prize Winner)

[Gonglin Chen](https://xtcpete.com/) · [Tianwen Fu](https://twfu.me/) · [Haiwei Chen](https://scholar.google.com/citations?user=LVWRssoAAAAJ&hl=en) · [Wenbin Teng](https://wbteng9526.github.io/) · [Hanyuan Xiao](https://corneliushsiao.github.io/index.html) · [Yajie Zhao](https://ict.usc.edu/about-us/leadership/research-leadership/yajie-zhao/)

[Project Page](https://xtcpete.github.io/rdd/) 

## Table of Contents
- [Updates](#updates)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Training](#training)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Updates
[02/11/2025] We updated RDD by replacing the backbone with ConvNeXt, achieving detector-free–level performance. The corresponding code is available in the v3 branch.

<table>
  <tr>
    <th></th>
    <th colspan="3">MegaDepth-1500</th>
    <th colspan="3">MegaDepth-View</th>
    <th colspan="3">Air-to-Ground</th>
  </tr>
  <tr>
    <td></td>
    <td>AUC 5&deg</td><td>AUC 10&deg</td><td>AUC 20&deg</td>
    <td>AUC 5&deg</td><td>AUC 10&deg</td><td>AUC 20&deg</td>
    <td>AUC 5&deg</td><td>AUC 10&deg</td><td>AUC 20&deg</td>
  </tr>
  <tr>
    <td>RDD-v3</td>
    <td>53.6</td><td>69.3</td><td>81.3</td>
    <td>56.9</td><td>71.9</td><td>82.3</td>
    <td>56.3</td><td>71.2</td><td>81.7</td>
  </tr>
  <tr>
    <td>RDD-v2</td>
    <td>52.4</td><td>68.5</td><td>80.1</td>
    <td>52.0</td><td>67.1</td><td>78.2</td>
    <td>45.8</td><td>58.6</td><td>71.0</td>
  </tr>
  <tr>
    <td>RDD-v1</td>
    <td>48.2</td><td>65.2</td><td>78.3</td>
    <td>38.3</td><td>53.1</td><td>65.6</td>
    <td>41.4</td><td>56.0</td><td>67.8</td>
  </tr>
  <tr>
    <td>RDD-v3+LG</td>
    <td>56.3</td><td>72.4</td><td>83.9</td>
    <td>60.3</td><td>74.2</td><td>84.4</td>
    <td>63.1</td><td>76.9</td><td>86.3</td>
  </tr>
  <tr>
    <td>RDD-v2+LG</td>
    <td>53.3</td><td>69.8</td><td>82.0</td>
    <td>59.0</td><td>74.2</td><td>84.0</td>
    <td>54.8</td><td>69.0</td><td>79.1</td>
  </tr>
  <tr>
    <td>RDD-v1+LG</td>
    <td>52.3</td><td>68.9</td><td>81.8</td>
    <td>54.2</td><td>69.3</td><td>80.3</td>
    <td>55.1</td><td>68.9</td><td>78.9</td>
  </tr>
</table>

[02/11/2025] Air-to-Ground training data is available [here](https://huggingface.co/datasets/xtcpete/air_ground). The aerial images are licensed under cc-by-4.0 and the ground images are sourced from [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/).

[06/06/2025] Evaluation code for ScanNet added. 
<table>
  <tr>
    <th></th>
    <th colspan="3">ScanNet-1500</th>
  </tr>
  <tr>
    <td></td>
    <td>AUC 5&deg</td><td>AUC 10&deg</td><td>AUC 20&deg</td>
  </tr>
  <tr>
    <td>RDD-v2</td>
    <td>13.7</td><td>29.3</td><td>45.3</td>
  </tr>
  <tr>
    <td>RDD-v2+LG</td>
    <td>20.2</td><td>38.6</td><td>55.8</td>
  </tr>
  <tr>
    <td>eLoFTR</td>
    <td>19.6</td><td>37.7</td><td>54.4</td>
  </tr>
</table>

[05/16/2025] SfM reconstruction through [COLMAP](https://github.com/colmap/colmap.git) added. We provide a ready-to-use [notebook](./demo_sfm.ipynb) for a simple example. Code adopted from [hloc](https://github.com/cvg/Hierarchical-Localization.git).

[05/12/2025] Training code and new weights released.

[05/12/2025] We have updated the training code compared to what was described in the paper. In the original setup, the RDD was trained on the MegaDepth and Air-to-Ground datasets by resizing all images to the training resolution. In this release, we retrained RDD on MegaDepth only, using a combination of resizing and cropping, a strategy used by [ALIKE](https://github.com/Shiaoming/ALIKE). This change significantly improves robustness.

## Installation

```bash
git clone --recursive https://github.com/xtcpete/rdd
cd rdd

# Create conda env
conda create -n rdd python=3.10 pip
conda activate rdd

# Install CUDA 
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
# Install torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# Install all dependencies
pip install -r requirements.txt
# Compile custom operations.
# You don't have to compile them to run RDD, but it is recommended for better performance.
cd ./RDD/models/ops
pip install -e .
```

We provide the [download link](https://drive.google.com/drive/folders/1QgVaqm4iTUCqbWb7_Fi6mX09EHTId0oA?usp=sharing) to:
  - the MegaDepth-1500 test set
  - the MegaDepth-View test set
  - the Air-to-Ground test set
  - the SacnNet-1500 test set
  - 2 pretrained models, RDD and LightGlue for matching RDD

Create and unzip downloaded test data to the `data` folder.

Create and add weights to the `weights` folder and you are ready to go.

## Usage
For your convenience, we provide a ready-to-use [notebook](./demo_matching.ipynb) for some examples.

### Inference

```python
from RDD.RDD import build

RDD_model = build()

output = RDD_model.extract(torch.randn(1, 3, 480, 640))
```

### Evaluation

Please note that due to the different GPU architectures and the stochastic nature of RANSAC, you may observe slightly different results; however, they should be very close to those reported in the paper. To reproduce the number in paper, use v1 weights instead.

Results can be visualized by passing argument --plot

**MegaDepth-1500**

```bash
# Sparse matching
python ./benchmarks/mega_1500.py

# Dense matching
python ./benchmarks/mega_1500.py --method dense

# LightGlue
python ./benchmarks/mega_1500.py --method lightglue
```

**MegaDepth-View**

```bash
# Sparse matching
python ./benchmarks/mega_view.py

# Dense matching
python ./benchmarks/mega_view.py --method dense

# LightGlue
python ./benchmarks/mega_view.py --method lightglue
```

**Air-to-Ground**

```bash
# Sparse matching
python ./benchmarks/air_ground.py

# Dense matching
python ./benchmarks/air_ground.py --method dense

# LightGlue
python ./benchmarks/air_ground.py --method lightglue
```

**ScanNet-1500**

```bash
# Sparse matching
python ./benchmarks/scannet_1500.py

# Dense matching
python ./benchmarks/scannet_1500.py --method dense

# LightGlue
python ./benchmarks/scannet_1500.py --method lightglue
```

### Training

1. Download MegaDepth dataset using [download.sh](./data/megadepth/download.sh) and megadepth_indices from [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets). Then the MegaDepth root folder should look like the following:
```bash
./data/megadepth/megadepth_indices # indices
./data/megadepth/depth_undistorted # depth maps
./data/megadepth/Undistorted_SfM # images and poses
./data/megadepth/scene_info # indices for training LightGlue
```
2. Then you can train RDD in two steps; Descriptor first
```bash
# distributed training with 8 gpus
python -m training.train --ckpt_save_path ./ckpt_descriptor --distributed --batch_size 32

# single gpu 
python -m training.train --ckpt_save_path ./ckpt_descriptor
```
and then Detector
```bash
python -m training.train --ckpt_save_path ./ckpt_detector --weights ./ckpt_descriptor/RDD_best.pth --train_detector --training_res 480
```

I am working on recollecting the Air-to-Ground dataset because of licensing issues.

## Citation
```
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Gonglin and Fu, Tianwen and Chen, Haiwei and Teng, Wenbin and Xiao, Hanyuan and Zhao, Yajie},
    title     = {RDD: Robust Feature Detector and Descriptor using Deformable Transformer},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6394-6403}
}
```


## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements

We thank these great repositories: [ALIKE](https://github.com/Shiaoming/ALIKE), [LoFTR](https://github.com/zju3dv/LoFTR), [DeDoDe](https://github.com/Parskatt/DeDoDe), [XFeat](https://github.com/verlab/accelerated_features), [LightGlue](https://github.com/cvg/LightGlue), [Kornia](https://github.com/kornia/kornia), and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and many other inspiring works in the community.

LightGlue is trained with [Glue Factory](https://github.com/cvg/glue-factory).

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0423C0075. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government. We would like to thank Yayue Chen for her help with visualization.
