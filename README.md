[![Static Badge](https://img.shields.io/badge/Matches_for-COLMAP-red)](https://github.com/colmap/colmap) [![Static Badge](https://img.shields.io/badge/Powered_by-Kornia-green)](https://github.com/kornia/kornia) [![Static Badge](https://img.shields.io/badge/Powered_by-hloc-blue)](https://github.com/kornia/kornia)

## DEEP-IMAGE-MATCHING


|   SIFT                      |   DISK                      |   DISK                           |
| ----------------------------| ----------------------------| ---------------------------------|
| ![X1](assets/nadar_sift_matches.png) | ![X2](assets/nadar_disk_matches.png) | ![X3](assets/nadar_disk.png) |


Multivew matcher for COLMAP. Support both deep-learning based and hand-crafted local features and matchers and export keypoints and matches directly in a COLMAP database or to Agisoft Metashape by importing the reconstruction in Bundler format. It supports both CLI and GUI.

Key features:

- [x] Multiview
- [x] Large format images
- [x] SOTA deep-learning and hand-crafted features
- [x] Full combatibility with COLMAP
- [x] Support for image rotations
- [x] Compatibility with Agisoft Metashape
- [ ] Support image retrieval with deep-learning local features

The repo is under construction but it already works with SuperPoint, DISK, ALIKE, ALIKES, ORB and SIFT local features and LightGlue, SuperGlue and nearest neighbor matchers.

Feel free to collaborate!

## Install and run

Install in a conda environment:

```bash
conda create -n deep-image-matching python=3.10
conda activate deep-image-matching
```

Install pytorch. See [https://pytorch.org/get-started/locally/#linux-pip](https://pytorch.org/get-started/locally/#linux-pip)

```bash
python -m pip install --upgrade pip
pip install -e .
```

Install hloc (https://github.com/cvg/Hierarchical-Localization/tree/master):
```
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
git submodule update --init --recursive
```

## Example usage

### Sequential matching with LighGlue

Before running check options with `python ./main.py --help`, then:

```bash
python ./main.py  --config superpoint+lightglue --images assets/example_images --outs assets/output --strategy sequential --overlap 1
```

See other examples in run.bat. If you want to customize detector and descpritor options, change default options in config.py.

To run with the GUI:

```bash
python ./main.py --gui
```

![X4](assets/gui.png)

### Merging databases with different local features

See scripts in the `./scripts` dir

## Multiview tests

Supported extractors:

- [x] SuperPoint
- [x] DISK
- [x] ALIKE
- [x] ALIKED
- [ ] Superpoint free
- [x] KeyNet + OriNet + HardNet8
- [x] ORB (opencv)
- [x] SIFT (opencv)

Matchers:

- [x] Lightglue (with Superpoint, Disk and ALIKED)
- [x] SuperGlue (with Superpoint)
- [x] LoFTR
- [x] Nearest neighbor (with KORNIA Descriptor Matcher)
- [ ] GlueStick
- [X] RoMa

## TODO:

- [x] Tile processing for high resolution images
- [x] Manage image rotations
- [ ] Add image retrieval with global descriptors
- [x] add GUI
- [x] Add pycolmap compatibility
- [x] Add exporting to Bundler format ready for importing into Metashape (only on linux with pycolmap)
- [ ] Add visualization for extracted features and matches
- [ ] Improve speed (parallization and threading)
- [ ] Autoselect tiling grid in order to fit images in GPU memory
- [ ] Add tests, documentation and examples
- [ ] Apply mask during feature extraction
- [ ] Check scripts