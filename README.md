[![Static Badge](https://img.shields.io/badge/Powered_by-Kornia-green)](https://github.com/kornia/kornia) [![Static Badge](https://img.shields.io/badge/Matches_for-COLMAP-red)](https://github.com/colmap/colmap)

## DEEP-IMAGE-MATCHING

Multivew matcher for COLMAP. Support both deep-learning based and hand-crafted local features and matchers and export keypoints and matches directly in a COLMAP database. It supports both CLI and GUI.

Key features:

- [x] multiview
- [x] large format images
- [x] SOTA deep-learning and hand-crafted features
- [x] full combatibility with COLMAP
- [ ] Support for image rotations
- [ ] Compatibility with Agisoft Metashape
- [ ] support image retrieval with deep-learning local features

The repo is under construction but it already works with SuperGlue, LightGlue, ALIKE, DISK, Key.Net+HardNet8, ORB.

Feel free to collaborate!

## Install and run

Install in a conda environment:

```bash
conda create -n deep-image-matching python=3.9.17
conda activate deep-image-matching
```

Install pytorch. See [https://pytorch.org/get-started/locally/#linux-pip](https://pytorch.org/get-started/locally/#linux-pip)

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Example usage

Before running check options with `python ./main.py --help`, then:

```bash
python ./main.py cli  --config "superpoint+lightglue" --images "assets/imgs" --outs "output" --strategy "sequential" --overlap 2
```

See other examples in run.bat. If you want to customize detector and descpritor options, change default options in config.py.

To run with the GUI:

```bash
python ./main.py --gui
```

## Multiview tests

Supported extractors:

- [x] SuperPoint
- [x] DISK
- [x] ALIKE
- [x] ALIKED
- [ ] Superpoint free
- [ ] KeyNet + OriNet + HardNet8
- [x] ORB (opencv)
- [x] SIFT (opencv)

Matchers:

- [x] Lightglue (with Superpoint, Disk and ALIKED)
- [x] SuperGlue (with Superpoint)
- [ ] LoFTR
- [x] Nearest neighbor (with KORNIA DescriptorMatcher)

## TODO

- [x] extend to tile processing
- [ ] add kornia features
- [ ] manage image rotation
- [ ] add image retrieval with global descriptors
- [x] add GUI
- [ ] Add exporting to Metashape
- [ ] Add pycolmap compatibility
- [ ] Add visualization for extracted features and matches
- [ ] improve speed
- [ ] Add tests, documentation and examples
