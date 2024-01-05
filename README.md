[![Static Badge](https://img.shields.io/badge/Matches_for-COLMAP-red)](https://github.com/colmap/colmap)
![Static Badge](https://img.shields.io/badge/Matches_for-Metashape-blue) [![Static Badge](https://img.shields.io/badge/Powered_by-Kornia-green)](https://github.com/kornia/kornia) [![Static Badge](https://img.shields.io/badge/Powered_by-hloc-blue)](https://github.com/kornia/kornia)

## DEEP-IMAGE-MATCHING

| SIFT                                             | DISK                                               | IMAGES ORIENTATION                                   | DENSE WITH ROMA                                |
| ------------------------------------------------ | -------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| <img src='assets/matches_sift.gif' height="100"> | <img src='assets/matches_joined.gif' height="100"> | <img src='assets/orientation_deep.gif' height="100"> | <img src='assets/roma_dense.gif' height="100"> |

| SIFT                                             | SUPERGLUE                                            |            
| ------------------------------------             | ------------------------------------                 |
| <img src='assets/temple_rsift.gif' height="165"> | <img src='assets/temple_superglue.gif' height="165"> |

Multivew matcher for COLMAP. Support both deep-learning based and hand-crafted local features and matchers and export keypoints and matches directly in a COLMAP database or to Agisoft Metashape by importing the reconstruction in Bundler format. It supports both CLI and GUI. Feel free to collaborate!

**Please, note that `deep-image-matching` is under active development** and it is still in an experimental stage. If you find any bug, please open an issue.

Key features:

- [x] Multiview
- [x] Large format images
- [x] SOTA deep-learning and hand-crafted features
- [x] Full compatibility with COLMAP
- [x] Support for image rotations
- [x] Compatibility with Agisoft Metashape (only on Linux and MacOS by using pycolmap)
- [x] Support image retrieval with deep-learning local features

| Supported Extractors               | Supported Matchers                                        |
| ---------------------------------- | --------------------------------------------------------- |
| &check; SuperPoint                 | &check; Lightglue (with Superpoint, Disk, and ALIKED)     |
| &check; DISK                       | &check; SuperGlue (with Superpoint)                       |
| &check; ALIKE                      | &check; LoFTR                                             |
| &check; ALIKED                     | &check; SE2-LoFTR                                         |
| &#x2610; Superpoint free           | &check; Nearest neighbor (with KORNIA Descriptor Matcher) |
| &check; KeyNet + OriNet + HardNet8 | &check; RoMa                                              |
| &check; ORB (opencv)               | &#x2610; GlueStick                                        |
| &check; SIFT (opencv)              |
| &check; DeDoDe                     |

## Colab demo  ➡️ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gwdyLSiBIsq6e9_6g4X3VqucO0LBLuww?usp=sharing)
Want to run on a sample dataset? Try the Colab demo. Working to extend to other examples and visulize results..

## Installation

`Deep-image-matching` is tested on Ubuntu 22.04 and Windows 10 with `Python 3.9`. It is strongly recommended to have a NVIDIA GPU with at least 8GB of memory.

Please, note that deep-image-matching relies on [pydegensac](https://github.com/ducha-aiki/pydegensac) for the geometric verification of matches, which is only available for `Python <=3.9` on Windows. If you are using Windows, please, install `Python 3.9` (on Linux, you can also use `Pythom 3.10`).

For installing `deep-image-matching`, first create a conda environment:

```bash
conda create -n deep-image-matching python=3.9
conda activate deep-image-matching
pip install --upgrade pip
```

Clone the repository and install `deep-image-matching` in editable mode:

```bash
git clone https://github.com/3DOM-FBK/deep-image-matching.git
cd deep-image-matching
pip install -e .
```

If you run into any troubles installing Pytorch (and its related packages, such as Kornia), please check the official website ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) and follow the instructions for your system and CUDA architecture. Then, try to install again deep-image-matching.

For automatize 3D reconstruction, DEEP-IMAGE-MATCHING uses [pycolmap](https://github.com/colmap/pycolmap), which is only available in [pypi](https://pypi.org/project/pycolmap/) for Linux and macOS.
If you are using Windows, you can use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) for installing pycolmap (please refer to issue [#34](https://github.com/colmap/pycolmap/issues/34) in pycolmap repo).

Pycolmap is needed for running directly the 3D reconstruction (without the need to use COLMAP by GUI or CLI) and to export the reconstruction in Bundler format for importing into Metashape. Pycolmap is alse needed to create cameras from exif metadata in the COLMAP database.
If pycolmap is not installed, deep-image-matching will still work and it will export the matches in a COLMAP SQlite databse, which can be opened by COLMAP GUI or CLI to run the 3D reconstruction.

If you are on Linux or macOS, you can install pycolmap with:

```bash
pip install pycolmap
```

## Usage instructions

You can run deep-image-matching with the CLI or with the GUI.

All the configurations (that are used both from the CLI and the GUI) are in `config.py`.
There are two main configuration in `config.py`: `conf_general` and `confs`.

- `conf_general` contains some general configuration that is valid for all the combinations of local features and matchers, including the option to run the matching by tiles, run it on full images or on downsampled images, and the options for the geometric verification.

  ```python
    conf_general = {
      "quality": Quality.HIGH, -> `Control the resolution of the images, where HIGH = full-res; MEDIUM = half res; LOW = 1/4 res; HIGHEST = 2x res`
      "tile_selection": TileSelection.PRESELECTION, -> `Control the tiling approach. Options are NONE: disable tiling; PRESELECTION: divide the images into regular tiles and select the tiles to be matched by a low-resolution preselection (suggested for large images); GRID: divide images into regular tiles and match only tiles at the same location in the grid (e.g., 1-1, 2-2 etc); EXHAUSTIVE: match all the tiles with all the tiles. (slow)`
      "tile_size": [3, 3], -> `Define the tile grid as [number of rows, number of columns] of the grid.`
      "tile_overlap": 0, -> `Optionally, overlap tiles by a certain amount of pixels`
      "geom_verification": GeometricVerification.PYDEGENSAC, -> `Enable or disable Geometric Verification. Options are: NONE: disabled; PYDEGENSAC: use pydegensac; MAGSAC: use OpenCV geometric verification with MAGSAC.`
      "gv_threshold": 4, -> `Threshold [px] for the geometric verification`
      "gv_confidence": 0.9999, -> `Confidence value for the geometric verification`
      "preselection_size_max": 2000, -> `if tile_selection == TileSelection.PRESELECTION, define the resolution at which the images are downsampled to run the low-resolution tile preselection.`
    }
  ```

- `confs` is a dictionary that contains all the possibile combinations of local feature extrators and matchers that can be used in deep-image-matching and their configuration. Each configuration is defined by a name (e.g., "superpoint+lightglue") and it must be a dictionary containing two sub-dictionaries for the 'extractor' and the 'matcher'.

  Each subdictionary contains the name of the extractor or the matcher and then a series of optional parameters to to be passed to the extractor or matcher. Please refer to the specific implementation of the Extractor or Matcher (located in the folders `src/deep_image_matching/extractors` or `src/deep_image_matching/matchers` for a list of all the possible options.

  <details>

  <summary>Show dictionary</summary>

  ```python
  confs = {
      "superpoint+lightglue": {
          "extractor": {
              "name": "superpoint",
              "keypoint_threshold": 0.0001,
              "max_keypoints": 4096,
          },
          "matcher": {
              "name": "lightglue",
              "n_layers": 9,
              "depth_confidence": -1,  # 0.95,  # early stopping, disable with -1
              "width_confidence": -1,  # 0.99,  # point pruning, disable with -1
              "filter_threshold": 0.5,  # match threshold
          },
      },
      "aliked+lightglue": {
          "extractor": {
              "name": "aliked",
              ...
          },
          "matcher": {
              "name": "lightglue",
              ...
          },
      },
      "orb+kornia_matcher": {
          "extractor": {
              "name": "orb",
              ...
          },
          "matcher": {
              "name": "kornia_matcher",
              ...
          },
      },
    }
  ```

  </details>

From both the CLI and GUI you can select a configuration by its name (e.g., "superpoint+lightglue") and the corresponding configuration will be used.

### CLI

Before running the CLI, check the options with `python ./main.py --help`.

The minimal required options are:

- `--images`: the path to the folder containing the images
- `--config`: the name of the configuration to use (e.g., "superpoint+lightglue")

Other additional options are:

- `--outs`: the path to the folder where the matches will be saved (default: `./output`)
- `--strategy`: the strategy to use for the matching (default: `sequential`)
- `--overlap`: if `strategy` is set to `sequential`, set the number of images that are sequentially matched to each image in the sequence (default: `1`)
- `--retrieval`: if `strategy` is set to `retrieval`, the global descriptor to use for image retrieval (default: `None`)
- `--upright`: if passed, try to find the best image rotation before running the matching
- `--force`: if the output folder already exists, overwrite it
- `-V`: enable verbose output

To run sequential matching with Superpoint+LighGlue, you can use the following command:

```bash
python ./main.py  --config superpoint+lightglue --images assets/example_cyprus --outs assets/output --strategy sequential --overlap 2
```

To run bruteforce matching with ALIKED+LightGlue and with image rotation, you can use the following command:

```bash
python ./main.py  --config aliked+lightglue --images assets/example_cyprus --strategy bruteforce --upright
```

See other examples in run.bat. If you want to customize detector and descpritor options, change default options in config.py.

### GUI

To run with the GUI:

```bash
python ./main.py --gui
```

![X4](assets/gui.png)

In the GUI, you can define the same parameters that are available in the CLI.
The GUI loads the available configurations from `config.py` and it shows them in the dropdown menu `choose available matching configuration`.

### Merging databases with different local features

To run the matching with different local features and/or matchers and marging together the results, you can use scripts in the `./scripts` directory for merging the COLMAP databases.
```bash
python ./join_databases.py --help
python ./join_databases.py --input assets/to_be_joined --output assets/to_be_joined
```

## TODO:

- [x] Tile processing for high resolution images
- [x] Manage image rotations
- [x] Add image retrieval with global descriptors
- [x] add GUI
- [x] Add pycolmap compatibility
- [x] Add exporting to Bundler format ready for importing into Metashape (only on Linux and MacOS by using pycolmap)
- [ ] Add visualization for extracted features and matches
- [ ] Improve speed
- [ ] Autoselect tiling grid in order to fit images in GPU memory
- [ ] Add tests, documentation and examples (e.g. colab, ..)
- [ ] Apply masks during feature extraction
- [ ] Check scripts
- [ ] Integrate support for Pix4D [Open Photogrammetry Format](https://github.com/Pix4D/opf-spec)

## References

If you find the repository useful for your work consider citing the papers:

```bibtex
@article{morelli2022photogrammetry,
  title={PHOTOGRAMMETRY NOW AND THEN--FROM HAND-CRAFTED TO DEEP-LEARNING TIE POINTS--},
  author={Morelli, Luca and Bellavia, Fabio and Menna, Fabio and Remondino, Fabio},
  journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={48},
  pages={163--170},
  year={2022},
  publisher={Copernicus GmbH}
}
```

```bibtex
@article{ioli2023replicable,
  title={A Replicable Open-Source Multi-Camera System for Low-Cost 4d Glacier Monitoring},
  author={Ioli, F and Bruno, E and Calzolari, D and Galbiati, M and Mannocchi, A and Manzoni, P and Martini, M and Bianchi, A and Cina, A and De Michele, C and others},
  journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={48},
  pages={137--144},
  year={2023},
  publisher={Copernicus GmbH}
}
```

Depending on the options used, consider citing the corresponding work of [KORNIA](https://github.com/kornia/kornia), [HLOC](https://github.com/cvg/Hierarchical-Localization), and local features.
