<div align="center">
 
 [![Static Badge](https://img.shields.io/badge/Matches_for-COLMAP-red)](https://github.com/colmap/colmap)
 ![Static Badge](https://img.shields.io/badge/Matches_for-Metashape-blue) [![Static Badge](https://img.shields.io/badge/Powered_by-Kornia-green)](https://github.com/kornia/kornia) [![Static Badge](https://img.shields.io/badge/Powered_by-hloc-blue)](https://github.com/kornia/kornia)
 
  [![GitHub Release](https://img.shields.io/github/v/release/3DOM-FBK/deep-image-matching)](https://github.com/3DOM-FBK/deep-image-matching/releases) [![Static Badge](https://img.shields.io/badge/docs-DeepImageMatching-blue
 )](https://3dom-fbk.github.io/deep-image-matching/)
 
</div>

# DEEP-IMAGE-MATCHING

| SIFT                                             | DISK                                               | IMAGES ORIENTATION                                   | DENSE WITH ROMA                                |
| ------------------------------------------------ | -------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| <img src='docs/assets/matches_sift.gif' height="100"> | <img src='docs/assets/matches_joined.gif' height="100"> | <img src='docs/assets/orientation_deep.gif' height="100"> | <img src='docs/assets/roma_dense.gif' height="100"> |

| SIFT                                             | SUPERGLUE                                            |
| ------------------------------------------------ | ---------------------------------------------------- |
| <img src='docs/assets/temple_rsift.gif' height="165"> | <img src='docs/assets/temple_superglue.gif' height="165"> |


Multivew matcher for SfM software. Support both deep-learning based and hand-crafted local features and matchers and export keypoints and matches directly in a COLMAP database or to Agisoft Metashape by importing the reconstruction in Bundler format. It supports both CLI and GUI. Feel free to collaborate!

Check the documentation at <a href="https://3dom-fbk.github.io/deep-image-matching/">Docs</a>.
If you want the most recent features, you should check and use the `dev` branch.

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
| &check; ALIKED                     | &#x2610; SE2-LoFTR                                        |
| &#x2610; Superpoint free           | &check; Nearest neighbor (with KORNIA Descriptor Matcher) |
| &check; KeyNet + OriNet + HardNet8 | &check; RoMa                                              |
| &check; ORB (opencv)               | &#x2610; GlueStick                                        |
| &check; SIFT (opencv)              |
| &check; DeDoDe                     |

| Supported SfM software                        |
| --------------------------------------------- |
| &check; COLMAP                                |
| &#x2610; OpenMVG                               |
| &#x2610; MICMAC                               |
| &check; Agisoft Metashape                     |
| &check; Software that supports bundler format |

## Colab demo 

Want to run on a sample dataset? ➡️ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3DOM-FBK/deep-image-matching/blob/master/notebooks/colab_run_from_bash_example.ipynb)

Want to run on your images? ➡️ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3DOM-FBK/deep-image-matching/blob/master/notebooks/colab_run_from_bash_custom_images.ipynb)

## Installation

For installing deep-image-matching, first create a conda environment:

```bash
conda create -n deep-image-matching python=3.9
conda activate deep-image-matching
pip install --upgrade pip
```

Clone the repository and install deep-image-matching in editable mode:

```bash
git clone https://github.com/3DOM-FBK/deep-image-matching.git
cd deep-image-matching
pip install -e .
```

Install pycolmap:

```bash
pip install pycolmap
```

As [pycolmap](https://github.com/colmap/pycolmap) was released on PyPi only for Linux and macOS (up to version 0.4.0), it is not installed by default with deep-image-matching. 
From version 0.5.0, pycolmap can be installed on Windows too. However, it needs some testing before including in dependencies of deep-image-matching, as there are some errors on Windows that are blocking deep_image_matching pipeline (while it works completely fine on Linux).

For more information, check the [documentation](https://3dom-fbk.github.io/deep-image-matching/installation/).

## Usage instructions

You can run deep-image-matching from the command line or from the GUI.


Use the following command to see all the available options from the CLI:

```bash
python main.py --help
```

For example, to run the matching with SuperPoint and LightGlue on a dataset, you can use the following command:

```bash
python main.py --dir assets/example_cyprus --pipeline superpoint+lightglue
```

The `--dir` parameter defines the processing directory, where all the results will be saved. This directory must constain a subfolder named **images** in which all the images must be stored. We currentely do not support image retrieval from multiple directories or subdirectories, but we may add this feature in the future. 

The `--pipeline` parameter defines the combaination of local feature extractor and matcher. 

For all the usage instructions and configurations, refer to the [documenation](https://3dom-fbk.github.io/deep-image-matching/getting_started) or check the example notebooks.

To run the GUI, you can use the following command:

```bash
python main.py --gui
```

Please note that the GUI is currently out of date and should be upgraded to the latest version of Deep-Image-Matching. We will update it soon, thank you for your patience.

## Advanced usage

For advanced usage, please refer to the [documentation](https://3dom-fbk.github.io/deep-image-matching/) and/or check the `scripts` directory.

### Merging databases with different local features

To run the matching with different local features and/or matchers and marging together the results, you can use scripts in the `./scripts` directory for merging the COLMAP databases.

```bash
python ./join_databases.py --help
python ./join_databases.py --input assets/to_be_joined --output docs/assets/to_be_joined
```

### Exporting the solution to Metashape

To export the solution to Metashape, you can export the COLMAP database to Bundler format and then import it into Metashape.
This can be done from Metashape GUI, by first importing the images and then use the function `Import Cameras` (File -> Import -> Import Cameras) to select Bundler file (e.g., bundler.out) and the image list file (e.g., bundler_list.txt).

Alternatevely, you can use the `export_to_metashape.py` script to automatically create a Metashape project from a reconstruction saved in Bundler format.
The script `export_to_metashape.py` takes as input the solution in Bundler format and the images and it exports the solution to Metashape.
It requires to install Metashape as a Python module in your environment and to have a valid license.
Please, refer to the instructions at [https://github.com/franioli/metashape](https://github.com/franioli/metashape).

## How to contribute

Any contribution to this repo is really welcome!
If you want to contribute to the project, please, check the [contributing guidelines](./CONTRIBUTING.md).

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
- [x] Add tests, documentation and examples (e.g. colab, ..)
- [ ] Apply masks during feature extraction
- [ ] Integrate support for Pix4D [Open Photogrammetry Format](https://github.com/Pix4D/opf-spec)
- [ ] Work with submodules
- [ ] Automatically download weights for all the models
- [x] Cleanup repository to removed large files from Git history
- [x] Update README CLI options


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
