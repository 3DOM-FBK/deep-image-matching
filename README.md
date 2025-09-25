<div align="center">

[![Static Badge](https://img.shields.io/badge/Matches_for-COLMAP-red)](https://github.com/colmap/colmap) [![Static Badge](https://img.shields.io/badge/Matches_for-OpenMVG-red)](https://github.com/openMVG/openMVG) [![Static Badge](https://img.shields.io/badge/Matches_for-MICMAC-red)](https://github.com/micmacIGN/micmac) ![Static Badge](https://img.shields.io/badge/Matches_for-Metashape-red)

[![Static Badge](https://img.shields.io/badge/Powered_by-Kornia-green)](https://github.com/kornia/kornia) [![Static Badge](https://img.shields.io/badge/Powered_by-hloc-green)](https://github.com/kornia/kornia) [![GitHub Release](https://img.shields.io/github/v/release/3DOM-FBK/deep-image-matching)](https://github.com/3DOM-FBK/deep-image-matching/releases) [![Static Badge](https://img.shields.io/badge/docs-DeepImageMatcher-blue)](https://3dom-fbk.github.io/deep-image-matching/)

</div>

# DEEP-IMAGE-MATCHING

| SIFT                                                  | DISK                                                    | IMAGES ORIENTATION                                        | DENSE WITH ROMA                                     |
| ----------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------- |
| <img src='docs/assets/matches_sift.gif' height="100"> | <img src='docs/assets/matches_joined.gif' height="100"> | <img src='docs/assets/orientation_deep.gif' height="100"> | <img src='docs/assets/roma_dense.gif' height="100"> |

| SIFT                                                  | SUPERGLUE                                                 |
| ----------------------------------------------------- | --------------------------------------------------------- |
| <img src='docs/assets/temple_rsift.gif' height="165"> | <img src='docs/assets/temple_superglue.gif' height="165"> |

Multivew matcher for SfM software. Support both deep-learning based and hand-crafted local features and matchers and export keypoints and matches directly in a COLMAP database or to Agisoft Metashape by importing the reconstruction in Bundler format. Now, it supports both OpenMVG and MicMac. Feel free to collaborate!

While `dev` branch is more frequently updated, `master` is the default more stable branch and is updated from `dev` less frequently. If you are looking for the newest developments, please switch to `dev`.

For how to use DIM, check the <a href="https://3dom-fbk.github.io/deep-image-matching/">Documentation</a> (updated for the master branch).

**Please, note that `deep-image-matching` is under active development** and it is still in an experimental stage. If you find any bug, please open an issue. **For the licence of individual local features and matchers please refer to the authors' original projects**.

Key features:

- Multiview
- Large format images
- SOTA deep-learning and hand-crafted features
- Support for image rotations
- Compatibility with several SfM software
- Support image retrieval with deep-learning local features

| Supported Extractors               | Supported Matchers                                        |
| ---------------------------------- | --------------------------------------------------------- |
| &check; SuperPoint                 | &check; Lightglue (with Superpoint, Disk, and ALIKED)     |
| &check; DISK                       | &check; SuperGlue (with Superpoint)                       |
| &#x2610; Superpoint free           | &check; Nearest neighbor (with KORNIA Descriptor Matcher) |
| &check; ALIKE                      | &check; LoFTR (only GPU)                                  |
| &check; ALIKED                     | &check; SE2-LoFTR (no tiling and only GPU)                |
| &check; KeyNet + OriNet + HardNet8 | &check; RoMa                                              |
| &check; DeDoDe (only GPU)          | &#x2610; GlueStick                                        |
| &check; SIFT (from Opencv)         |
| &check; ORB (from Opencv)          |

| Supported SfM software                        |
| --------------------------------------------- |
| &check; COLMAP                                |
| &check; OpenMVG                               |
| &check; MICMAC                                |
| &check; Agisoft Metashape                     |
| &check; Software that supports bundler format |

## Colab demo and notebooks

Want to run on a sample dataset? ➡️ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3DOM-FBK/deep-image-matching/blob/dev/notebooks/colab_run_from_bash_example.ipynb)

Want to run on your images? ➡️ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3DOM-FBK/deep-image-matching/blob/dev/notebooks/colab_run_from_bash_custom_images.ipynb)

DIM can also be utilized as a library instead of being executed through the Command Line Interface (refer to the `Usage Instructions`).

For quick examples, see:

- `demo.py` - Simple script demonstrating the basic workflow
- `demo.ipynb` - Interactive notebook version of the demo
- `notebooks/sfm_pipeline.ipynb` - Complete SfM pipeline with detailed explanations

## Local Installation

For installing deep-image-matching, we recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable package management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv --python 3.9
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then, you can install deep-image-matching using uv:

```bash
uv pip install -e .
```

This command will install the package in editable mode, allowing you to modify the source code and see changes immediately without needing to reinstall. If you want to use deep-image-matching as a non-editable library, you can also install it without the `-e` flag.

This will also install `pycolmap` as a dependency, which is required for running the 3D reconstruction.
If you have any issues with `pycolmap`, you can manually install it following the official instructions [here](https://colmap.github.io/pycolmap/index.html).

To verify that deep-image-matching is correctly installed, you can try to import the package in a Python shell:

```python
import deep_image_matching as dim
```

To test most of the functionality, run the tests to check if deep-image-matching is correctly installed, run:

```bash
uv pytest tests
```

For more information, check the [documentation](https://3dom-fbk.github.io/deep-image-matching/installation/).

### Why uv?

This project has migrated from conda/pip to [uv](https://docs.astral.sh/uv/) for dependency management. Benefits include:

- Faster installation: uv is significantly faster than pip for dependency resolution and installation
- Better dependency resolution: More reliable resolution of complex dependency trees
- Lockfile support: `uv.lock` ensures reproducible installations across different environments
- Integrated tooling: Built-in support for virtual environments, Python version management, and project building
- Cross-platform consistency: Better support for different operating systems and architectures

### Conda/pip installation

If you have any issue with uv, you prefer to have a global installation of DIM, or you have any other problem with the installation, you can use conda/manba to create an environment and install DIM from source using pip:

```bash
git clone https://github.com/3DOM-FBK/deep-image-matching.git
cd deep-image-matching

conda create -n deep-image-matching python=3.9
conda activate deep-image-matching
pip install -e .
```

### Docker Installation

For Docker installation, see the [Docker Installation](https://3dom-fbk.github.io/deep-image-matching/installation#using-docker/) section in the documentation.

## Usage instructions

### Quick start with the demo

For a quick start, check out the `demo.py` script or `demo.ipynb` notebook that demonstrate basic usage with the example dataset:

```bash
python demo.py --dir assets/example_cyprus --pipeline superpoint+lightglue
```

The demo runs the complete pipeline from feature extraction to 3D reconstruction using the provided example dataset.

A similar demo example is also available as a notebook in `demo.ipynb`.

### Command Line Interface

Use the following command to see all the available options from the CLI:

```bash
python -m deep_image_matching --help
```

For example, to run the matching with SuperPoint and LightGlue on the example_cyprus dataset:

```bash
python -m deep_image_matching --dir assets/example_cyprus --pipeline superpoint+lightglue
```

The `--dir` parameter defines the processing directory, where all the results will be saved. This directory must contain a subfolder named **images** with all the images to be processed.

### Library usage

Deep-image-matching can also be used as a Python library. For a comprehensive example showing the complete SfM pipeline, see `notebooks/sfm_pipeline.ipynb`.

For detailed usage instructions and configurations, refer to the [documentation](https://3dom-fbk.github.io/deep-image-matching/getting_started).

<!-- To run the GUI, you can use the following command:

```bash
python main.py --gui
``` -->

## Advanced usage

For advanced usage, please refer to the [documentation](https://3dom-fbk.github.io/deep-image-matching/) and/or check the `scripts` directory.

### Merging databases with different local features

To run the matching with different local features and/or matchers and marging together the results, you can use scripts in the `./scripts` directory for merging the COLMAP databases.

```bash
python ./join_databases.py --help
python ./join_databases.py --input path/to/dir/with/databases --output path/to/output/dir
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

## To Do List

See the [TODO list](notes.md) for the list of features and improvements that are planned for the future.

## References

If you find the repository useful for your work consider citing the papers:

```bibtex
@article{morelli2024_deep_image_matching,
  AUTHOR = {Morelli, L. and Ioli, F. and Maiwald, F. and Mazzacca, G. and Menna, F. and Remondino, F.},
  TITLE = {DEEP-IMAGE-MATCHING: A TOOLBOX FOR MULTIVIEW IMAGE MATCHING OF COMPLEX SCENARIOS},
  JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  VOLUME = {XLVIII-2/W4-2024},
  YEAR = {2024},
  PAGES = {309--316},
  DOI = {10.5194/isprs-archives-XLVIII-2-W4-2024-309-2024}
}
```

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
@article{ioli2024,
  title={Deep Learning Low-cost Photogrammetry for 4D Short-term Glacier
Dynamics Monitoring},
  author={Ioli, Francesco and Dematteis, Nicolò and Giordan, Daniele and Nex, Francesco and Pinto Livio},
  journal={PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science},
  year={2024},
  DOI = {10.1007/s41064-023-00272-w}
}
```

Depending on the options used, consider citing the corresponding work of:

- [KORNIA](https://github.com/kornia/kornia)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [COLMAP](https://github.com/colmap/colmap)
- [OpenMVG](https://github.com/openMVG/openMVG)
- [MICMAC](https://github.com/micmacIGN/micmac)
- used local features and matchers
