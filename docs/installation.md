# Installation

## Requirements

Deep-image-matching is tested on Ubuntu 22.04 and Windows 10 with `Python 3.9`. It is strongly recommended to have a NVIDIA GPU with at least 8GB of memory.
Due to dependencies issues, it is recommended to use `Python 3.9` on Windows and MacOS, while on Linux you can also use `Python 3.10` (see [pydegensac](#pydegensac).

All the dependencies are listed in the `pyproject.toml` file.

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

Up to version 0.4.0, [pycolmap](https://github.com/colmap/pycolmap) was released on PyPi only for Linux and macOS.
Therefore, it was not included in the dependencies of deep-image-matching, so you need to install it manually.
From [version 0.5.0](https://github.com/colmap/pycolmap/releases/tag/v0.5.0), pycolmap can be installed on Windows too. However, it needs some testing before including in dependencies of deep-image-matching, as there are some errors on Windows that are blocking deep_image_matching pipeline (while it works completely fine on Linux).

## Notes and troubleshooting

### Pytorch

If you run into any troubles installing Pytorch (and its related packages, such as Kornia), please check the official website ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) and follow the instructions for your system and CUDA architecture. Then, try to install again deep-image-matching.

### Pydegensac
Deep-image-matching relies on [pydegensac](https://github.com/ducha-aiki/pydegensac) for the geometric verification of matches, which is only available for `Python <=3.9` on Windows. If you are using Windows or MacOS, please, use `Python 3.9`, on Linux, you can also use `Python 3.10`.

### Pycolmap

Deep-image-matching uses [pycolmap](https://github.com/colmap/pycolmap) for automatize 3D reconstruction (without the need to use COLMAP by GUI or CLI) and to export the reconstruction in Bundler format for importing into Metashape. 
Pycolmap is alse needed to create cameras from exif metadata in the COLMAP database.
If pycolmap is not installed, deep-image-matching will still work and it will export the matches in a COLMAP SQlite databse, which can be opened by COLMAP GUI or CLI to run the 3D reconstruction.

Up to version 0.4.0, pycolmap was avalable in [pypi](https://pypi.org/project/pycolmap/) only for Linux and MacOS, therefore it was not installed by default with
deep-image-matching to avoid errors on installations on Windows.

Recently, [pycolmap 0.5.0](https://github.com/colmap/pycolmap/releases/tag/v0.5.0) has been released on pypi also for Windows, so we are considering to add it as a dependency of deep-image-matching and it will be installed by default, but we need to carry out some tests before doing it.

Before, if you were using Windows, you could use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) for installing pycolmap (please refer to issue [#34](https://github.com/colmap/pycolmap/issues/34) in pycolmap repo).
However, now you can also try installing it with pip, it should work fine.
