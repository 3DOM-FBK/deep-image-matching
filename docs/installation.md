# Installation

`Deep-image-matching` is tested on Ubuntu 22.04 and Windows 10 with `Python 3.9`. It is strongly recommended to have a NVIDIA GPU with at least 8GB of memory.

Please, note that deep-image-matching relies on [pydegensac](https://github.com/ducha-aiki/pydegensac) for the geometric verification of matches, which is only available for `Python <=3.9` on Windows. If you are using Windows, please, install `Python 3.9` (on Linux, you can also use `Python 3.10`).

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
