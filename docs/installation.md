# Installation

## Requirements

Deep-image-matching is tested on Ubuntu 22.04, Windows 10, and macOS with `Python 3.9`. It is strongly recommended to have a NVIDIA GPU with at least 8GB of memory for optimal performance with deep learning models.

Due to dependencies issues, it is recommended to use `Python 3.9` on Windows and macOS, while on Linux you can also use `Python 3.10` (see [pydegensac](#pydegensac)).

All the dependencies are listed in the `pyproject.toml` file and managed with `uv.lock`.

## Installation with uv (Recommended)

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

This command will install the package in editable mode. If you want to use deep-image-matching as a non-editable library, you can install it without the `-e` flag.

This will also install `pycolmap` as a dependency, which is required for running the 3D reconstruction.

To verify that deep-image-matching is correctly installed, you can try to import the package in a Python shell:

```python
import deep_image_matching as dim
```

To test most of the functionality, run:

```bash
uv run pytest tests
```

### Why uv?

This project has migrated from conda/pip to [uv](https://docs.astral.sh/uv/) for dependency management. Benefits include:

- Faster installation: uv is significantly faster than pip for dependency resolution and installation
- Better dependency resolution: More reliable resolution of complex dependency trees
- Lockfile support: `uv.lock` ensures reproducible installations across different environments
- Integrated tooling: Built-in support for virtual environments, Python version management, and project building
- Cross-platform consistency: Better support for different operating systems and architectures

## Conda/pip installation

If you have any issue with uv, you prefer to have a global installation of DIM, or you have any other problem with the installation, you can use conda/mamba to create an environment and install DIM from source using pip:

```bash
git clone https://github.com/3DOM-FBK/deep-image-matching.git
cd deep-image-matching

conda create -n deep-image-matching python=3.9
conda activate deep-image-matching
pip install -e .
```

## Docker Installation

For Docker installation, see the [Docker Installation](#using-docker) section below.

## Notes and troubleshooting

#### Pytorch

If you run into any troubles installing Pytorch (and its related packages, such as Kornia), please check the official website ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) and follow the instructions for your system and CUDA architecture. Then, try to install again deep-image-matching.

#### Pydegensac

Deep-image-matching relies on [pydegensac](https://github.com/ducha-aiki/pydegensac) for the geometric verification of matches, which is only available for `Python <=3.9` on Windows. If you are using Windows or MacOS, please, use `Python 3.9`, on Linux, you can also use `Python 3.10`.

#### Pycolmap

Deep-image-matching uses [pycolmap](https://github.com/colmap/pycolmap) for automatize 3D reconstruction (without the need to use COLMAP by GUI or CLI) and to export the reconstruction in Bundler format for importing into Metashape.
Pycolmap is alse needed to create cameras from exif metadata in the COLMAP database.
If pycolmap is not installed, deep-image-matching will still work and it will export the matches in a COLMAP SQlite databse, which can be opened by COLMAP GUI or CLI to run the 3D reconstruction.

Up to version 0.4.0, pycolmap was avalable in [pypi](https://pypi.org/project/pycolmap/) only for Linux and MacOS, therefore it was not installed by default with
deep-image-matching to avoid errors on installations on Windows.

Recently, [pycolmap 0.5.0](https://github.com/colmap/pycolmap/releases/tag/v0.5.0) has been released on pypi also for Windows, so we are considering to add it as a dependency of deep-image-matching and it will be installed by default, but we need to carry out some tests before doing it.

Before, if you were using Windows, you could use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) for installing pycolmap (please refer to issue [#34](https://github.com/colmap/pycolmap/issues/34) in pycolmap repo).
However, now you can also try installing it with pip, it should work fine.

## Using Docker

If you prefer using Docker, first, build the image:

```bash
docker build --tag deep-image-matching .
```

Note that the first time you run the command, it will take a while to download the base image and install all the dependencies.

Once the image is built, you can run it with the following commands.

On Linux:

```bash
docker run --name run-deep-image-matching --mount type=bind,source=/home/username/data,target=/workspace/data --gpus all -it deep-image-matching
```

On Windows (please use Powershell):

```powershell
docker run --name run-deep-image-matching --mount type=bind,source=D:\data,target=/workspace/data --gpus all -it deep-image-matching
```

**Replace** `/home/username/data` (on Linux) or `D:\data` (on Windows) with the desired path for mounting a shared volume between the local OS and the docker container. Make sure to use absolute paths. This folder will be used to store all the input data (images) and outputs.

### Docker Options

Include the `--detach` option to run the container in background:

```bash
docker run --name run-deep-image-matching --mount type=bind,source=/home/username/data,target=/workspace/data --gpus all --detach -it deep-image-matching
```

Include the `--rm` option to remove container on exit:

```bash
docker run --name run-deep-image-matching --mount type=bind,source=/home/username/data,target=/workspace/data --gpus all --rm -it deep-image-matching
```

Once the container is running, you can then open the repo cloned inside the container directly in VSCode using `ctrl+alt+O` and selecting the option "attach to running container" (make sure to have the Docker extension installed in VSCode), then enjoy!

### Building from Different Branches

If you want to build the docker image with deep-image-matching and pycolmap from a branch different from `master`, you can use the following command:

```bash
docker build --tag deep-image-matching --build-arg BRANCH=dev .
```

### Docker troubleshooting

#### Linux

With Linux you may face some issues when running the container with the `--gpus all` option.

1. If you get the following error.

   ```bash
   docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
   nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown.
   ERRO[0000] error waiting for container: context canceled
   ```

   You should try to use `sudo` for building and running the docker command

   ```bash
   sudo docker build --tag deep-image-matching .
   sudo docker run --name running-deep-image-matching --mount type=bind,source=/home/username/data,target=/workspace/data --gpus all -it deep-image-matching
   ```

2. If you get the following error.

   ```bash
   docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
   ERRO[0000] error waiting for container: context canceled
   ```

   In order to use the `--gpus all` option, you need to have the NVIDIA Container Toolkit installed. Please refer to the official documentation [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian).
