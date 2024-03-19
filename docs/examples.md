# Usage examples

## DIM as library
DIM can also be utilized as a library instead of being executed through the Command Line Interface (examples in the next section). For an illustrative example, please see `notebooks/sfm_pipeline.ipynb`.

## CLI

### Basic usage

In this section, we will explore some example usages, beginning with a basic command that defines the working directory containing a folder named `images`, which houses all the images to be processed, and the pipeline to be used (extractor + matcher).

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue```

If the images are located in a different folder relative to the working directory, you can specify the images directory by adding the option `--images`.

```python ./main.py --dir /path/to/working/dir --images /path/to/image/folder --pipeline superpoint+lightglue```

### Pass your camera models and options

DIM stores the image matches in an h5 file and a COLMAP database, which you can locate in the result folders. By default, DIM attempts to orient the image block using pycolmap with the default camera options specified in `config/cameras.yaml` and the pycolmap parameters defined at the end of `main.py`. If you wish to specify different camera models and parameters, you can either modify the default parameters in `config/cameras.yaml` or provide the path to a YAML file with the same structure using the option `--camera_options`.

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --camera_options path/to/camera/options/yaml/format```

**Please, for a detailed description of camera options, see the camera_model section in the documentation.**

### Skip reconstruction

To skip the reconstruction process with pycolmap or other SfM software like OpenMVG, simply include the option --skip_reconstruction. This enables you to directly access, for example, the database.db file created in the result folder using the COLMAP GUI. You can then proceed with the image block orientation as usual within the COLMAP GUI.

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --skip_reconstruction```

### Pass your options for feature extractor and matcher

The extractor and matcher are specified using the `--pipeline` option. To view all available pipelines, you can run python `./main.py --help`. If needed, you can modify parameters related to the extractor and matcher by providing a YAML file to the `--config` option. Examples of YAML configurations can be found in the config folder. Please, see `config` section in the documentation for more info.

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --config superpoint+lightglue.yaml```

### Image quality and tiling

If the images are high resolution and you prefer to extract features at a specific lower image resolution, you can indicate the image quality (lowest, low, medium, high, highest) using the option `--quality` (refer to [Quality](./getting_started.md#quality)). If the image size remains too large for direct feature extraction at this resolution, you can employ a tiling process to extract features from tiles with option `--tiling` (see [Tiling](./getting_started.md#tiling) in the getting_started documentation).

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --quality medium --tiling preselection```

### Matching strategy

With the --strategy option (see [Matching strategies](./getting_started.md#matching-strategies) for more details), you can specify the matching strategy. For instance, for a brute-force approach:

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --strategy bruteforce```

For sequential matching, also defining how many images overlap:

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --strategy sequential --overlap 1```

Or passing your custom pairs:

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --strategy custom_pairs --pair_file path/to/your/txt/file```

Or using global descriptors to reduce the number of image pairs:

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --strategy retrieval --global_feature netvlad```

### Images upright

Try to orient all images upright for local features not invariant to rotations:

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --upright```

### Graph

Prepare a graph visualization in html of matches between images (found in results folder):

```python ./main.py --dir /path/to/working/dir --pipeline superpoint+lightglue --graph```

### OpenMVG processing

Please see [OpenMVG section](./openmvg.md).