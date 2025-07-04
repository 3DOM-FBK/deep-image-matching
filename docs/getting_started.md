# Getting started with Deep-Image-Matching

Deep-Image-Matching can be launched from the Command Line (CLI) or used as a Python library. A GUI is also available but is still under development.

In the `assets` folder there are some projects and images for testing.

## Quick Start with Demo

For a quick start, check out the demo files that demonstrate basic usage with the example dataset:

- `demo.py` - Simple script demonstrating the basic workflow
- `demo.ipynb` - Interactive notebook version of the demo

```bash
python demo.py --dir assets/example_cyprus --pipeline superpoint+lightglue
```

## Run Deep-Image-Matching

### Command Line Interface (CLI)

Before running the CLI, check the options with:

```bash
python -m deep_image_matching --help
```

The minimal required options are:

- `--dir` `-d`: the path of the 'project directory', i.e., the directory containing a folder named 'images', with all the images to be processed, and where the output will be saved
- `--pipeline` `-p`: the name of pipeline (i.e., the combination of local feature extractor and matcher) to use (e.g., "superpoint+lightglue"). See the [Pipelines](#pipelines) section for more details.

Example:

```bash
python -m deep_image_matching --dir ./assets/example_cyprus --pipeline superpoint+lightglue
```

Other optional parameters are:

- `--config_file` `-c`: the path to the YAML configuration file containing the custom configuration. See the [Advanced configuration](./advanced_configuration.md) section (default: `None`, so default configuration is used)
- `--strategy` `-s`: the strategy to use for matching the images. It can be `matching_lowres`, `bruteforce`, `sequential`, `retrieval`, `custom_pairs`. See [Matching strategies](#matching-strategies) section (default: `matching_lowres`)
- `--quality` `-q`: the quality of the images to be matched. It can be `lowest`, `low`, `medium`, `high` or `highest`. See [Quality](#quality) section (default: `high`).
- `tiling` `-t`: if passed, the images are tiled in 4 parts and each part is matched separately. This is useful for high-resolution images if you do not want to resize them. See [Tiling](#tiling) section (default: `None`).
- `--upright`: if passed, try to find the best image rotation before running the matching (default: `False`).
- `--skip_reconstruction` : Skip reconstruction step carried out with pycolmap, but save the matched features into a Sqlite3 database that can be opened by COLMAP GUI for bundle adjustment. The reconstruction with pycolmap is necessary to export the solution in Bundler format for Agisoft Metashape (default: `False`)
- `--force`: if the output folder already exists, overwrite it (default: `False`)
- `-V`: enable verbose output (default: `False`)
- `--help` `-h`: show the help message
<!-- - `--images` `-i`: if the folder containing the image is not located in the project directory, you can manually specify the path to the folder containing the images If nothing is passed, deep_image_matching will look for a folder named "image" inside the project directory (default: `None`).
- `--outs` `-o`: if you want the outputs to be save to a specific folder, different than the one set with '--dir', the path to the folder where the matches will be saved. If nothing is passed, the output will e saved in a folder 'results' inside the project direcoty (default: `None`) -->

Finally, there are some 'strategy-dependent' options (i.e., options that are used only with specific strategies). See [Matching strategies](#matching-strategies) section for more information. These options are:

- `--overlap`: if 'strategy' is set to 'sequential', set the number of images that are sequentially matched in the sequence (default: `1`)
- `--global_feature`: if `strategy` is set to `retrieval`, set the global descriptor to use for image retrieval. Options are: "netvlad", "openibl", "cosplace", "dir" (default: `netvlad`).
- `--pair_file`: if `strategy` is set to `custom_pairs`, set the path to the text file containing the pairs of images to be matched. (default: `None`).

### Library usage

Deep-image-matching can also be used as a Python library. For comprehensive examples, see:

- `demo.py` - Simple script demonstrating the basic workflow
- `demo.ipynb` - Interactive notebook version of the demo
- `notebooks/sfm_pipeline.ipynb` - Complete SfM pipeline with detailed explanations

### GUI (Under Development)

**Note:** The GUI is still under development and may have some bugs.

To run with the GUI:

```bash
python -m deep_image_matching --gui
```

In the GUI, you can define the same parameters that are available in the CLI.
The GUI loads the available configurations from [`config.py`](https://github.com/3DOM-FBK/deep-image-matching/blob/master/src/deep_image_matching/config.py) located in `/src/deep_image_matching`.

### From Jupyter notebooks

If you want to use Deep_Image_Matching from a Jupyter notebook, you can check the examples in the [`notebooks`](https://github.com/3DOM-FBK/deep-image-matching/tree/master/notebooks) folder.

## Pipelines

The `pipeline` parameter defines the combination of local feature extractor and matcher to be used for the matching is is defined by the `--pipeline` option in the CLI.

Possible configurations are:

- superpoint+lightglue
- superpoint+lightglue_fast
- superpoint+superglue
- disk+lightglue
- aliked+lightglue
- orb+kornia_matcher (i.e., ORB (OpenCV) + Nearest Neighbor matcher)
- sift+kornia_matcher (i.e., sift (OpenCV) + Nearest Neighbor matcher)
- keynetaffnethardnet+kornia_matcher (i.e., keynetaffnethardnet (Kornia) + Nearest Neighbor matcher)
- dedode+kornia_matcher (i.e., DeDoDe + Nearest Neighbor matcher)
- loftr (detector-free matcher)
- roma (detector-free matcher)

You can check all the available configurations by running:

```bash
python -m deep_image_matching --help
```

Alternatively, if you are working in a notebook you can use the Config class:

```python
from pprint import pprint
from deep_image_matching.config import Config

pprint(Config.get_config_names())
```

From the GUI, you can chose the configuration from the drop-down menu.

More information can be obtained looking to the code in [`config.py`](https://github.com/3DOM-FBK/deep-image-matching/blob/master/src/deep_image_matching/config.py) file located in `/src/deep_image_matching` directory.

## Matching strategies

The matching strategy defines how the pairs of images to be matches are selected. Available matching strategies are:

- `matching_lowres`: the images are first matched at low resolution (resizing images with the longest edge to 1000 px) and candidates are selected based on the number of matches (the minimum number of matches is 20). Once the candidate pairs are selected, the images are matched at the desired resolution, specified by the `Quality` parameter in the configuration. This is the default option and the recommended strategy, especially for large datasets.
- `bruteforce`: all the possible pairs of images are matched. This is is useful in case of very challenging datasets, where some image pairs may be rejected by the previous strategies, but it can take significantly more time for large datasets.
- `sequential`: the images are matched sequentially, i.e., the first image is matched with the second, the second with the third, and so on. The number of images to be matched sequentially is defined by the `--overlap` option in the CLI. This strategy is useful for datasets where the images are taken sequentially, e.g., from a drone or a car.
- `retrieval`: the images are first matched based with a global descriptor to select the pairs. The global descriptor to be used is defined by the `--retrieval` option in the CLI. Available global descriptors are: "netvlad", "openibl", "cosplace", "dir".
- `custom_pairs`: the pairs of images to be matched are defined in a text file. The path to the text file is defined by the `--pairs` option in the CLI. The text file must contain one pair of images per line, separated by a space. The images must be identified by their full name (i.e., the name of the image file with the extension).
  For example:

```text
    image_1.jpg image_2.jpg
    image_3.jpg image_4.jpg
```

## Quality

The `Quality` parameter define the resolution at which the images are matched. The available options are:

- `highest`: each image size is upsampled by a factor of 2 by using a bicubic interpolation ([cv2.INTER_CUBIC](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121a55e404e7fa9684af79fe9827f36a5dc1)).
- `high`: images are matched at the original resolution (default)
- `medium`: images are downsampled by a factor of 2 by using the OpenCV pixel-area approach ([cv2.INTER_AREA](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb)).
- `low`: images are downsampled by a factor of 4 by using the OpenCV pixel-area approach ([cv2.INTER_AREA](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb)).
- `lowest`: images are downsampled by a factor of 8 by using the OpenCV pixel-area approach ([cv2.INTER_AREA](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb)).

## Tiling

If images have a high resolution (e.g., larger than 3000 px, but this limit depends on the memory of your GPU) and you do not want to downsample them (e.g., to avoid loosing accuracy in feature detection), it may be useful to carry out the matching by dividing the images into regular tiles.
This can be done by specifying the tiling approach with the `--tiling` option in the CLI.
If you want to run the matching by tile, you can choose different approaches for selecting the tiles to be matched. Available options are:

- `None`: no tiling is applied (default)
- `preselection`: images are divided into a regular grid of size 2400x2000 px and the features are extracted from each tile separately on all the images. For the matching, each image pair is first matched at a low resolution to select which are the tiles that most likely see the same scene and therefore are good candidates to be matched. Then, the selected candidate tiles are matched at full resolution. This is the recommended option for most of the cases as it allows for a significantly reduction in processing time compared to the `exhaustive` approach.
- `grid`: the images are divided into a regular grid of size 2400x2000 px for feature extraction. The matching is carried out by matching the tiles in the same position in the grid (e.g., the tile in the top-left corner of the first image is matched with the tile in the top-left corner of the second image). This method is recommended only if the images are taken from the same point of view and the camera is not rotated.
- `exhaustive`: the images are divided into a regular grid of size 2400x2000 px to extract the features. The matching is carried out by matching all the possible combinations of tiles (brute-force). This method can be very slow for large images or in combination with the `highest` quality option and, in some cases, it may lead to error in the geometric verification if too many wrong matches are detected.

To control the tile size and the tile overlap, refer to the [Advanced configuration](./advanced_configuration.md) section.
