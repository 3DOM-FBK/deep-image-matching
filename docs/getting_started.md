# Getting started with Deep-Image-Matching

Deep_Image_Matching can be launched from the Command Line (CLI), from the GUI (note that the GUI is still under development) or use Deep_Image_Matching as a Python library. 

## Command Line Interface (CLI)

Before running the CLI, check the options with `python ./main.py --help`.

The minimal required option are:

- `--dir` `-d`: it is the path of the 'project directory', i.e., the directory containing a folder names 'images', with all the image to be processed, and where the output will be saved
- `--config` `-c`: the name of the combination of local feature extractor and matcher to use (e.g., "superpoint+lightglue"). See the [Local feature extractor and matcher](#local-feature-extractor-and-matcher) section for more details.

Other optional parameters are:

- `--strategy`: the strategy to use for matching the images. See [Matching strategies](#matching-strategies) section (default: `matching_lowres`)
- `--quality` `-Q`  : the quality of the images to be matched. It can be `low`, `medium` or `high` See [Quality](#quality) section (default: `high`).
- `tiling` `-t`: if passed, the images are tiled in 4 parts and each part is matched separately. This is useful for high-resolution images if you do not want to resize them. See [Tiling](#tiling) section (default: `None`).
- `--images`: if the folder containing the image is not located in the project directory, you can manually specify the path to the folder containing the images If nothing is passed, deep_image_matching will look for a folder named "image" inside the project directory (default: `None`).
- `--outs`: if you want the outputs to be save to a specific folder, different than the one set with '--dir', the path to the folder where the matches will be saved. If nothing is passed, the output will e saved in a folder 'results' inside the project direcoty (default: `None`)
- `--upright`: if passed, try to find the best image rotation before running the matching (default: `False`).
- `--skip_reconstruction` : Skip reconstruction step carried out with pycolmap, but save the matched features into a Sqlite3 database that can be opened by COLMAP GUI for bundle adjustment. The reconstruction with pycolmap is necessary to export the solution in Bundler format for Agisoft Metashape (default: `False`)
- `--force`: if the output folder already exists, overwrite it (default: `False`)
- `-V`: enable verbose output (default: `False`)
- `--help`: show the help message

Finally, there are some 'strategy-dependent' options (i.e., options that are used only with specific strategies). These options are:

- `--overlap`: if  'strategy' is set to 'sequential', set the number of images that are sequentially matched in the sequence (default: `None`)
- `--retrieval`: if `strategy` is set to `retrieval`, set the global descriptor to use for image retrieval. Options are: "netvlad", "openibl", "cosplace", "dir" (default: `None`).


## GUI

**Note that the GUI is still under development and it may have some bugs**

To run with the GUI:

```bash
python ./main.py --gui
```

In the GUI, you can define the same parameters that are available in the CLI.
The GUI loads the available configurations from `config.py` `config.py` file located in `/src/deep_image_matching`.

## From a Jupyter notebook

If you want to use Deep_Image_Matching from a Jupyter notebook, you can check the examples in the `notebooks` folder.

## Basic configuration

### Local feature extractor and matcher

The combination of local feature extractor and matcher to be used for the matching is is defined by the `--config` option in the CLI.

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



All the available configurations are defined in the file `config.py` file located in `/src/deep_image_matching`. You can check them by running:

```bash
python ./main.py --help
```

Alternatively, if you are working in a notebook you can use the Config class:

```python
from pprint import pprint
from deep_image_matching.config import Config

pprint(Config.get_config_names())
```

From the GUI, you can chose the configuration from the drop-down menu.


### Matching strategies

The matching strategy defines how the pairs of images to be matches are selected. Available matching strategies are:

- `matching_lowres`: the images are first matched at low resolution (resizing images with the longest edge to 1000 px) and candidates are selected based on the number of matches (the minimum number of matches is 20). Once the candidate pairs are selected, the images are matched at the desired resolution, specified by the `Quality` parameter in the configuration. This is the default option and the recommended strategy, especially for large datasets.
- `bruteforce`: all the possible pairs of images are matched. This is is usefult in case of very challenging datasets, where some image pairs may be rejected by the previous strategies, but it can take significantly more time for large datasets. 
- `sequential`: the images are matched sequentially, i.e., the first image is matched with the second, the second with the third, and so on. The number of images to be matched sequentially is defined by the `--overlap` option in the CLI. This strategy is useful for datasets where the images are taken sequentially, e.g., from a drone or a car.
- retrieval: the images are matched based on a global descriptor. The global descriptor is computed for each image and the images are matched based on the distance between the descriptors. The global descriptor to be used is defined by the `--retrieval` option in the CLI. Available global descriptors are: "netvlad", "openibl", "cosplace", "dir".
- `custom_pairs`: the pairs of images to be matched are defined in a text file. The path to the text file is defined by the `--pairs` option in the CLI. The text file must contain one pair of images per line, separated by a space. The images must be identified by their full name (i.e., the name of the image file with the extension). 
For example:

```bash
    image_1.jpg image_2.jpg
    image_3.jpg image_4.jpg
```

### Quality

The `Quality` parameter define the resolution at which the images are matched. The available options are:

- `high`: the images are matched at the original resolution (default)
- `highest`: the images are upsampled by a factor of 2 by using a bicubic interpolation.
- `medium`: the images are downsampled by a factor of 2 by using the OpenCV pixel-area approach ([cv2.INTER_AREA](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121acf959dca2480cc694ca016b81b442ceb))
- `low`: the images are downsampled by a factor of 4 
- `lowest`: the images are downsampled by a factor of 8

### Tiling

### Advanced configuration

If you want to set any additional parameter, you can do it by editing the `config.yaml` file that must be located in the same directory of the `main.py` file.