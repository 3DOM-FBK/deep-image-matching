# Advanced configuration

If you want to set any additional parameter, you can do it by editing the `config.yaml` file that must be located in the same directory of the `main.py` file.
These are non-mandatory parameters that can be used to fine-tune the matching process, but none of them is specifically required to run the matching, as default values are set for all of them.

The `config.yaml` file is a YAML file that contains all the parameters that can be set for the matching. The parameters are divided into different sections: 'general', 'extractor', 'matcher'.

The 'general' section contains general parameters for the processing (in addition to those defined by the CLI arguments):

- `tile_size`: size of the tiles defined as a Tuple (width, height) in pixel (default `(2400, 2000)`),
- `tile_overlap`: the tiles can be overlapped by a certain amount of pixel to avoid features/matches detected close to the tile borders (default `10`),
- `min_matches_per_tile`: the minimum number of matches per each tile. Below this number, the matches are rejected because they are considered not robust enough (default `10`),
- `geom_verification`: defines the geometric verification method to be used. Available options are: `NONE` (no geometric verification), `PYDEGENSAC` (use pydegensac), `MAGSAC` (use opencv MAGSAC). (default `GeometricVerification.PYDEGENSAC`),
- `gv_threshold`: the threshold for the geometric verification (default `4`),
- `gv_confidence`: the confidence for the geometric verification (default `0.99999`),
- `min_inliers_per_pair`: the minimum number of inliers matches per image pair (default `15`),
- `min_inlier_ratio_per_pair`: the minimum inlier ratio (i.e., number of valid matches/number of total matches) per image pair (default `0.25`),
- `try_match_full_images`: even if the features are extracted by tiles, you can try to match the features of the entire image first (if the number of features is not too high and they can fit into memory) (Default is False).

For example, if you want to change the tile size and the tile overlap, you can set the `tile_size` and `tile_overlap` parameters in the `general` section as follows:

```yaml
general:
  tile_size: (2400, 2000)
  tile_overlap: 20
```

The `extractor` and `matcher` sections contain the parameters that control the local feature extractor and the matcher selected by the '--pipeline' option from the CLI (or from GUI).
Both the sections **must contain the name** of the local feature extractor or the matcher that will be used for the matching (the name must be the same as the one used in the `--pipeline` option in the CLI).
In addition, you can specify any other parameters for controlling the extractor and the matcher.
The default values of all the configuration parameters are defined in the [`config.py`](https://github.com/3DOM-FBK/deep-image-matching/blob/master/src/deep_image_matching/config.py) file located in `/src/deep_image_matching` directory.
Please, note that different extractors or matchers may have different parameters, so you need to check carefully the available parameters for each extractor/matcher in the file [`config.py`](https://github.com/3DOM-FBK/deep-image-matching/blob/master/src/deep_image_matching/config.py).

```yaml
extractor:
  name: "superpoint"
  max_keypoints: 8000

matcher:
  name: "lightglue"
  filter_threshold: 0.1
```

Note, that you can use an arbitrary number of parameters for each configuration section ("general", "extractor", "matcher"): you can set only one parameter or all of them, depending on your needs.
Here is an example of `config.yaml` file for the `superpoint+lightglue` pipeline:

<details>

<summary>config.yaml</summary>

```yaml
general:
  tile_size: (2400, 2000)
  tile_overlap: 10
  tile_preselection_size: 1000,
  min_matches_per_tile: 10,
  geom_verification: "PYDEGENSAC", # or NONE, PYDEGENSAC, MAGSAC
  gv_threshold: 4,
  gv_confidence: 0.99999,
  min_inliers_per_pair: 10
  min_inlier_ratio_per_pair: 0.1


extractor:
  name: "superpoint"
  max_keypoints: 4096
  keypoint_threshold: 0.0005
  nms_radius: 3

matcher:
  name: "lightglue"
  filter_threshold: 0.1
  n_layers: 9
  mp: False # enable mixed precision
  flash: True # enable FlashAttention if available.
  depth_confidence: 0.95 # early stopping, disable with -1
  width_confidence: 0.99 # point pruning, disable with -1
  filter_threshold: 0.1 # match threshold

```

</details>
