# Configuration

### DIM as library

DIM can also be utilized as a library instead of being executed through the Command Line Interface (CLI). For an illustrative example also setting custom configurations, please see `notebooks/sfm_pipeline.ipynb`.

### CLI

When utilizing the CLI, specific configuration parameters linked to feature extraction and matching can be specified using the option `--config /path/to/yaml/configuration/file`. Examples of YAML configuration files can be found in the `config` folder.

For example, in the `config/sift.yaml` file, under the general section, parameters such as tile size and others related to geometric verification can be adjusted. Under the extractor and matcher sections, there are options specific to the SIFT local feature.

```
general:
  tile_size: (2400, 2000)
  geom_verification: pydegensac
  min_inliers_per_pair: 10
  min_inlier_ratio_per_pair: 0.25

extractor:
  name: "sift"
  n_features: 8000
  nOctaveLayers: 3
  contrastThreshold: 0.04
  edgeThreshold: 10
  sigma: 1.6

matcher:
  name: "kornia_matcher"
  match_mode: "smnn"
  th: 0.85
```

### The Config Class

::: deep_image_matching.config.Config
    options:
      show_root_heading: true
      show_source: false
      members:
