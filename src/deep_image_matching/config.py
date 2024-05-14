import ast
import json
import logging
import shutil
from copy import deepcopy
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Tuple

import yaml

from .constants import GeometricVerification, Quality, TileSelection
from .utils.logger import change_logger_level

logger = logging.getLogger("dim")

# General configuration for the matching process.
# It defines the quality of the matching process, the tile selection strategy, the tiling grid, the overlap between tiles, the geometric verification method, and the geometric verification parameters.
conf_general = {
    # run GUI
    "gui": False,
    # Project directory containing all the data (it must contain the "images" folder)
    "dir": None,
    # Path to the images folder (if not provided, it is assumed that the images are in the project directory)
    "images": None,
    # Output directory where the results will be saved (if not provided, it is created in the project directory)
    "outs": None,
    # Flag to force the execution even if the output folder already exists (default is True)
    "force": True,
    # Pipeline to use for feature extraction and matching
    "pipeline": None,
    # External configuration file to use for feature extraction and matching
    "config_file": None,
    # Matching strategy to use for the matching process
    #   bruteforce (match all the images with all the other images),
    #   matching_lowres (match images using low resolution features),
    #   sequential (match each image with the next n images, where n is the overlap parameter),
    #   retrieval (match images using global features),
    #   custom_pairs (match images using a custom pair file),
    "strategy": "matching_lowres",
    # Image resolution presets:
    #   Quality.HIGHEST (x2)
    #   Quality.HIGH (x1 - full resolutions)
    #   Quality.MEDIUM (x1/2)
    #   Quality.LOW (x1/4)
    #   Quality.LOWEST (x1/8)
    "quality": Quality.HIGH,
    # Tile selection approach:
    #   TileSelection.NONE (no tile selection, use entire images),
    #   TileSelection.PRESELECTION (select tiles based on a low resolution matching),
    #   TileSelection.GRID (match all the corresponding tiles in a grid)
    #   TileSelection.EXHAUSTIVE (match all the possible pairs of tiles)
    "tile_selection": TileSelection.PRESELECTION,
    # File containing the pairs to match (used only if strategy is set to custom_pairs)
    "pair_file": None,
    # Overlap between images (used only if strategy is set to sequential)
    "overlap": None,
    # Global feature to use for the retrieval strategy (used only if strategy is set to retrieval)
    "global_feature": None,
    # Path to the database file (used only if strategy is set to covisibility)
    "db_path": None,
    # Flag to indicate if hthe best image roation must be estiamted prior the matching (default is False)
    "upright": False,
    # Flag to skip the reconstruction step with pycolmap (default is False)
    "skip_reconstruction": False,
    # Flag to enable/disable the verbose mode (default is False)
    "verbose": False,
    # Flag to enable/disable the graph visualization (default is True)
    "graph": True,
    # Path to the OpenMVG configuration file (used only if the reconstruction step is enabled)
    "openmvg_conf": None,
    # Path to the camera options file (used only if the reconstruction step is enabled)
    "camera_options": None,
    # Size of the tiles in pixels (width, height) or (x, y)
    "tile_size": (2400, 2000),
    # Overlap between tiles in pixels
    "tile_overlap": 10,
    # Size of the low resolution tiles used for the tile preselection
    "tile_preselection_size": 1000,
    # Minimum number of matches per tile
    "min_matches_per_tile": 10,
    # Geometric verification method and parameters:
    #   GeometricVerification.NONE (no geometric verification),
    #   GeometricVerification.PYDEGENSAC (use pydegensac),
    #   GeometricVerification.MAGSAC (use opencv MAGSAC),
    #   Other methods: RANSAC, LMEDS, RHO, USAC_DEFAULT, USAC_PARALLEL, USAC_FM_8PTS, USAC_FAST, USAC_ACCURATE, USAC_PROSAC, USAC_MAGSAC
    "geom_verification": GeometricVerification.PYDEGENSAC,
    "gv_threshold": 4,
    "gv_confidence": 0.99999,
    # Minimum number of inliers matches and minumum inlier ratio per pair
    "min_inliers_per_pair": 15,
    "min_inlier_ratio_per_pair": 0.25,
    # Even if the features are extracted by tiles, you can try to match the features of the entire image first (if the number of features is not too high and they can fit into memory). Default is False.
    "try_match_full_images": False,
    # Pipeline to use for tile preselection
    "preselection_pipeline": "superpoint+lightglue",
}


# The configuration for DeepImageMatcher is defined as a dictionary with the following keys:
# - 'extractor': extractor configuration
# - 'matcher': matcher configuration
# The 'extractor' and 'matcher' values must contain a 'name' key with the name of the extractor/matcher to be used. Additionally, the other parameters of the extractor/matcher can be specified.
confs = {
    "superpoint+lightglue": {
        "extractor": {
            "name": "superpoint",
            "nms_radius": 3,
            "keypoint_threshold": 0.0005,
            "max_keypoints": 4096,
        },
        "matcher": {
            # Refer to https://github.com/cvg/LightGlue/tree/main for the meaning of the parameters
            "name": "lightglue",
            "n_layers": 9,
            "mp": False,  # enable mixed precision
            "flash": True,  # enable FlashAttention if available.
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
        },
    },
    "superpoint+superglue": {
        "extractor": {
            "name": "superpoint",
            "nms_radius": 3,
            "keypoint_threshold": 0.0005,
            "max_keypoints": 4096,
        },
        "matcher": {
            "name": "superglue",
            "weights": "outdoor",
            "match_threshold": 0.3,
            "sinkhorn_iterations": 100,
        },
    },
    "superpoint+kornia_matcher": {
        "extractor": {
            "name": "superpoint",
            "nms_radius": 3,
            "keypoint_threshold": 0.0005,
            "max_keypoints": 4096,
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.95},
    },
    "disk+lightglue": {
        "extractor": {
            "name": "disk",
            "max_keypoints": 4096,
            "nms_window_size": 5,
            "detection_threshold": 0.0,
            "pad_if_not_divisible": True,
        },
        "matcher": {
            "name": "lightglue",
        },
    },
    "aliked+lightglue": {
        "extractor": {
            "name": "aliked",
            "model_name": "aliked-n16rot",
            "max_num_keypoints": 4000,
            "detection_threshold": 0.2,
            "nms_radius": 3,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
        },
    },
    "orb+kornia_matcher": {
        "extractor": {
            "name": "orb",
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "snn"},
    },
    "sift+kornia_matcher": {
        "extractor": {
            "name": "sift",
            "n_features": 4000,
            "nOctaveLayers": 3,
            "contrastThreshold": 0.04,
            "edgeThreshold": 10,
            "sigma": 1.6,
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.85},
    },
    "loftr": {
        "extractor": {"name": "no_extractor"},
        "matcher": {"name": "loftr", "pretrained": "outdoor"},
    },
    "se2loftr": {
        "extractor": {"name": "no_extractor"},
        "matcher": {"name": "se2loftr", "pretrained": "outdoor"},
    },
    "roma": {
        "extractor": {"name": "no_extractor"},
        "matcher": {"name": "roma", "pretrained": "outdoor"},
    },
    "keynetaffnethardnet+kornia_matcher": {
        "extractor": {
            "name": "keynetaffnethardnet",
            "n_features": 4000,
            "upright": False,
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.95},
    },
    "dedode+kornia_matcher": {
        "extractor": {
            "name": "dedode",
            "max_keypoints": 10_000,
            "detector_weights": "L-SO2",  # [L, L-C4, L-SO2, L-C4-v2]
            "descriptor_weights": "G-C4",  # [B-upright, G-upright, B-C4, B-SO2, G-C4]
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.99},
    },
    "xfeat+kornia_matcher": {
        "extractor": {
            "name": "xfeat",
            "top_k": 4000,
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.99},
    },
    "sift+lightglue": {
        "extractor": {
            "name": "sift",
            "n_features": 8192,
            "nOctaveLayers": 4,
            "contrastThreshold": 0.006,
            "edgeThreshold": 10,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
        },
    },
    "dedode+lightglue": {
        "extractor": {
            "name": "dedode",
            "max_keypoints": 10_000,
            "detector_weights": "L-SO2",  # [L, L-C4, L-SO2, L-C4-v2]
            "descriptor_weights": "G-C4",  # [B-upright, G-upright, B-C4, B-SO2, G-C4]
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
        },
    },
    "sift_pycolmap+lightglue": {
        "extractor": {
            "name": "sift_pycolmap",
            "num_threads": -1,
            "max_image_size": 3200,
            "max_num_features": 8192,
            "first_octave": -1,
            "num_octaves": 4,
            "octave_resolution": 3,
            "peak_threshold": 0.006666666666666667,
            "edge_threshold": 10.0,
            "estimate_affine_shape": False,
            "max_num_orientations": 2,
            "upright": False,
            "darkness_adaptivity": False,
            "domain_size_pooling": False,
            "dsp_min_scale": 0.16666666666666666,
            "dsp_max_scale": 3.0,
            "dsp_num_scales": 10,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
        },
    },
}

opt_zoo = {
    "retrieval": ["netvlad", "openibl", "cosplace", "dir"],
    "matching_strategy": [
        "bruteforce",
        "sequential",
        "retrieval",
        "custom_pairs",
        "matching_lowres",
        "covisibility",
    ],
}


class Config:
    """
    Configuration class for deep image matching.

    This class represents the configuration settings for deep image matching. It provides methods to parse user input,
    retrieve configuration options, update configuration from a YAML file, print the configuration settings, and save
    the configuration to a file.

    Attributes:
        _cfg (dict): The configuration dictionary with the following keys: general, extractor, matcher.

    Methods:
        general: Get the general configuration options.
        extractor: Get the extractor configuration options.
        matcher: Get the matcher configuration options.
        __init__: Initialize the Config object.
        as_dict: Get the configuration dictionary.
        get_config: Get a specific configuration by name.
        get_config_names: Get a list of available configuration names.
        get_matching_strategy_names: Get a list of available matching strategy names.
        get_extractor_names: Get a list of available extractor names.
        get_matcher_names: Get a list of available matcher names.
        get_retrieval_names: Get a list of available retrieval names.
        parse_user_config: Parse the user configuration and perform checks on the input arguments.
        update_from_yaml: Update the configuration from a YAML file.
        print: Print the configuration settings.
        save: Save the configuration to a file.
    """

    _cfg = {
        "general": {},
        "extractor": {},
        "matcher": {},
    }

    @property
    def general(self):
        return self._cfg["general"]

    @property
    def extractor(self):
        return self._cfg["extractor"]

    @property
    def matcher(self):
        return self._cfg["matcher"]

    def __repr__(self) -> str:
        return f"DeepImageMatching Configuration Object"

    def __init__(self, args: dict):
        """
        Initialize the Config object.

        Args:
            args (dict): The input arguments provided by the user.
        """
        # Parse input arguments
        general = self.parse_general_config(args)

        # Build configuration dictionary
        self._cfg["general"] = {**conf_general, **general}
        features_config = self.get_config(args["pipeline"])
        self._cfg["extractor"] = features_config["extractor"]
        self._cfg["matcher"] = features_config["matcher"]

        # If the user has provided a configuration file, update the configuration
        if "config_file" in args and args["config_file"] is not None:
            config_file = Path(args["config_file"]).resolve()
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file {config_file} not found.")
            self.update_from_yaml(config_file)
            self.print()

        self.config_file = self._cfg["general"]["output_dir"] / "config.json"
        self.save(self.config_file)

    def as_dict(self) -> dict:
        """
        Get the configuration dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return self._cfg

    @staticmethod
    def get_config(name: str) -> dict:
        """
        Get a specific configuration by name.

        Args:
            name (str): The name of the configuration.

        Returns:
            dict: The configuration dictionary.

        Raises:
            ValueError: If the configuration name is invalid.
        """
        try:
            return confs[name]
        except KeyError:
            raise ValueError(f"Invalid configuration name: {name}")

    @staticmethod
    def get_pipelines() -> list:
        return list(confs.keys())

    @staticmethod
    def get_matching_strategies() -> list:
        return opt_zoo["matching_strategy"]

    @staticmethod
    def get_retrieval_names() -> list:
        return opt_zoo["retrieval"]

    @staticmethod
    def parse_general_config(args: dict) -> dict:
        """
        Parses the user configuration and performs checks on the input arguments.

        Args:
            args (dict): The input arguments provided by the user (e.g., from CLI parser).

        Returns:
            dict: The configuration dictionary with the following keys: general, extractor, matcher.

        """

        # Rename some input arguments
        dict_map = {
            "tile_selection": "tiling",
            "openmvg_conf": "openmvg",
        }
        for k, v in dict_map.items():
            if v in args:
                args[k] = args.pop(v)

        args = {**conf_general, **args}

        # Check that at least one of the two options is provided
        if args["images"] is None and args["dir"] is None:
            raise ValueError("Invalid input. Either '--images' or '--dir' option must be provided.")

        # If the project directory is not provided, check that the image folder is provided (and it valid)
        if args["dir"] is None:
            args["images"] = Path(args["images"])
            if not args["images"]:
                raise ValueError("Invalid input. '--images' option is required when '--dir' option is not provided.")
            if not args["images"].exists() or not args["images"].is_dir():
                raise ValueError(f"Invalid images folder {args['images']}. Direcotry does not exist")
            args["dir"] = args["images"].parent
        else:
            args["dir"] = Path(args["dir"])
            # Check and defines all input/output folders
            if not args["dir"].exists() or not args["dir"].is_dir():
                raise ValueError(f"Folder {args['dir']} does not exist")
            else:
                if not args["dir"].exists() or not args["dir"].is_dir():
                    raise ValueError(f"Folder {args['dir']} does not exist")

        # Check images folder
        if args["images"] is None:
            args["images"] = args["dir"] / "images"
            if not args["images"].exists() or not args["images"].is_dir():
                raise ValueError(
                    "'images' folder not found in project directory. Create an 'images' folder containing all the images or use '--images' option to specify a valid folder."
                )
        else:
            args["images"] = Path(args["images"])
            if not args["images"].exists() or not args["images"].is_dir():
                raise ValueError(f"Invalid images folder {args['images']}. Direcotry does not exist")

        # if output folder is not provided, use the default one
        if args["outs"] is None:
            args["outs"] = args["dir"] / f"results_{args['pipeline']}_{args['strategy']}_quality_{args['quality']}"

        if args["outs"].exists():
            if args["force"]:
                logger.warning(f"{args['outs']} already exists, but the '--force' option is used. Deleting the folder.")
                shutil.rmtree(args["outs"])
            else:
                logger.warning(
                    f"{args['outs']} already exists. Use '--force' option to overwrite the folder. Using existing features is not yet fully implemented (it will be implemented in a future release). Exiting."
                )
                # TODO: implement the option to use existing features
                # exit(1)
        args["outs"].mkdir(parents=True, exist_ok=True)

        # Check extraction and matching configuration
        if args["pipeline"] is None or args["pipeline"] not in confs:
            raise ValueError(
                "Invalid config. --pipeline option is required and must be a valid pipeline. Check --help for details"
            )
        pipeline = args["pipeline"]
        extractor = confs[pipeline]["extractor"]["name"]
        matcher = confs[pipeline]["matcher"]["name"]

        # Check matching strategy and related options
        if args["strategy"] not in opt_zoo["matching_strategy"]:
            raise ValueError(
                f"Invalid strategy option: {args['strategy']}. Valid options are: {opt_zoo['matching_strategy']}. Check --help for details"
            )

        if args["strategy"] == "sequential":
            num_imgs = len(list(args["images"].glob("*")))
            if args["overlap"] is None or not isinstance(args["overlap"], int):
                raise ValueError(
                    "Invalid overlap. --overlap 'int' option is required when --strategy is set to sequential"
                )
            elif args["overlap"] < 1:
                raise ValueError("Invalid overlap. --overlap must be a positive integer greater than 0")
            elif args["overlap"] >= num_imgs - 1:
                raise ValueError("Invalid overlap. --overlap must be less than the number of images-1")

        elif args["strategy"] == "retrieval":
            if args["global_feature"] is None:
                raise ValueError("--global_feature option is required when --strategy is set to retrieval")
            elif args["global_feature"] not in opt_zoo["retrieval"]:
                raise ValueError(
                    f"Invalid global_feature option: {args['global_feature']}. Valid options are: {opt_zoo['retrieval']}"
                )

        elif args["strategy"] == "covisibility":
            if args["db_path"] is None:
                raise ValueError("--db_path option is required when --strategy is set to covisibility")
            args["db_path"] = Path(args["db_path"])
            if not args["db_path"].exists():
                raise ValueError(f"File {args['db_path']} does not exist")

        elif args["strategy"] == "custom_pairs":
            if args["pair_file"] is None:
                raise ValueError("--pair_file option is required when --strategy is set to custom_pairs")
            args["pair_file"] = Path(args["pair_file"])
            if not args["pair_file"].exists():
                raise ValueError(f"File {args['pair_file']} does not exist")

        if args["strategy"] != "custom_pairs":
            args["pair_file"] = args["outs"] / "pairs.txt"

        # Check quality and tile selection options
        if isinstance(args["quality"], str):
            args["quality"] = Quality[args["quality"].upper()]
        elif isinstance(args["quality"], int):
            args["quality"] = Quality(args["quality"])
        elif not isinstance(args["quality"], Quality):
            raise ValueError(f"Invalid quality option: {args['quality']}")
        if isinstance(args["tile_selection"], str):
            args["tile_selection"] = TileSelection[args["tile_selection"].upper()]
        elif isinstance(args["tile_selection"], int):
            args["tile_selection"] = TileSelection(args["tile_selection"])
        elif not isinstance(args["tile_selection"], TileSelection):
            raise ValueError(f"Invalid tile_selection option: {args['tile_selection']}")

        if args["verbose"]:
            change_logger_level(logger.name, "debug")

        if args["openmvg_conf"] is not None:
            args["openmvg_conf"] = Path(args["openmvg_conf"])
            if not args["openmvg_conf"].exists():
                raise ValueError(f"File {args['openmvg_conf']} does not exist")

        if args["camera_options"] is not None:
            if Path(args["camera_options"]).suffix != ".yaml":
                raise ValueError("File passed to --camera_options must be .yaml file")

        if args["upright"] is True:
            if args["strategy"] == "matching_lowres":
                raise ValueError(
                    "With option '--upright' is not possible to use '--strategy matching_lowres', since pairs are chosen with superpoint+lightglue that is not rotation invariant. Use another strategy, e.g. 'bruteforce'."
                )

        # Build configuration dictionary
        cfg = {
            "image_dir": args["images"],
            "output_dir": args["outs"],
            "quality": args["quality"],
            "tile_selection": args["tile_selection"],
            "matching_strategy": args["strategy"],
            "retrieval": args["global_feature"],
            "pair_file": args["pair_file"],
            "overlap": args["overlap"],
            "db_path": args["db_path"],
            "upright": args["upright"],
            "verbose": args["verbose"],
            "graph": args["graph"],
            "skip_reconstruction": args["skip_reconstruction"],
            "openmvg_conf": args["openmvg_conf"],
            "camera_options": args["camera_options"],
        }

        return cfg

    def update_from_yaml(self, path: Path):
        """
        Update the configuration from a YAML file.

        Args:
            path (Path): The path to the YAML file.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If the extractor name in the configuration file does not match with the extractor chosen from CLI or GUI.
            ValueError: If the matcher name in the configuration file does not match with the matcher chosen from CLI or GUI.

        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file {path} not found.")

        print(f"Using a custom configuration file: {path}")

        with open(path, "r") as file:
            cfg = yaml.safe_load(file)

        if "general" in cfg:
            if "quality" in cfg["general"]:
                cfg["general"]["quality"] = Quality[cfg["general"]["quality"].upper()]
            if "tile_selection" in cfg["general"]:
                cfg["general"]["tile_selection"] = TileSelection[cfg["general"]["tile_selection"].upper()]
            if "geom_verification" in cfg["general"]:
                cfg["general"]["geom_verification"] = GeometricVerification[cfg["general"]["geom_verification"].upper()]
            if "tile_size" in cfg["general"]:
                tile_sz = cfg["general"]["tile_size"]
                if isinstance(tile_sz, Tuple):
                    tile_sz = tuple([int(x) for x in tile_sz])
                    cfg["general"]["tile_size"] = tile_sz
                elif isinstance(tile_sz, str):
                    cfg["general"]["tile_size"] = ast.literal_eval(tile_sz)
                elif isinstance(tile_sz, list):
                    cfg["general"]["tile_size"] = tuple(tile_sz)
                else:
                    raise ValueError(
                        f"Invalid tile_size option: {tile_sz} in the configuration file {path}. Valid options are: Tuple[int, int], str, list"
                    )

            self._cfg["general"].update(cfg["general"])

        if "extractor" in cfg:
            if "name" not in cfg["extractor"]:
                logger.error(
                    f"Extractor name is missing in configuration file {path}. Please specify the extractor name for which you want to update the configuration."
                )
                exit(1)
            if cfg["extractor"]["name"] != self._cfg["extractor"]["name"]:
                logger.warning(
                    f"Extractor name in configuration file {path} does not match with the extractor chosen from CLI or GUI. The custom configuration is not set, but matching is run with the default options."
                )
            self._cfg["extractor"].update(cfg["extractor"])
        if "matcher" in cfg:
            if "name" not in cfg["matcher"]:
                logger.error(
                    f"Matcher name is missing in configuration file {path}. Please specify the matcher name for which you want to update the configuration."
                )
                exit(1)
            if cfg["matcher"]["name"] != self._cfg["matcher"]["name"]:
                logger.warning(
                    f"Matcher name in configuration file {path} does not match with the matcher chosen from CLI or GUI. The custom configuration is not set, but matching is run with the default options."
                )
            self._cfg["matcher"].update(cfg["matcher"])

    def print(self):
        """
        Print the configuration settings.

        This method prints the general, extractor, and matcher configuration settings.
        """
        print("Config general:")
        pprint(self.general)
        print("\n")
        print("Config extractor:")
        pprint(self.extractor)
        print("\n")
        print("Config matcher:")
        pprint(self.matcher)
        print("\n")

    def save(self, path: Path = None):
        """Save configuration to file.

        Args:
            path (Path, optional): The path where the configuration will be saved. If not provided, the default
                configuration file path will be used.

        """
        if path is None:
            path = self.config_file
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        cfg = deepcopy(self._cfg)

        # Convert enums to strings
        for k, v in cfg.items():
            for kk, vv in v.items():
                if isinstance(vv, Enum):
                    cfg[k][kk] = vv.name

        # Convert Path objects to strings
        for k, v in cfg.items():
            for kk, vv in v.items():
                if isinstance(vv, Path):
                    cfg[k][kk] = str(vv)

        with open(path, "w") as file:
            json.dump(cfg, file, indent=4)
