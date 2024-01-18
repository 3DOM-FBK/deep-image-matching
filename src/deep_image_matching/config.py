import json
from copy import deepcopy
from enum import Enum
from pathlib import Path
from pprint import pprint

import yaml

from deep_image_matching import (
    GeometricVerification,
    Quality,
    TileSelection,
    change_logger_level,
    logger,
)

# General configuration for the matching process.
# It defines the quality of the matching process, the tile selection strategy, the tiling grid, the overlap between tiles, the geometric verification method, and the geometric verification parameters.
conf_general = {
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
    "geom_verification": GeometricVerification.PYDEGENSAC,
    "gv_threshold": 4,
    "gv_confidence": 0.99999,
    # Minimum number of inliers matches and minumum inlier ratio per pair
    "min_inliers_per_pair": 15,
    "min_inlier_ratio_per_pair": 0.25,
    # Even if the features are extracted by tiles, you can try to match the features of the entire image first (if the number of features is not too high and they can fit into memory). Default is False.
    "try_match_full_images": False,
}


# The configuration for DeepImageMatching is defined as a dictionary with the following keys:
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
    "superpoint+lightglue_fast": {
        "extractor": {
            "name": "superpoint",
            "nms_radius": 3,
            "keypoint_threshold": 0.001,
            "max_keypoints": 1024,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 7,
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
            "match_threshold": 0.3,
        },
    },
    "disk+lightglue": {
        "extractor": {
            "name": "disk",
            "max_keypoints": 4096,
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
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.99},
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
    "dedode": {
        "extractor": {
            "name": "dedode",
            "n_features": 1000,
            "upright": False,
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.99},
    },
}

opt_zoo = {
    "extractors": [
        "superpoint",
        "alike",
        "aliked",
        "disk",
        "dedode",
        "keynetaffnethardnet",
        "orb",
        "sift",
        "no_extractor",
    ],
    "matchers": [
        "superglue",
        "lightglue",
        "loftr",
        "se2loftr",
        "adalam",
        "kornia_matcher",
        "roma",
    ],
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
    default_cli_opts = {
        "gui": False,
        "dir": None,
        "images": None,
        "outs": None,
        "config": None,
        "quality": "high",
        "tiling": "none",
        "strategy": "matching_lowres",
        "pairs": None,
        "overlap": None,
        "retrieval": None,
        "db_path": None,
        "upright": False,
        "skip_reconstruction": False,
        "force": True,
        "verbose": False,
    }
    cfg = {
        "general": {},
        "extractor": {},
        "matcher": {},
    }

    @property
    def general(self):
        return self.cfg["general"]

    @property
    def extractor(self):
        return self.cfg["extractor"]

    @property
    def matcher(self):
        return self.cfg["matcher"]

    def __init__(self, args: dict):
        # Parse input arguments
        user_conf = self.parse_user_config(args)

        # Build configuration dictionary
        self.cfg["general"] = {**conf_general, **user_conf}
        features_config = self.get_config(args["config"])
        self.cfg["extractor"] = features_config["extractor"]
        self.cfg["matcher"] = features_config["matcher"]

        self._config_file = user_conf["output_dir"] / "config.json"
        self.save(self._config_file)

    def as_dict(self):
        return self.cfg

    @staticmethod
    def get_config(name: str) -> dict:
        try:
            return confs[name]
        except KeyError:
            raise ValueError(f"Invalid configuration name: {name}")

    @staticmethod
    def get_config_names() -> list:
        return list(confs.keys())

    @staticmethod
    def get_matching_strategy_names() -> list:
        return opt_zoo["matching_strategy"]

    @staticmethod
    def get_extractor_names() -> list:
        return opt_zoo["extractors"]

    @staticmethod
    def get_matcher_names() -> list:
        return opt_zoo["matchers"]

    @staticmethod
    def get_retrieval_names() -> list:
        return opt_zoo["retrieval"]

    @staticmethod
    def parse_user_config(input_args: dict):
        """Do checks on the input arguments and return the configuration dictionary with the following keys: general, extractor, matcher"""

        args = {**Config.default_cli_opts, **input_args}

        # Check and defines all input/output folders
        if not args["dir"]:
            raise ValueError(
                "Invalid project directory. A valid folder must be passed to '--dir' option."
            )
        else:
            args["dir"] = Path(args["dir"])
            if not args["dir"].exists() or not args["dir"].is_dir():
                raise ValueError(f"Folder {args['dir']} does not exist")

        if args["images"] is None:
            args["images"] = args["dir"] / "images"
            if not args["images"].exists() or not args["images"].is_dir():
                raise ValueError(
                    "'images' folder not found in project directory. Create an 'images' folder containing all the images or use '--images' option to specify a valid folder."
                )
        else:
            args["images"] = Path(args["images"])
            if not args["images"].exists() or not args["images"].is_dir():
                raise ValueError(
                    f"Invalid images folder {args['images']}. Direcotry does not exist"
                )

        args["outs"] = (
            args["dir"]
            / f"results_{args['config']}_{args['strategy']}_quality_{args['quality']}"
        )
        if args["outs"].exists():
            if args["force"]:
                logger.warning(
                    f"{args['outs']} already exists, removing {args['outs']}"
                )
                # shutil.rmtree(args["outs"])
            # else:
            #     raise ValueError(
            #         f"{args['outs']} already exists, use '--force' to overwrite it"
            #     )
        # args["outs"].mkdir(parents=True)
        args["outs"].mkdir(parents=True, exist_ok=True)

        # Check extraction and matching configuration
        if args["config"] is None or args["config"] not in confs:
            raise ValueError(
                "Invalid config. --config option is required and must be a valid configuration. Check --help for details"
            )
        extractor = confs[args["config"]]["extractor"]["name"]
        if extractor not in opt_zoo["extractors"]:
            raise ValueError(
                f"Invalid extractor option: {extractor}. Valid options are: {opt_zoo['extractors']}"
            )
        matcher = confs[args["config"]]["matcher"]["name"]
        if matcher not in opt_zoo["matchers"]:
            raise ValueError(
                f"Invalid matcher option: {matcher}. Valid options are: {opt_zoo['matchers']}"
            )

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
                raise ValueError(
                    "Invalid overlap. --overlap must be a positive integer greater than 0"
                )
            elif args["overlap"] >= num_imgs - 1:
                raise ValueError(
                    "Invalid overlap. --overlap must be less than the number of images-1"
                )

        elif args["strategy"] == "retrieval":
            if args["retrieval"] is None:
                raise ValueError(
                    "--retrieval option is required when --strategy is set to retrieval"
                )
            elif args["retrieval"] not in opt_zoo["retrieval"]:
                raise ValueError(
                    f"Invalid retrieval option: {args['retrieval']}. Valid options are: {opt_zoo['retrieval']}"
                )

        elif args["strategy"] == "covisibility":
            if args["db_path"] is None:
                raise ValueError(
                    "--db_path option is required when --strategy is set to covisibility"
                )
            args["db_path"] = Path(args["db_path"])
            if not args["db_path"].exists():
                raise ValueError(f"File {args['db_path']} does not exist")

        elif args["strategy"] == "custom_pairs":
            if args["pairs"] is None:
                raise ValueError(
                    "--pairs option is required when --strategy is set to custom_pairs"
                )
            args["pairs"] = Path(args["pairs"])
            if not args["pairs"].exists():
                raise ValueError(f"File {args['pairs']} does not exist")

        if args["strategy"] != "custom_pairs":
            args["pairs"] = args["outs"] / "pairs.txt"

        if args["verbose"]:
            change_logger_level(logger.name, "debug")

        # Build configuration dictionary
        cfg = {
            "image_dir": args["images"],
            "output_dir": args["outs"],
            "quality": Quality[args["quality"].upper()],
            "tile_selection": TileSelection[args["tiling"].upper()],
            "matching_strategy": args["strategy"],
            "retrieval": args["retrieval"],
            "pair_file": args["pairs"],
            "overlap": args["overlap"],
            "db_path": args["db_path"],
            "upright": args["upright"],
            "verbose": args["verbose"],
            "skip_reconstruction": args["skip_reconstruction"],
        }

        return cfg

    def update_from_yaml(self, path: Path):
        with open(path, "r") as file:
            cfg = yaml.safe_load(file)

        self.cfg["general"].update(cfg["general"])
        self.cfg["extractor"].update(cfg["extractor"])
        self.cfg["matcher"].update(cfg["matcher"])

    def print(self):
        print("Config general:")
        pprint(self.general)
        print("\n")
        print("Config extractor:")
        pprint(self.extractor)
        print("\n")
        print("Config matcher:")
        pprint(self.matcher)

    def save(self, path: Path = None):
        """Save configuration to file"""

        if path is None:
            path = self._config_file
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        cfg = deepcopy(self.cfg)

        # Convert enums to strings
        for k, v in cfg.items():
            for kk, vv in v.items():
                if isinstance(vv, Enum):
                    cfg[k][kk] = vv.name

        # Conver Path objects to strings
        for k, v in cfg.items():
            for kk, vv in v.items():
                if isinstance(vv, Path):
                    cfg[k][kk] = str(vv)

        with open(path, "w") as file:
            json.dump(cfg, file, indent=4)
