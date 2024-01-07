import shutil
from pathlib import Path

from deep_image_matching import Quality, TileSelection, change_logger_level, logger

from . import GeometricVerification, Quality, TileSelection

# General configuration for the matching process.
# It defines the quality of the matching process, the tile selection strategy, the tiling grid, the overlap between tiles, the geometric verification method, and the geometric verification parameters.
conf_general = {
    "quality": Quality.HIGH,  # Quality.HIGHEST, Quality.HIGH, Quality.MEDIUM, Quality.LOW, Quality.LOWEST
    "tile_selection": TileSelection.PRESELECTION,  # [TileSelection.NONE, TileSelection.PRESELECTION, TileSelection.GRID]
    "tile_size": (2400, 2000),  # (x, y) or (width, height)
    "tile_overlap": 50,  # in pixels
    "geom_verification": GeometricVerification.PYDEGENSAC,
    "gv_threshold": 2,
    "gv_confidence": 0.999999,
    "preselection_size_max": 2000,
}


# The configuration for DeepImageMatching is defined as a dictionary with the following keys:
# - 'general': general configuration (it is independent from the extractor/matcher and it is defined in the 'conf_general' variable)
# - 'extractor': extractor configuration
# - 'matcher': matcher configuration
# The 'extractor' and 'matcher' values must contain a 'name' key with the name of the extractor/matcher to be used. Additionally, the other parameters of the extractor/matcher can be specified.
# You can get your configuration by accessing the 'confs' dictionary with the name of the configuration (e.g., 'superpoint+lightglue').
confs = {
    "superpoint+lightglue": {
        "extractor": {
            "name": "superpoint",
            "keypoint_threshold": 0.005,
            "max_keypoints": 4096,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "flash": True,  # enable FlashAttention if available.
            "depth_confidence": -1,  # 0.95,  # early stopping, disable with -1
            "width_confidence": -1,  # 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.5,  # match threshold
        },
    },
    "superpoint+lightglue_fast": {
        "extractor": {
            "name": "superpoint",
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 7,
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
        },
    },
    "superpoint+superglue": {
        "extractor": {
            "name": "superpoint",
            "keypoint_threshold": 0.005,
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
            "max_num_keypoints": 4096,
            "detection_threshold": 0.2,
            "nms_radius": 2,
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
    cfg = {
        "general": {},
        "extractor": {},
        "matcher": {},
    }
    default_cli_opts = {
        "gui": False,
        "dir": None,
        "images": None,
        "outs": None,
        "config": "superpoint+lightglue",
        "quality": "high",
        "tiling": "none",
        "strategy": "matching_lowres",
        "pairs": None,
        "overlap": 1,
        "retrieval": None,
        "upright": False,
        "skip_reconstruction": False,
        "force": True,
        "verbose": False,
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
        general = self.parse_config(args)

        # Build configuration dictionary
        self.cfg["general"] = {**conf_general, **general}
        config = self.get_config(args["config"])
        self.cfg["extractor"] = config["extractor"]
        self.cfg["matcher"] = config["matcher"]

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
    def get_extractor_names() -> list:
        return opt_zoo["extractors"]

    @staticmethod
    def get_matcher_names() -> list:
        return opt_zoo["matchers"]

    @staticmethod
    def get_retrieval_names() -> list:
        return opt_zoo["retrieval"]

    @staticmethod
    def get_matching_strategy_names() -> list:
        return opt_zoo["matching_strategy"]

    @staticmethod
    def parse_config(input_args: dict):
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
                shutil.rmtree(args["outs"])
            else:
                raise ValueError(
                    f"{args['outs']} already exists, use '--force' to overwrite it"
                )
        args["outs"].mkdir(parents=True)

        # Check extraction and matching configuration
        if args["config"] is None or args["config"] not in confs:
            raise ValueError(
                "--config option is required and must be a valid configuration (check config.py))"
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

        # Matching strategy
        if args["strategy"] is None:
            raise ValueError("--strategy option is required")
        if args["strategy"] not in opt_zoo["matching_strategy"]:
            raise ValueError(
                f"Invalid strategy option: {args['strategy']}. Valid options are: {opt_zoo['matching_strategy']}"
            )
        if args["strategy"] == "retrieval":
            if args["retrieval"] is None:
                raise ValueError(
                    "--retrieval option is required when --strategy is set to retrieval"
                )
            elif args["retrieval"] not in opt_zoo["retrieval"]:
                raise ValueError(
                    f"Invalid retrieval option: {args['retrieval']}. Valid options are: {opt_zoo['retrieval']}"
                )
        else:
            args["retrieval"] = None
        if args["strategy"] == "custom_pairs":
            if args["pairs"] is None:
                raise ValueError(
                    "--pairs option is required when --strategy is set to custom_pairs"
                )
            args["pairs"] = Path(args["pairs"])
            if not args["pairs"].exists():
                raise ValueError(f"File {args['pairs']} does not exist")
        else:
            args["pairs"] = args["outs"] / "pairs.txt"
        if args["strategy"] == "sequential":
            if args["overlap"] is None:
                raise ValueError(
                    "--overlap option is required when --strategy is set to sequential"
                )
        else:
            args["overlap"] = None

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
            "upright": args["upright"],
            "verbose": args["verbose"],
            "skip_reconstruction": args["skip_reconstruction"],
        }

        return cfg
