from . import GeometricVerification, Quality, TileSelection

# General configuration for the matching process.
# It defines the quality of the matching process, the tile selection strategy, the tiling grid, the overlap between tiles, the geometric verification method, and the geometric verification parameters.
conf_general = {
    "quality": Quality.HIGH,
    "tile_selection": TileSelection.PRESELECTION,
    "tiling_grid": [3, 3],
    "tiling_overlap": 0,
    "geom_verification": GeometricVerification.PYDEGENSAC,
    "gv_threshold": 4,
    "gv_confidence": 0.9999,
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
            "keypoint_threshold": 0.0001,
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
            "max_num_keypoints": 4000,
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
    "roma": {
        "extractor": {"name": "no_extractor"},
        "matcher": {"name": "roma", "pretrained": "outdoor"},
    },
    "keynetaffnethardnet+kornia_matcher": {
        "extractor": {
            "name": "keynetaffnethardnet",
            "n_features": 2000,
            "upright": True,
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.99},
    },
}


class Config:
    config_general = conf_general
    confs = confs
    confs_names = list(confs.keys())

    def __init__(self):
        pass

    @classmethod
    def from_name(cls, name: str) -> dict:
        cfg = cls.get_config(name)
        cfg["general"] = conf_general
        return cfg

    @staticmethod
    def get_config(name: str) -> dict:
        try:
            return confs[name]
        except KeyError:
            raise ValueError(f"Invalid configuration name: {name}")

    @staticmethod
    def get_config_names() -> list:
        return list(confs.keys())


opt_zoo = {
    "extractors": [
        "superpoint",
        "alike",
        "aliked",
        "disk",
        "keynetaffnethardnet",
        "orb",
        "sift",
        "no_extractor",
    ],
    "matchers": [
        "superglue",
        "lightglue",
        "loftr",
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
    ],
}
