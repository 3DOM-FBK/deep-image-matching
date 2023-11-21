from src.deep_image_matching import GeometricVerification, Quality, TileSelection

extractors_zoo = [
    "superpoint",
    "alike",
    "aliked",
    "disk",
    "keynetaffnethardnet",
    "orb",
    "sift",
    "no_extractor",
]
matchers_zoo = [
    "superglue",
    "lightglue",
    "loftr",
    "adalam",
    "kornia_matcher",
    "roma",
]
retrieval_zoo = ["netvlad", "openibl", "cosplace", "dir"]
matching_strategy = ["bruteforce", "sequential", "retrieval", "custom_pairs"]


def get_config(name: str):
    try:
        return confs[name]
    except KeyError:
        raise ValueError(f"Invalid configuration name: {name}")


# The confiugration is defined by a desired name (e.g., "superpoint+lightglue") and it must be a dictionary with the following keys:
# - 'general': general configuration
# - 'extractor': extractor configuratio
# - 'matcher': matcher configuration
# The 'extractor' and 'matcher' values must contain a 'name' key with the name of the extractor/matcher to be used. Additionally, the other parameters of the extractor/matcher can be specified.
# Each configration can be retrieved by calling the function get_config(name)
# The configuration systme is ispired from that to HLOC
confs = {
    "superpoint+lightglue": {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "tiling_grid": [3, 3],
            "tiling_overlap": 0,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "gv_threshold": 4,
            "gv_confidence": 0.9999,
            "preselction_resize_max": 2000,
        },
        "extractor": {
            "name": "superpoint",
            "keypoint_threshold": 0.0001,
            "max_keypoints": 10000,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "depth_confidence": -1,  # 0.95,  # early stopping, disable with -1
            "width_confidence": -1,  # 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.5,  # match threshold
        },
    },
    "superpoint+lightglue_tiling": {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "gv_threshold": 3,
            "gv_confidence": 0.9999,
        },
        "extractor": {
            "name": "superpoint",
            "keypoint_threshold": 0.005,
            "max_keypoints": 4096,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "depth_confidence": -1,  # 0.95,  # early stopping, disable with -1
            "width_confidence": -1,  # 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
        },
    },
    "superpoint+lightglue_fast": {
        "general": {
            "quality": Quality.MEDIUM,
            "tile_selection": TileSelection.NONE,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
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
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
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
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
        "extractor": {
            "name": "disk",
            "max_keypoints": 4096,
        },
        "matcher": {
            "name": "lightglue",
        },
    },
    "aliked+lightglue": {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
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
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
        "extractor": {
            "name": "orb",
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "snn"},
    },
    "sift+kornia_matcher": {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
        "extractor": {
            "name": "sift",
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.99},
    },
    "loftr": {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.NONE,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
        "extractor": {"name": "no_extractor"},
        "matcher": {"name": "loftr", "pretrained": "outdoor"},
    },
    "roma": {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.NONE,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
        "extractor": {"name": "no_extractor"},
        "matcher": {"name": "roma", "pretrained": "outdoor"},
    },
    "keynetaffnethardnet+kornia_matcher": {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.NONE,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
        "extractor": {
            "name": "keynetaffnethardnet",
            "n_features": 2000,
            "upright" : True,
        },
        "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.99},
    },
}