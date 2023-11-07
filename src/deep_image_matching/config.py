from src.deep_image_matching import GeometricVerification, Quality, TileSelection

extractors_zoo = [
    "superpoint",
    "alike",
    "aliked",
    "orb",
    "disk",
    "keynetaffnethardnet",
    "sift",
]
matchers_zoo = [
    "superglue",
    "lightglue",
    "loftr",
    "adalam",
    "smnn",
    "nn",
    "snn",
    "mnn",
    "smnn",
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
            "tiling_overlap": 100,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "gv_threshold": 3,
            "gv_confidence": 0.9999,
        },
        "extractor": {
            "name": "superpoint",
            "keypoint_threshold": 0.005,
            "max_keypoints": 10000,
        },
        "matcher": {
            "name": "lightglue",
            "n_layers": 9,
            "depth_confidence": -1,  # 0.95,  # early stopping, disable with -1
            "width_confidence": -1,  # 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
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
    "orb": {
        "general": {
            "quality": Quality.MEDIUM,
            "tile_selection": TileSelection.NONE,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "tiling_grid": [3, 3],
            "tiling_overlap": 50,
            "gv_threshold": 3,
            "gv_confidence": 0.99999,
        },
        "extractor": {
            "name": "orb",
        },
        "matcher": {"name": "superglue"},
    },
}

# Old configuration system
custom_config = {
    "general": {
        # "detector_and_descriptor": "ALIKE",  # To be used in combination with --detect_and_describe option. ALIKE, ORB, DISK, SuperPoint, KeyNetAffNetHardNet
        "quality": Quality.MEDIUM,
        "force_cpu": False,
        "output_dir": "res",
        "do_viz": False,
        "fast_viz": True,
        "hide_matching_track": False,
        "tile_selection": TileSelection.NONE,
        "tiling_grid": [3, 3],
        "tiling_overlap": 50,
        "do_viz_tiles": True,
        "geometric_verification": GeometricVerification.PYDEGENSAC,
        "gv_threshold": 2,
        "gv_confidence": 0.9999,
        # "kornia_matcher": "smnn",  #'nn' or 'snn' or 'mnn' or 'smnn'
        # "ratio_threshold": 0.95,  # valid range [0-1]
    },
    # SuperPoint
    "SuperPoint": {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_keypoints": 4096,
        "keypoint_threshold": 0.005,
        "remove_borders": 4,
        "fix_sampling": True,
    },
    # ALIKE options
    "ALIKE": {
        "model": "alike-s",
        "device": "cuda",
        "top_k": 15000,
        "scores_th": 0.2,
        "n_limit": 15000,
        "subpixel": True,
    },
    # ORB options (see opencv doc)
    "ORB": {
        "scaleFactor": 1.2,
        "nlevels": 1,
        "edgeThreshold": 1,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": 0,
        "patchSize": 31,
        "fastThreshold": 0,
    },
    # DISK from KORNIA (https://kornia.github.io/)
    "DISK": {
        "max_keypoints": 2000,
    },
    # Key.Net + OriNet + HardNet8 from KORNIA (https://kornia.github.io/)
    "KeyNetAffNetHardNet": {
        "upright": False,
    },
    # Matching options
    # LightGlue (https://github.com/cvg/LightGlue)
    "LightGlue": {
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # 0.95,  # early stopping, disable with -1
        "width_confidence": -1,  # 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    },
    # SuperGlue options
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.3,
    },
    # LoFTR options from KORNIA (https://kornia.github.io/)
    "loftr": {
        "pretrained": "outdoor",
    },
}
