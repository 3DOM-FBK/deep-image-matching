from src.deep_image_matching import Quality, TileSelection, GeometricVerification

custom_config = {
    "general": {
        # "detector_and_descriptor": "ALIKE",  # To be used in combination with --detect_and_describe option. ALIKE, ORB, DISK, SuperPoint, KeyNetAffNetHardNet
        "quality": Quality.HIGH,
        "force_cpu": False,
        "save_dir": "res",
        "do_viz": False,
        "fast_viz": True,
        "hide_matching_track": False,
        "tile_selection": TileSelection.NONE,
        "min_matches_per_tile": 10,
        "tiling_grid": [2, 2],
        "tiling_overlap": 50,
        "do_viz_tiles": True,
        "geometric_verification": GeometricVerification.PYDEGENSAC,
        "gv_threshold": 3,
        "gv_confidence": 0.9999,
        "kornia_matcher": "smnn",  #'nn' or 'snn' or 'mnn' or 'smnn'
        "ratio_threshold": 0.95,  # valid range [0-1]
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
    "DISK": {},
    # SperPoint+LightGlue (https://github.com/cvg/LightGlue)
    "SperPoint+LightGlue": {
        "SuperPoint": {
            "descriptor_dim": 256,
            "nms_radius": 4,
            "max_num_keypoints": 1000,
            "detection_threshold": 0.005,
            "remove_borders": 4,
        },
        "LightGlue": {
            "descriptor_dim": 256,
            "n_layers": 9,
            "num_heads": 4,
            "flash": True,  # enable FlashAttention if available.
            "mp": False,  # enable mixed precision
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
            "weights": None,
        },
    },
    # Key.Net + OriNet + HardNet8 from KORNIA (https://kornia.github.io/)
    "KeyNetAffNetHardNet": {
        "upright": False,
    },
    # SuperPoint+SuperGlue options
    "SperPoint+SuperGlue": {
        "superpoint": {
            "nms_radius": 3,
            "keypoint_threshold": 0.001,
            "max_keypoints": -1,
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.3,
        },
        "force_cpu": False,
    },
    # LoFTR options from KORNIA (https://kornia.github.io/)
    "loftr": {
        "pretrained": "outdoor",
    },
    # LightGlue with SuperPoint features (LG with DISK features to be implemented)
    "lightglue": {},
}
