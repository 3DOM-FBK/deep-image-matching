from src.deep_image_matching import Quality, TileSelection, GeometricVerification

custom_config = {
    "general": {
        # "detector_and_descriptor": "ALIKE",  # To be used in combination with --detect_and_describe option. ALIKE, ORB, DISK, SuperPoint, KeyNetAffNetHardNet
        "quality": Quality.HIGH,
        "geometric_verification": GeometricVerification.PYDEGENSAC,
        "gv_threshold": 1.5,
        "gv_confidence": 0.999,
        "tile_selection": TileSelection.NONE,
        "force_cpu": False,
        "save_dir": "res",
        "min_matches_per_tile": 5,
        "tiling_grid": [3, 2],
        "tiling_overlap": 200,
        "do_viz_tiles": False,
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
    # SuperPoint from LightGlue repository (https://github.com/cvg/LightGlue)
    "SuperPoint": {},
    # Key.Net + OriNet + HardNet8 from KORNIA (https://kornia.github.io/)
    "KeyNetAffNetHardNet": {
        "upright": False,
    },
    # SuperGlue options
    "superglue": {
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
