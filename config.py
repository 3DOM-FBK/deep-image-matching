from lib.deep_image_matcher import (Quality, TileSelection, GeometricVerification)
custom_config = {

    "general" : {
        "detector_and_descriptor" : "KeyNetAffNetHardNet",
        "quality" : Quality.HIGH, # 'Quality.HIGH'
        "tile_selection" : TileSelection.NONE,
        "grid" : [3,2],
        "overlap" : 200,
        "min_matches_per_tile" : 5,
        "do_viz_tiles" : False,
        "save_dir" : "./res/superglue_matches",
        "geometric_verification" : GeometricVerification.PYDEGENSAC,
        "threshold" : 1.5,
    },


    "ALIKE" : {
        "model": "alike-s",
        "device": "cuda",
        "top_k": 15000,
        "scores_th": 0.2,
        "n_limit": 15000,
        "subpixel": True,
    },


    "ORB" : {
        "scaleFactor": 1.2,
        "nlevels": 1,
        "edgeThreshold": 1,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": 0,
        "patchSize": 31,
        "fastThreshold": 0,
            },

    "DISK" : {

    },

    "SuperPoint" : {

    },

    "KeyNetAffNetHardNet" : {

    }

}