import os
import time
from pathlib import Path

import yaml

import deep_image_matching as dim
from deep_image_matching.utils.loftr_roma_to_multiview import LoftrRomaToMultiview

start_time = time.time()

logger = dim.setup_logger("dim")

# Define the configuration parameters
args = {
    "dir": "./assets/example_cyprus",
    "pipeline": "superpoint+lightglue", # ["superpoint+lightglue", "superpoint+lightglue_fast", "superpoint+superglue", "superpoint+kornia_matcher", "disk+lightglue", "aliked+lightglue", "orb+kornia_matcher", "sift+kornia_matcher", "loftr", "se2loftr", "roma", "srif", "keynetaffnethardnet+kornia_matcher", "dedode+kornia_matcher"]
    "strategy": "bruteforce", # ["matching_lowres", "bruteforce", "sequential", "retrieval", "custom_pairs", "covisibility"]
    "quality": "medium", # ["lowest", "low", "medium", "high", "highest"]
    "tiling": "none", # ["none", "preselection", "grid", "exhaustive"]
    "camera_options": "./assets/example_cyprus/cameras.yaml",
    "openmvg": None,
    "force": True,  # Remove existing features and matches
    "skip_reconstruction": False,
    "graph": True,
    #"upright": "custom", # ["custom", "2clusters", "exif"] With "custom" option, rotations must be specified in ./config/rotations.txt
    #"config_file": "./config_superpoint_lightglue.yaml", # Path to custom config file (YAML format) for matcher
    "verbose": False,
}

# Alternatively, you can parse the parameters from the command line with
# from deep_image_matching.parser import parse_cli
# args = parse_cli()

# Build the configuration from the parameters or command line arguments
config = dim.Config(args)
imgs_dir = config.general["image_dir"]
output_dir = config.general["output_dir"]

# Initialize ImageMatcher class
matcher = dim.ImageMatcher(config)

# Run image matching
feature_path, match_path = matcher.run()

# Export in colmap format
database_path = output_dir / "database.db"
if database_path.exists():
    database_path.unlink()
dim.io.export_to_colmap(
    img_dir=imgs_dir,
    feature_path=feature_path,
    match_path=match_path,
    database_path=database_path,
    camera_config_path=config.general["camera_options"],
)


if matcher.matching in ["loftr", "se2loftr", "roma", "srif"]:
    images = os.listdir(imgs_dir)
    image_format = Path(images[0]).suffix
    LoftrRomaToMultiview(
        input_dir=feature_path.parent,
        output_dir=feature_path.parent,
        image_dir=imgs_dir,
        img_ext=image_format,
    )

# Visualize view graph
if config.general["graph"]:
    try:
        dim.graph.view_graph(database_path, output_dir, imgs_dir)
    except Exception as e:
        logger.error(f"Unable to visualize view graph: {e}")

# If --skip_reconstruction is not specified, run reconstruction with pycolmap
if not config.general["skip_reconstruction"]:
    # Optional - You can specify some reconstruction configuration
    # reconst_opts = (
    #     {
    #         "ba_refine_focal_length": True,
    #         "ba_refine_principal_point": False,
    #         "ba_refine_extra_params": False,
    #     },
    # )
    reconst_opts = {}
    model = dim.reconstruction.incremental_reconstruction(
        database_path=output_dir / "database.db",
        image_dir=imgs_dir,
        sfm_dir=output_dir / "reconstruction",
        reconstruction_options=reconst_opts,
        refine_intrinsics=True,
        ignore_two_view_tracks=True,
        filter_min_tri_angle=None,
        export_ply=True,
        export_text=True,
        export_bundler=False,
    )


# Export in openMVG format
if config.general["openmvg_conf"]:
    with open(config.general["openmvg_conf"]) as file:
        openmvgcfg = yaml.safe_load(file)
    openmvg_sfm_bin = openmvgcfg["general"]["path_to_binaries"]
    openmvg_database = openmvgcfg["general"]["openmvg_database"]
    openmvg_out_path = output_dir / "openmvg"
    dim.io.export_to_openmvg(
        img_dir=imgs_dir,
        feature_path=feature_path,
        match_path=match_path,
        openmvg_out_path=openmvg_out_path,
        openmvg_sfm_bin=openmvg_sfm_bin,
        openmvg_database=openmvg_database,
        camera_config_path=config.general["camera_options"],
    )

    # If skip_reconstruction is not specified, run OpenMVG reconstruction
    if not config.general["skip_reconstruction"]:
        from deep_image_matching.openmvg import openmvg_reconstruction

        openmvg_reconstruction(
            openmvg_out_path=openmvg_out_path,
            skip_reconstruction=config.general["skip_reconstruction"],
            openmvg_sfm_bin=openmvg_sfm_bin,
        )
