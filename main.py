import logging
from importlib import import_module

import deep_image_matching as dim
import yaml

logger = dim.setup_logger("dim")

# Parse arguments from command line
args = dim.parse_cli()

# Build configuration
config = dim.Config(args)
imgs_dir = config.general["image_dir"]
output_dir = config.general["output_dir"]

# Initialize ImageMatcher class
matcher = dim.ImageMatcher(config)

# Run image matching
feature_path, match_path = matcher.run()

# Export in colmap format
with open(config.general["camera_options"], "r") as file:
    camera_options = yaml.safe_load(file)
database_path = output_dir / "database.db"
dim.io.export_to_colmap(
    img_dir=imgs_dir,
    feature_path=feature_path,
    match_path=match_path,
    database_path=database_path,
    camera_options=camera_options,
)

# Visualize view graph
if config.general["graph"]:
    try:
        graph = import_module("deep_image_matching.graph")
        graph.view_graph(database_path, output_dir, imgs_dir)
    except ImportError:
        logger.error("pyvis is not available. Unable to visualize view graph.")

# If --skip_reconstruction is not specified, run reconstruction
# Export in openMVG format
if config.general["openmvg_conf"]:
    with open(config.general["openmvg_conf"], "r") as file:
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
        camera_options=camera_options,
    )

    # Reconstruction with OpenMVG
    reconstruction = import_module("deep_image_matching.openmvg_reconstruction")
    reconstruction.main(
        openmvg_out_path=openmvg_out_path,
        skip_reconstruction=config.general["skip_reconstruction"],
        openmvg_sfm_bin=openmvg_sfm_bin,
    )

# Reconstruction with pycolmap
if not config.general["skip_reconstruction"]:
    use_pycolmap = True
    try:
        # To be sure, check if pycolmap is available, otherwise skip reconstruction
        pycolmap = import_module("pycolmap")
        logger.info(f"Using pycolmap version {pycolmap.__version__}")
    except ImportError:
        logger.error("Pycomlap is not available.")
        use_pycolmap = False

    if use_pycolmap:
        # import reconstruction module
        reconstruction = import_module("deep_image_matching.reconstruction")

        # Optional - You can specify some reconstruction configuration
        # reconst_opts = (
        #     {
        #         "ba_refine_focal_length": True,
        #         "ba_refine_principal_point": False,
        #         "ba_refine_extra_params": False,
        #     },
        # )
        reconst_opts = {}
        refine_intrinsics = config.general["refine_intrinsics"] if "refine_intrinsics" in config.general else True

        # Run reconstruction
        model = reconstruction.pycolmap_reconstruction(
            database_path=output_dir / "database.db",
            sfm_dir=output_dir,
            image_dir=imgs_dir,
            options=reconst_opts,
            verbose=config.general["verbose"],
            refine_intrinsics=refine_intrinsics,
        )
