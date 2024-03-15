from importlib import import_module

import yaml
from deep_image_matching import logger, timer
from deep_image_matching.config import Config
from deep_image_matching.image_matching import ImageMatching
from deep_image_matching.io.h5_to_db import export_to_colmap
from deep_image_matching.io.h5_to_openmvg import export_to_openmvg
from deep_image_matching.parser import parse_cli

# Parse arguments from command line
args = parse_cli()

# Build configuration
config = Config(args)

# For simplicity, save some of the configuration parameters in variables.
imgs_dir = config.general["image_dir"]
output_dir = config.general["output_dir"]

# Initialize ImageMatching class
img_matching = ImageMatching(
    imgs_dir=imgs_dir,
    output_dir=output_dir,
    matching_strategy=config.general["matching_strategy"],
    local_features=config.extractor["name"],
    matching_method=config.matcher["name"],
    pair_file=config.general["pair_file"],
    retrieval_option=config.general["retrieval"],
    overlap=config.general["overlap"],
    existing_colmap_model=config.general["db_path"],
    custom_config=config.as_dict(),
)

# Generate pairs to be matched
pair_path = img_matching.generate_pairs()
timer.update("generate_pairs")

# Try to rotate images so they will be all "upright", useful for deep-learning approaches that usually are not rotation invariant
if config.general["upright"]:
    img_matching.rotate_upright_images()
    timer.update("rotate_upright_images")

# Extract features
feature_path = img_matching.extract_features()
timer.update("extract_features")

# Matching
match_path = img_matching.match_pairs(feature_path)
timer.update("matching")

# If features have been extracted on "upright" images, this function bring features back to their original image orientation
if config.general["upright"]:
    img_matching.rotate_back_features(feature_path)
    timer.update("rotate_back_features")

# Export in colmap format
with open(config.general["camera_options"], "r") as file:
    camera_options = yaml.safe_load(file)
database_path = output_dir / "database.db"
export_to_colmap(
    img_dir=imgs_dir,
    feature_path=feature_path,
    match_path=match_path,
    database_path=database_path,
    camera_options=camera_options,
)
timer.update("export_to_colmap")

# Visualize view graph
if config.general["graph"]:
    try:
        graph = import_module("deep_image_matching.graph")
        graph.view_graph(database_path, output_dir, imgs_dir)
        timer.update("show view graph")
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

    export_to_openmvg(
        img_dir=imgs_dir,
        feature_path=feature_path,
        match_path=match_path,
        openmvg_out_path=openmvg_out_path,
        openmvg_sfm_bin=openmvg_sfm_bin,
        openmvg_database=openmvg_database,
        camera_options=camera_options,
    )
    timer.update("export_to_openMVG")

    # Reconstruction with OpenMVG
    reconstruction = import_module("deep_image_matching.openmvg_reconstruction")
    reconstruction.main(
        openmvg_out_path=openmvg_out_path,
        skip_reconstruction=config.general["skip_reconstruction"],
        openmvg_sfm_bin=openmvg_sfm_bin,
    )

    timer.update("SfM with openMVG")

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

        # Run reconstruction
        model = reconstruction.main(
            database=output_dir / "database.db",
            image_dir=imgs_dir,
            sfm_dir=output_dir,
            reconst_opts=reconst_opts,
            verbose=config.general["verbose"],
        )

        timer.update("pycolmap reconstruction")

# Print timing
timer.print("Deep Image Matching")
