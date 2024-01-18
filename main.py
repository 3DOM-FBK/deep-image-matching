from importlib import import_module

from deep_image_matching import logger, timer
from deep_image_matching.config import Config
from deep_image_matching.image_matching import ImageMatching
from deep_image_matching.io.h5_to_db import export_to_colmap
from deep_image_matching.parser import parse_cli

from scripts.metashape.metashape_from_dim import export_to_metashape

# Hard-coded flag for exporting the solution to Metashape (TODO: move to the configuration settings)
do_export_to_metashape = False

# Parse arguments from command line
args = parse_cli()

# Build configuration
config = Config(args)

# If you know what you are doing, you can update some config parameters directly updating the config dictionary (check the file config.py in the scr folder for the available parameters)
# - General configuration
config.general["min_inliers_per_pair"] = 10
config.general["min_inlier_ratio_per_pair"] = 0.2

# - SuperPoint configuration
config.extractor["max_keypoints"] = 8000

# - LightGue configuration
config.matcher["filter_threshold"] = 0.1

# Save configuration to a json file in the output directory
config.save_config()

# For simplicity, save some of the configuration parameters in variables.
imgs_dir = config.general["image_dir"]
output_dir = config.general["output_dir"]
matching_strategy = config.general["matching_strategy"]
extractor = config.extractor["name"]
matcher = config.matcher["name"]

# Initialize ImageMatching class
img_matching = ImageMatching(
    imgs_dir=imgs_dir,
    output_dir=output_dir,
    matching_strategy=matching_strategy,
    local_features=extractor,
    matching_method=matcher,
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
database_path = output_dir / "database.db"
export_to_colmap(
    img_dir=imgs_dir,
    feature_path=feature_path,
    match_path=match_path,
    database_path=database_path,
    camera_model="simple-radial",
    single_camera=True,
)
timer.update("export_to_colmap")

# If --skip_reconstruction is not specified, run reconstruction
if not config.general["skip_reconstruction"]:
    use_pycolmap = True
    try:
        pycolmap = import_module("pycolmap")
    except ImportError:
        logger.error("Pycomlap is not available.")
        use_pycolmap = False

    if use_pycolmap:
        # import reconstruction module
        reconstruction = import_module("src.deep_image_matching.reconstruction")

        # Define database path
        database = output_dir / "database_pycolmap.db"

        # Define how pycolmap create the cameras. Possible CameraMode are:
        # CameraMode.AUTO: infer the camera model based on the image exif
        # CameraMode.PER_FOLDER: create a camera for each folder in the image directory
        # CameraMode.PER_IMAGE: create a camera for each image in the image directory
        # CameraMode.SINGLE: create a single camera for all images
        camera_mode = pycolmap.CameraMode.AUTO

        # Optional - You can manually define the cameras parameters (refer to https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h).
        # Note, that the cameras are first detected with the CameraMode specified and then overwitten with the custom model. Therefore, you MUST provide the SAME NUMBER of cameras and with the SAME ORDER in which the cameras appear in the COLMAP database.
        # To see the camera number and order, you can run the reconstruction a first time with the AUTO camera mode (and without manually define the cameras) and see the list of cameras in the database with
        #   print(list(model.cameras.values()))
        # or opening the database with the COLMAP gui.
        #
        # OPENCV camera models and number of parameters to be used
        #    SIMPLE_PINHOLE: f, cx, cy
        #    PINHOLE: fx, fy, cx, cy
        #    SIMPLE_RADIAL: f, cx, cy, k
        #    RADIAL: f, cx, cy, k1, k2
        #    OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        #    OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
        #    FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        #    FOV: fx, fy, cx, cy, omega
        #    SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
        #    RADIAL_FISHEYE: f, cx, cy, k1, k2
        #    THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
        # NOTE: Use the SIMPLE-PINHOLE camera model if you want to export the solution to Metashape, as there are some bugs in COLMAP (or pycolamp) when exporting the solution in the Bundler format.
        # e.g., using FULL-OPENCV camera model, the principal point is not exported correctly and the tie points are wrong in Metashape.
        #
        # cam0 = pycolmap.Camera(
        #    model="PINHOLE",
        #    width=1500,
        #    height=1000,
        #    params=[1500, 1500, 750, 500],
        # )
        # cam1 = pycolmap.Camera(
        #     model="SIMPLE_PINHOLE",
        #     width=6012,
        #     height=4008,
        #     params=[9.267, 3.053, 1.948],
        # )
        # cameras = [cam0] # or cameras = [cam0, cam1]
        cameras = []

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
            database=database,
            image_dir=imgs_dir,
            feature_path=feature_path,
            match_path=match_path,
            pair_path=pair_path,
            sfm_dir=output_dir,
            camera_mode=camera_mode,
            cameras=cameras,
            skip_geometric_verification=True,
            reconst_opts=reconst_opts,
            verbose=config.general["verbose"],
        )

        timer.update("pycolmap reconstruction")

    else:
        logger.warning("Reconstruction with COLMAP CLI is not implemented yet.")


if model and do_export_to_metashape:
    # Hard-coded parameters for Metashape # TODO: improve this implementation.
    # This is now given only as an example for how to use the export_to_metashape function.
    project_dir = config.general["output_dir"] / "metashape"
    project_name = config.general["output_dir"].name + ".psx"
    project_path = project_dir / project_name

    rec_dir = config.general["output_dir"] / "reconstruction"
    bundler_file_path = rec_dir / "bundler.out"
    bundler_im_list = rec_dir / "bundler_list.txt"

    prm_to_optimize = {
        "f": True,
        "cx": True,
        "cy": True,
        "k1": True,
        "k2": True,
        "k3": True,
        "k4": False,
        "p1": True,
        "p2": True,
        "b1": False,
        "b2": False,
        "tiepoint_covariance": True,
    }
    export_to_metashape(
        project_path=project_path,
        images_dir=config.general["image_dir"],
        bundler_file_path=bundler_file_path.resolve(),
        bundler_im_list=bundler_im_list.resolve(),
        prm_to_optimize=prm_to_optimize,
    )


timer.print("Deep Image Matching")
