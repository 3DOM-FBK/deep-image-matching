import os
import shutil
from importlib import import_module
from pathlib import Path
from src.deep_image_matching import logger, timer
from src.deep_image_matching.image_matching import ImageMatching
from src.deep_image_matching.io.h5_to_db import export_to_colmap
from src.deep_image_matching.io.h5_to_openmvg import export_to_openmvg
from src.deep_image_matching.parser import parse_config

# Parse arguments
config = parse_config()
imgs_dir = config["general"]["image_dir"]
output_dir = config["general"]["output_dir"]
matching_strategy = config["general"]["matching_strategy"]
retrieval_option = config["general"]["retrieval"]
pair_file = config["general"]["pair_file"]
overlap = config["general"]["overlap"]
upright = config["general"]["upright"]
extractor = config["extractor"]["name"]
matcher = config["matcher"]["name"]

# Initialize ImageMatching class
img_matching = ImageMatching(
    imgs_dir=imgs_dir,
    output_dir=output_dir,
    matching_strategy=matching_strategy,
    retrieval_option=retrieval_option,
    local_features=extractor,
    matching_method=matcher,
    pair_file=pair_file,
    custom_config=config,
    overlap=overlap,
)

# Generate pairs to be matched
pair_path = img_matching.generate_pairs()
timer.update("generate_pairs")

# Try to rotate images so they will be all "upright", useful for deep-learning approaches that usually are not rotation invariant
if upright:
    img_matching.rotate_upright_images()
    timer.update("rotate_upright_images")

# Extract features
feature_path = img_matching.extract_features()
timer.update("extract_features")

# Matching
match_path = img_matching.match_pairs(feature_path)
timer.update("matching")

# Features are extracted on "upright" images, this function report back images on their original orientation
if upright:
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

# Export in openMVG format
openmvg_out_path = output_dir / "openmvg"
export_to_openmvg(
    img_dir=imgs_dir,
    feature_path=feature_path,
    match_path=match_path,
    openmvg_out_path=openmvg_out_path,
    openmvg_sfm_bin=Path(r"C:\Users\lmorelli\Desktop\openMVG\ReleaseV1.6.Halibut.WindowsBinaries_VS2017"),
)
timer.update("export_to_openMVG")

# If --skip_reconstruction is not specified, run reconstruction
if not config["general"]["skip_reconstruction"]:
    # For debugging purposes, copy images to output folder
    if config["general"]["verbose"]:
        shutil.copytree(imgs_dir, output_dir / "images", dirs_exist_ok=True)

    use_pycolmap = True
    try:
        pycolmap = import_module("pycolmap")
    except ImportError:
        logger.error("Pycomlap is not available.")
        use_pycolmap = False

    if use_pycolmap:
        # import reconstruction module
        reconstruction = import_module("src.deep_image_matching.reconstruction")

        # Define database path and camera mode
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
        #
        # cam0 = pycolmap.Camera(
        #    model="PINHOLE",
        #    width=1500,
        #    height=1000,
        #    params=[1500, 1500, 750, 500],
        #)
        # cam1 = pycolmap.Camera(
        #     model="SIMPLE_PINHOLE",
        #     width=6012,
        #     height=4008,
        #     params=[9.267, 3.053, 1.948],
        # )
        # cam2 = pycolmap.Camera(
        #     model="SIMPLE_PINHOLE",
        #     width=6012,
        #     height=4008,
        #     params=[6.621, 3.013, 1.943],
        # )
        # cameras = [cam0] # or cameras = [cam1, cam2]
        cameras = None

        # Optional - You can specify some reconstruction configuration
        # reconst_opts = (
        #     {
        #         "ba_refine_focal_length": False,
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
            verbose=config["general"]["verbose"],
        )

        timer.update("pycolmap reconstruction")

    else:
        logger.warning("Reconstruction with COLMAP CLI is not implemented yet.")

timer.print("Deep Image Matching")
