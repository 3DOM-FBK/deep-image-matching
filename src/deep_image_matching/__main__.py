from importlib import import_module

from src.deep_image_matching import logger, timer
from src.deep_image_matching.image_matching import ImageMatching
from src.deep_image_matching.io.h5_to_db import export_to_colmap
from src.deep_image_matching.parser import parse_config


def main():
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

    # If --skip_reconstruction is not specified, run reconstruction
    if not config["general"]["skip_reconstruction"]:
        use_pycolmap = True
        try:
            pycolmap = import_module("pycolmap")
            reconstruction = import_module("deep_image_matching.reconstruction")
        except ImportError:
            logger.error("Pycomlap is not available.")
            use_pycolmap = False

        if use_pycolmap:

            # Define database path and camera mode
            database = output_dir / "database_pycolmap.db"
            camera_mode = pycolmap.CameraMode.AUTO

            # Run reconstruction
            model = reconstruction.main(
                database=database,
                image_dir=imgs_dir,
                feature_path=feature_path,
                match_path=match_path,
                pair_path=pair_path,
                output_dir=output_dir,
                camera_mode=camera_mode,
                skip_geometric_verification=True,
                verbose=False, 
            )

            timer.update("pycolmap reconstruction")

    timer.print("Deep Image Matching")


if __name__ == "__main__":
    main()
