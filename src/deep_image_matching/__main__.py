from . import timer
from .config import Config
from .image_matching import ImageMatching
from .io.h5_to_db import export_to_colmap
from .parser import parse_cli


def run_pipeline():

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


if __name__ == "__main__":
    run_pipeline()
