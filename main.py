import argparse
import shutil
from pathlib import Path

from config import (
    confs,
    extractors_zoo,
    matchers_zoo,
    matching_strategy,
    retrieval_zoo,
)
from src.deep_image_matching import change_logger_level, logger
from src.deep_image_matching.gui import gui
from src.deep_image_matching.image_matching import ImageMatching
from src.deep_image_matching.io.h5_to_db import export_to_colmap

# TODO: Add checks to the combination of extractor and matcher chosen by the user


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matching with hand-crafted and deep-learning based local features and image retrieval."
    )

    parser.add_argument(
        "--gui", action="store_true", help="Run GUI interface", default=False
    )

    parser.add_argument("-i", "--images", type=str, help="Input image folder")
    parser.add_argument("-o", "--outs", type=str, help="Output folder", default=None)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Extactor and matcher configuration",
        choices=confs.keys(),
        default="superpoint+lightglue",
    )
    parser.add_argument(
        "-m",
        "--strategy",
        choices=["bruteforce", "sequential", "retrieval", "custom_pairs"],
        default="sequential",
    )
    parser.add_argument("-p", "--pairs", type=str, default=None)
    parser.add_argument(
        "-v",
        "--overlap",
        type=int,
        help="Image overlap, if using sequential overlap strategy",
        default=1,
    )
    parser.add_argument(
        "-r",
        "--retrieval",
        choices=retrieval_zoo,
        default=None,
    )
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("-V", "--verbose", action="store_true", default=False)

    args = parser.parse_args()

    if args.gui is True:
        gui_out = gui()
        args.images = gui_out["image_dir"]
        args.outs = gui_out["out_dir"]
        args.config = gui_out["config"]
        args.strategy = gui_out["strategy"]
        args.pairs = gui_out["pair_file"]
        args.overlap = gui_out["image_overlap"]

    return args


def initialization():
    args = parse_args()

    # Checks for input arguments
    if args.images is None:
        raise ValueError("--images option is required")
    else:
        args.images = Path(args.images)
        if not args.images.exists() or not args.images.is_dir():
            raise ValueError(f"Folder {args.images} does not exist")
    if args.outs is None:
        args.outs = Path("output") / f"{args.images.name}_{args.config}_{args.strategy}"
    else:
        args.outs = Path(args.outs)

    if args.outs.exists():
        if args.force:
            logger.warning(f"{args.outs} already exists, removing {args.outs}")
            shutil.rmtree(args.outs)
        else:
            raise ValueError(
                f"{args.outs} already exists, use '--force' to overwrite it"
            )
    args.outs.mkdir(parents=True)

    if args.config is None:
        raise ValueError("--config option is required")
    else:
        args.cfg = confs[args.config]
        local_features = confs[args.config]["extractor"]["name"]
        if local_features not in extractors_zoo:
            raise ValueError(
                f"Invalid extractor option: {local_features}. Valid options are: {extractors_zoo}"
            )
        matching = confs[args.config]["matcher"]["name"]
        if matching not in matchers_zoo:
            raise ValueError(
                f"Invalid matcher option: {matching}. Valid options are: {matchers_zoo}"
            )

    if args.strategy is None:
        raise ValueError("--strategy option is required")
    if args.strategy not in matching_strategy:
        raise ValueError(
            f"Invalid strategy option: {args.strategy}. Valid options are: {matching_strategy}"
        )
    if args.strategy == "retrieval":
        if args.retrieval is None:
            raise ValueError(
                "--retrieval option is required when --strategy is set to retrieval"
            )
        elif args.retrieval not in retrieval_zoo:
            raise ValueError(
                f"Invalid retrieval option: {args.retrieval}. Valid options are: {retrieval_zoo}"
            )
    else:
        args.retrieval = None

    if args.strategy == "custom_pairs":
        if args.pairs is None:
            raise ValueError(
                "--pairs option is required when --strategy is set to custom_pairs"
            )
        args.pairs = Path(args.pairs)
        if not args.pairs.exists():
            raise ValueError(f"File {args.pairs} does not exist")
    else:
        args.pairs = args.outs / "pairs.txt"

    if args.strategy == "sequential":
        if args.overlap is None:
            raise ValueError(
                "--overlap option is required when --strategy is set to sequential"
            )
    else:
        args.overlap = None

    if args.verbose:
        change_logger_level("debug")

    return args
    quit()

def main():
    # Parse arguments
    args = initialization()
    imgs_dir = args.images
    output_dir = args.outs
    matching_strategy = args.strategy
    retrieval_option = args.retrieval
    pair_file = args.pairs
    overlap = args.overlap

    # Load configuration
    config = confs[args.config]
    local_features = config["extractor"]["name"]
    matching_method = config["matcher"]["name"]

    # Update configuration dictionary
    config["general"]["output_dir"] = output_dir

    # Generate pairs and matching
    img_matching = ImageMatching(
        imgs_dir=imgs_dir,
        output_dir=output_dir,
        matching_strategy=matching_strategy,
        retrieval_option=retrieval_option,
        local_features=local_features,
        matching_method=matching_method,
        pair_file=pair_file,
        custom_config=config,
        overlap=overlap,
    )
    pair_path = img_matching.generate_pairs()
    img_matching.estimate_rotations()
    feature_path = img_matching.extract_features()
    match_path = img_matching.match_pairs(feature_path)

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

    # Tests using pycolmap
    try:
        import pycolmap
        from deep_image_matching import reconstruction, triangulation

        database = output_dir / "database_pycolmap.db"
        camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO

        reconstruction.create_empty_db(database)
        reconstruction.import_images(imgs_dir, database, camera_mode)
        image_ids = reconstruction.get_image_ids(database)
        triangulation.import_features(image_ids, database, feature_path)
        triangulation.import_matches2(
            image_ids, database, match_path, skip_geometric_verification=True
        )

        # Run geometric verification
        reconstruction.estimation_and_geometric_verification(
            database, pair_path, verbose=True
        )

        # Run reconstruction
        model = reconstruction.run_reconstruction(
            sfm_dir=output_dir, database_path=database, image_dir=imgs_dir, verbose=True
        )
        if reconstruction is not None:
            logger.info(
                f"Reconstruction statistics:\n{model.summary()}"
                + f"\n\tnum_input_images = {len(image_ids)}"
            )

    except:
        logger.error("Error using pycolmap")

    # Plot statistics
    # images = img_matching.image_list
    # logger.info("Finished matching and exporting")
    # logger.info(f"\tProcessed images: {len(images)}")
    # logger.info(f"\tProcessed pairs: {len(pairs)}")


if __name__ == "__main__":
    main()

    logger.info("Done")
