import argparse
import shutil
from pathlib import Path

from config import custom_config
from src.deep_image_matching.gui import gui
from src.deep_image_matching.image_matching import ImageMatching
from src.deep_image_matching.io.h5_to_db import export_to_colmap
from src.deep_image_matching.utils import change_logger_level, setup_logger

# TODO: improve configuation manamgement
# The default configuration for each method (extractor and matchers)  must be defined inside each class.
# The user should be able to override the default configuration by passing a dictionary to the constructor.
# The user should be able to chose a configuration from a predefined list of configurations (e.g, confs_zoo) to be sure that the configuration is valid.

logger = setup_logger(log_level="info")


features_zoo = [
    "superpoint",
    "alike",
    "aliked",
    "orb",
    "disk",
    "keynetaffnethardnet",
    "sift",
]
matchers_zoo = [
    "superglue",
    "lightglue",
    "loftr",
    "adalam",
    "smnn",
    "nn",
    "snn",
    "mnn",
    "smnn",
]
retrieval_zoo = ["netvlad", "openibl", "cosplace", "dir"]
matching_strategy = ["bruteforce", "sequential", "retrieval", "custom_pairs"]

confs_zoo = {
    "superpoint+lightglue": {"extractor": "superpoint", "matcher": "lightglue"},
    "disk+lightglue": {"extractor": "disk", "matcher": "lightglue"},
    "superpoint+superglue": {"extractor": "superpoint", "matcher": "superglue"},
    # "aliked+lightglue": {"extractor": "aliked", "matcher": "lightglue"},
    # "sift+lightglue": {"extractor": "sift", "matcher": "lightglue"},
    # "keynetaffnethardnet+adalam": {
    #     "extractor": "keynetaffnethardnet",
    #     "matcher": "adalam",
    # },
    # "loftr": {"extractor": None, "matcher": "loftr"},
    # "alike": {"extractor": "alike", "matcher": None},
    # "orb": {"extractor": "orb", "matcher": None},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matching with hand-crafted and deep-learning based local features and image retrieval."
    )

    parser.add_argument(
        "--gui", action="store_true", help="Run command line interface", default=False
    )

    parser.add_argument("-i", "--images", type=str, help="Input image folder")
    parser.add_argument("-o", "--outs", type=str, help="Output folder", default=None)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Extactor and matcher configuration",
        choices=confs_zoo.keys(),
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
    parser.add_argument("-n", "--max_features", type=int, default=4000)
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
        args.max_features = gui_out["max_features"]

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
        if args.config not in confs_zoo:
            raise ValueError(
                f"Invalid configuration option: {args.config}. Valid options are: {confs_zoo.keys()}"
            )
        args.local_features = confs_zoo[args.config]["extractor"]
        args.matching = confs_zoo[args.config]["matcher"]

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


def main():
    # Parse arguments
    args = parse_args()
    imgs_dir = args.images
    output_dir = args.outs
    matching_strategy = args.strategy
    retrieval_option = args.retrieval
    pair_file = args.pairs
    overlap = args.overlap
    max_features = args.max_features

    # TODO: temporary! Must be replaced by a configuration
    if args.local_features in features_zoo:
        local_features = args.local_features
    else:
        raise ValueError(
            f"Invalid combination of extractor and matcher. Chose one of the following combinations: {confs_zoo.keys()}"
        )
    if args.matching in matchers_zoo:
        matching_method = args.matching

    # Update configuration dictionary
    # TODO: improve configuration management
    custom_config["general"]["output_dir"] = output_dir
    if max_features is not None:
        custom_config["SuperPoint"]["max_keypoints"] = max_features

    # Generate pairs and matching
    img_matching = ImageMatching(
        imgs_dir=imgs_dir,
        matching_strategy=matching_strategy,
        retrieval_option=retrieval_option,
        local_features=local_features,
        matching_method=matching_method,
        pair_file=pair_file,
        custom_config=custom_config,
        overlap=overlap,
    )
    pairs = img_matching.generate_pairs()
    feature_path = img_matching.extract_features()
    match_path = img_matching.match_pairs(feature_path)

    # Plot statistics
    images = img_matching.image_list
    logger.info("Finished matching and exporting")
    logger.info(f"\tProcessed images: {len(images)}")
    logger.info(f"\tProcessed pairs: {len(pairs)}")

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

    # print("using pycolmap...")
    import pycolmap
    from deep_image_matching.hloc.reconstruction import (
        create_empty_db,
        get_image_ids,
        import_images,
    )
    from deep_image_matching.hloc.triangulation import import_features, import_matches2

    database = output_dir / "database_pycolmap.db"
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO

    create_empty_db(database)
    import_images(imgs_dir, database, camera_mode)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, feature_path)
    import_matches2(
        image_ids, database, match_path, skip_geometric_verification=True
    )

    # Backward compatibility
    # Export in colmap format
    # ExportToColmap(
    #     images,
    #     img_matching.width,
    #     img_matching.height,
    #     keypoints,
    #     correspondences,
    #     output_dir,
    # )


if __name__ == "__main__":
    main()

    logger.info("Done")
