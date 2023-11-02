import argparse
from pathlib import Path

from config import custom_config
from src.deep_image_matching.gui import gui
from src.deep_image_matching.image_matching import ImageMatching
from src.deep_image_matching.io.export_to_colmap import ExportToColmap
from src.deep_image_matching.io.h5_to_db import import_into_colmap
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


confs_zoo = {
    "superpoint+lightglue": {"extractor": "superpoint", "matcher": "lightglue"},
    "disk+lightglue": {"extractor": "disk", "matcher": "lightglue"},
    # "aliked+lightglue": {"extractor": "aliked", "matcher": "lightglue"},
    # "sift+lightglue": {"extractor": "sift", "matcher": "lightglue"},
    "superpoint+superglue": {"extractor": "superpoint", "matcher": "superglue"},
    "keynetaffnethardnet+adalam": {
        "extractor": "keynetaffnethardnet",
        "matcher": "adalam",
    },
    "loftr": {"extractor": None, "matcher": "loftr"},
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
    parser.add_argument("-n", "--max_features", type=int, default=2048)
    parser.add_argument("--debug", action="store_true", default=False)

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
    args.outs.mkdir(parents=True, exist_ok=True)

    if args.config is None:
        raise ValueError("--config option is required")
    else:
        args.local_features = confs_zoo[args.config]["extractor"]
        args.matching = confs_zoo[args.config]["matcher"]

    if args.strategy is None:
        raise ValueError("--strategy option is required")
    if args.strategy not in ["bruteforce", "sequential", "retrieval", "custom_pairs"]:
        raise ValueError(
            f"Invalid strategy option: {args.strategy}. Valid options are: bruteforce, sequential, retrieval, custom_pairs"
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
        args.pairs = None

    if args.strategy == "sequential":
        if args.overlap is None:
            raise ValueError(
                "--overlap option is required when --strategy is set to sequential"
            )
    else:
        args.overlap = None

    if args.debug:
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
    # else:
    #     local_features = "detect_and_describe"
    #     custom_config["general"]["detector_and_descriptor"] = args.local_features

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
        custom_config=custom_config,
        # max_feat_numb=max_features,
        pair_file=pair_file,
        overlap=overlap,
    )
    pairs = img_matching.generate_pairs()
    feature_path = img_matching.extract_features()
    keypoints, correspondences = img_matching.match_pairs(feature_path)

    # Plot statistics
    images = img_matching.image_list
    logger.info("Finished matching and exporting")
    logger.info(f"\tProcessed images: {len(images)}")
    logger.info(f"\tProcessed pairs: {len(pairs)}")

    # Export in colmap format
    ExportToColmap(
        images,
        img_matching.width,
        img_matching.height,
        keypoints,
        correspondences,
        output_dir,
    )

    # Using also h5_to_db.py
    database_path = Path(output_dir) / "db2.db"
    if database_path.exists():
        database_path.unlink()
    import_into_colmap(
        imgs_dir,
        feature_dir=feature_path.parent,
        database_path=database_path,
        single_camera=True,
    )


if __name__ == "__main__":
    main()

    logger.info("Done")
