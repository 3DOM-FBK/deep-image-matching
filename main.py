import shutil
import argparse
from easydict import EasyDict as edict
from pathlib import Path

from src.deep_image_matching.image_matching import ImageMatching
from src.deep_image_matching.io.export_to_colmap import ExportToColmap
from src.deep_image_matching.utils import setup_logger
from src.deep_image_matching.gui import gui

from config import custom_config

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matching with hand-crafted and deep-learning based local features and image retrieval."
    )

    parser.add_argument(
        "interface", type=str, help="Run command line interface", choices=["cli", "gui"]
    )

    parser.add_argument(
        "-i", "--images", type=str, help="Input image folder", required=True
    )
    parser.add_argument("-o", "--outs", type=str, help="Output folder", required=True)
    parser.add_argument(
        "-m",
        "--strategy",
        choices=["bruteforce", "sequential", "retrieval", "custom_pairs"],
        required=True,
    )
    parser.add_argument("-p", "--pairs", type=str)
    parser.add_argument(
        "-v",
        "--overlap",
        type=int,
        help="Image overlap if using sequential overlap strategy",
    )
    parser.add_argument(
        "-r", "--retrieval", choices=["netvlad", "openibl", "cosplace", "dir"]
    )
    parser.add_argument(
        "-f",
        "--features",
        choices=["superglue", "lightglue", "loftr", "ALIKE", "ORB", "DISK", "SuperPoint", "KeyNetAffNetHardNet"],
    )
    parser.add_argument("-n", "--max_features", type=int, required=True)

    args = parser.parse_args()

    return args


def main(debug: bool = False):
    if debug:
        args = edict(
            {
                "images": "data",
                "outs": "res",
                "strategy": "sequential",
                "features": "lightglue",
                "retrieval": "netvlad",
                "overlap": 1,
                "max_features": 1000,
            }
        )
    else:
        args = parse_args()


    if args.interface == "cli":
        if args.strategy == "retrieval" and args.retrieval is None:
            raise ValueError(
                "--retrieval option is required when --strategy is set to retrieval"
            )
        elif args.strategy == "retrieval":
            retrieval_option = args.retrieval
        else:
            retrieval_option = None

        if args.strategy == "custom_pairs" and args.pairs is None:
            raise ValueError(
                "--pairs option is required when --strategy is set to custom_pairs"
            )
        elif args.strategy == "custom_pairs":
            pair_file = Path(args.pairs)
        else:
            pair_file = None

        if args.strategy == "sequential" and args.overlap is None:
            raise ValueError(
                "--overlap option is required when --strategy is set to sequential"
            )
        elif args.strategy == "sequential":
            overlap = args.overlap
        else:
            overlap = None

        imgs_dir = Path(args.images)
        output_dir = Path(args.outs)
        matching_strategy = args.strategy
        max_features = args.max_features

        if args.features in [ "superglue", "lightglue", "loftr"]:
            local_features = args.features
        else:
            local_features = "detect_and_describe"
            custom_config["general"]["detector_and_descriptor"] = args.features
        
    elif args.interface == "gui":
        matching_strategy, imgs_dir, output_dir, pair_file, overlap, feat, max_features = gui()
        retrieval_option = None
        if feat in [ "superglue", "lightglue", "loftr"]:
            local_features = feat
        else:
            local_features = "detect_and_describe"
            custom_config["general"]["detector_and_descriptor"] = feat
   

    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pairs and matching
    img_matching = ImageMatching(
        imgs_dir,
        matching_strategy,
        pair_file,
        retrieval_option,
        overlap,
        local_features,
        custom_config,
        max_features,
    )

    images = img_matching.img_names()
    pairs = img_matching.generate_pairs()
    keypoints, correspondences = img_matching.match_pairs()

    # Plot statistics
    logger.info("Finished matching and exporting")
    logger.info(f"\tProcessed images: {len(images)}")
    logger.info(f"\tProcessed pairs: {len(pairs)}")

    # Export in colmap format
    ExportToColmap(
        images,
        img_matching.img_format,
        img_matching.width,
        img_matching.height,
        keypoints,
        correspondences,
        output_dir,
        )


if __name__ == "__main__":
    main(debug=False)

    logger.info("Done")
