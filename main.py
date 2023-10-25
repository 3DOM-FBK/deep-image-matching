import argparse
import shutil
from pathlib import Path

from easydict import EasyDict as edict

from config import custom_config
from src.deep_image_matching.gui import gui
from src.deep_image_matching.image_matching import ImageMatching
from src.deep_image_matching.io.export_to_colmap import ExportToColmap
from src.deep_image_matching.utils import setup_logger

logger = setup_logger(console_log_level="debug")


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
        choices=[
            "superglue",
            "lightglue",
            "loftr",
            "ALIKE",
            "ORB",
            "DISK",
            "SuperPoint",
            "KeyNetAffNetHardNet",
        ],
    )
    parser.add_argument("-n", "--max_features", type=int, required=True)

    args = parser.parse_args()

    return args


def main(debug: bool = False):
    if debug:
        args = edict(
            {
                "interface": "cli",
                "images": "data/easy_small",
                "outs": "res",
                "strategy": "sequential",
                "features": "lightglue",
                "retrieval": "netvlad",
                "overlap": 2,
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

        if not imgs_dir.exists() or not imgs_dir.is_dir():
            raise ValueError(f"Folder {imgs_dir} does not exist")

        if args.features in ["superglue", "lightglue", "loftr"]:
            local_features = args.features
        else:
            local_features = "detect_and_describe"
            custom_config["general"]["detector_and_descriptor"] = args.features

    elif args.interface == "gui":
        (
            matching_strategy,
            imgs_dir,
            output_dir,
            pair_file,
            overlap,
            feat,
            max_features,
        ) = gui()
        retrieval_option = None
        if feat in ["superglue", "lightglue", "loftr"]:
            local_features = feat
        else:
            local_features = "detect_and_describe"
            custom_config["general"]["detector_and_descriptor"] = feat

    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pairs and matching
    img_matching = ImageMatching(
        imgs_dir=imgs_dir,
        matching_strategy=matching_strategy,
        retrieval_option=retrieval_option,
        local_features=local_features,
        custom_config=custom_config,
        max_feat_numb=max_features,
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

    # Tests for using h5_to_db.py
    from deep_image_matching.io.h5_to_db import (
        COLMAPDatabase,
        add_keypoints,
        add_matches,
    )

    def import_into_colmap(
        img_dir, feature_dir=".featureout", database_path="colmap.db", img_ext=".jpg"
    ):
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()
        single_camera = False
        fname_to_id = add_keypoints(db, feature_dir, img_dir)
        add_matches(
            db,
            feature_dir,
            fname_to_id,
        )

        db.commit()
        return

    database_path = "res/colmap2.db"
    if Path(database_path).exists():
        Path(database_path).unlink()
    # import_into_colmap(imgs_dir, feature_dir="res", database_path=database_path)


if __name__ == "__main__":
    main(debug=True)

    logger.info("Done")
