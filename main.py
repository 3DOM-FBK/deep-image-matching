import argparse
import shutil
from pathlib import Path

from config import conf_general, confs, opt_zoo
from src.deep_image_matching import change_logger_level, logger, timer
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
        help="Extractor and matcher configuration",
        choices=confs.keys(),
        default="superpoint+lightglue",
    )
    parser.add_argument(
        "-m",
        "--strategy",
        choices=[
            "bruteforce",
            "sequential",
            "retrieval",
            "custom_pairs",
            "matching_lowres",
        ],
        default="sequential",
        help="Matching strategy",
    )
    parser.add_argument(
        "-p", "--pairs", type=str, default=None, help="Specify pairs for matching"
    )
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
        choices=opt_zoo["retrieval"],
        default=None,
        help="Specify image retrieval method",
    )
    parser.add_argument(
        "--upright",
        action="store_true",
        help="Enable the estimation of the best image rotation for the matching (useful in case of aerial datasets).",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of output folder",
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.gui is True:
        gui_out = gui()
        args.images = gui_out["image_dir"]
        args.outs = gui_out["out_dir"]
        args.config = gui_out["config"]
        args.strategy = gui_out["strategy"]
        args.pairs = gui_out["pair_file"]
        args.overlap = gui_out["image_overlap"]
        args.upright = gui_out["upright"]
        args.force = True

    return args


def initialization():
    """Do checks on the input arguments and return the configuration dictionary with the following keys: general, extractor, matcher"""
    args = parse_args()

    # Input folder
    if args.images is None:
        raise ValueError("--images option is required")
    else:
        args.images = Path(args.images)
        if not args.images.exists() or not args.images.is_dir():
            raise ValueError(f"Folder {args.images} does not exist")

    # Output folder
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

    # Check extraction and matching configuration
    if args.config is None or args.config not in confs:
        raise ValueError(
            "--config option is required and must be a valid configuration (check config.py))"
        )
    extractor = confs[args.config]["extractor"]["name"]
    if extractor not in opt_zoo["extractors"]:
        raise ValueError(
            f"Invalid extractor option: {extractor}. Valid options are: {opt_zoo['extractors']}"
        )
    matcher = confs[args.config]["matcher"]["name"]
    if matcher not in opt_zoo["matchers"]:
        raise ValueError(
            f"Invalid matcher option: {matcher}. Valid options are: {opt_zoo['matchers']}"
        )

    # Matching strategy
    if args.strategy is None:
        raise ValueError("--strategy option is required")
    if args.strategy not in opt_zoo["matching_strategy"]:
        raise ValueError(
            f"Invalid strategy option: {args.strategy}. Valid options are: {opt_zoo['matching_strategy']}"
        )
    if args.strategy == "retrieval":
        if args.retrieval is None:
            raise ValueError(
                "--retrieval option is required when --strategy is set to retrieval"
            )
        elif args.retrieval not in opt_zoo["retrieval"]:
            raise ValueError(
                f"Invalid retrieval option: {args.retrieval}. Valid options are: {opt_zoo['retrieval']}"
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
        change_logger_level(logger.name, "debug")

    # Build configuration dictionary
    conf_general["image_dir"] = args.images
    conf_general["output_dir"] = args.outs
    conf_general["matching_strategy"] = args.strategy
    conf_general["retrieval"] = args.retrieval
    conf_general["pair_file"] = args.pairs
    conf_general["overlap"] = args.overlap
    conf_general["upright"] = args.upright
    conf_general["verbose"] = args.verbose
    cfg = {
        "general": conf_general,
        "extractor": confs[args.config]["extractor"],
        "matcher": confs[args.config]["matcher"],
    }

    return cfg


def main():
    # Parse arguments
    config = initialization()
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

    # Try to run reconstruction with pycolmap
    try:
        import pycolmap
    except ImportError:
        logger.error("Pycomlap is not available, skipping reconstruction")
        use_pycolmap = False

    if use_pycolmap:
        from deep_image_matching import reconstruction

        # Define database path and camera mode
        database = output_dir / "database_pycolmap.db"
        camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO

        # Define cameras
        # cam1 = pycolmap.Camera(
        #     model="FULL_OPENCV",
        #     width=6012,
        #     height=4008,
        #     params=[
        #         9.26789262766209504e03,
        #         9.26789262766209504e03,
        #         3.05349107994520591e03,
        #         1.94835654532114540e03,
        #         -8.07042713029020586e-02,
        #         9.46617629940955385e-02,
        #         3.31782983128223608e-04,
        #         -4.32106111976037410e-04,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #     ],
        # )
        # cam2 = pycolmap.Camera(
        #     model="FULL_OPENCV",
        #     width=6012,
        #     height=4008,
        #     params=[
        #         6.62174345720628298e03,
        #         6.62174345720628298e03,
        #         3.01324420057086490e03,
        #         1.94347461466223308e03,
        #         -9.41830394356213407e-02,
        #         8.55303528514532035e-02,
        #         1.68948638308769863e-04,
        #         -8.74637609310216697e-04,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #     ],
        # )
        # cameras = [cam1, cam2]
        cameras = None

        # Define options
        # options = (
        #     {
        #         "ba_refine_focal_length": False,
        #         "ba_refine_principal_point": False,
        #         "ba_refine_extra_params": False,
        #     },
        # )
        options = {}

        # Run reconstruction
        model = reconstruction.main(
            database=database,
            image_dir=imgs_dir,
            feature_path=feature_path,
            match_path=match_path,
            pair_path=pair_path,
            output_dir=output_dir,
            camera_mode=camera_mode,
            cameras=cameras,
            skip_geometric_verification=True,
            options=options,
            verbose=False,  # config["general"]["verbose"],
        )
        timer.update("pycolmap reconstruction")

    # Export in Bundler format for Metashape using colmap CLI
    # if not use_pycolmap:

    #     def export_to_bundler(
    #         database: Path, image_dir: Path, output_dir: Path, out_name: str = "bundler"
    #     ) -> bool:
    #         import subprocess
    #         from pprint import pprint

    #         colamp_path = "colmap"

    #         cmd = [
    #             colamp_path,
    #             "model_converter",
    #             "--input_path",
    #             str(database.parent.resolve()),
    #             "--output_path",
    #             str(database.parent.resolve() / out_name),
    #             "--output_type",
    #             "Bundler",
    #         ]
    #         ret = subprocess.run(cmd, capture_output=True)
    #         if ret.returncode != 0:
    #             logger.error("Unable to export to Bundler format")
    #             pprint(ret.stdout.decode("utf-8"))
    #             return False

    #         shutil.copytree(image_dir, output_dir / "images", dirs_exist_ok=True)
    #         logger.info("Export to Bundler format completed successfully")

    #         return True

    #     out_name = "bundler"
    #     export_to_bundler(database, imgs_dir, output_dir, out_name)

    timer.print("Deep Image Matching")


if __name__ == "__main__":
    main()
