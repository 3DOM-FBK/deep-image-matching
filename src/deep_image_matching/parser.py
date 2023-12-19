import argparse
import shutil
from pathlib import Path

from . import change_logger_level, logger
from .config import conf_general, confs, opt_zoo
from .gui import gui


def parse_cli():
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
        "-Q",
        "--quality",
        type=str,
        choices=["lowest", "low", "medium", "high", "highest"],
        default="high",
        help="Set the image resolution for the matching. High means full resolution images, medium is half res, low is 1/4 res, highest is x2 upsampling. Default is high.",
    ),
    parser.add_argument(
        "-t",
        "--tiling",
        type=str,
        choices=["none", "preselection", "grid", "exhaustive"],
        default="none",
        help="Set the tiling strategy for the matching. Default is none.",
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
        "--skip_reconstruction",
        action="store_true",
        help="Skip reconstruction step carried out with pycolmap. This step is necessary to export the solution in Bundler format for Agisoft Metashape.",
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


def parse_config():
    """Do checks on the input arguments and return the configuration dictionary with the following keys: general, extractor, matcher"""
    args = parse_cli()

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
    cfg_general_user = {
        "image_dir": args.images,
        "output_dir": args.outs,
        "quality": args.quality,
        "tile_selection": args.tiling,
        "matching_strategy": args.strategy,
        "retrieval": args.retrieval,
        "pair_file": args.pairs,
        "overlap": args.overlap,
        "upright": args.upright,
        "verbose": args.verbose,
        "skip_reconstruction": args.skip_reconstruction,
    }
    cfg = {
        "general": {**conf_general, **cfg_general_user},
        "extractor": confs[args.config]["extractor"],
        "matcher": confs[args.config]["matcher"],
    }

    return cfg
