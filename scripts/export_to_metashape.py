import argparse
from pathlib import Path

from deep_image_matching import logger

try:
    import Metashape
except ImportError:
    logger.error(
        "Metashape module is not available. Please, install it first by following the instructions at https://github.com/franioli/metashape."
    )
    exit()

from scripts.metashape.metashape_from_dim import project_from_bundler

prm_to_optimize = {
    "f": True,
    "cx": True,
    "cy": True,
    "k1": True,
    "k2": True,
    "k3": True,
    "k4": False,
    "p1": True,
    "p2": True,
    "b1": False,
    "b2": False,
    "tiepoint_covariance": True,
}


def main(args):
    images_dir = Path(args.image_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist.")
    bundler_file_path = Path(args.bundler_file)
    if not bundler_file_path.exists():
        raise FileNotFoundError(f"Bundler file {bundler_file_path} does not exist.")
    bundler_im_list = Path(args.bundler_im_list)
    if not bundler_im_list.exists():
        raise FileNotFoundError(f"Bundler image list {bundler_im_list} does not exist.")

    project_path = Path(args.project_path)
    project_dir = project_path.parent

    if not project_dir.exists():
        project_dir.mkdir(parents=True)

    if project_path.suffix != ".psx":
        project_path = project_path.name + ".psx"

    project_from_bundler(
        project_path=project_path,
        images_dir=images_dir,
        bundler_file_path=bundler_file_path.resolve(),
        bundler_im_list=bundler_im_list.resolve(),
        prm_to_optimize=prm_to_optimize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge all COLMAP databases of the input folder."
    )

    parser.add_argument(
        "--project_path",
        type=str,
        help="Path to the Metashape project file (.psx).",
        required=True,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the images directory.",
        required=True,
    )
    parser.add_argument(
        "bundler_file",
        type=str,
        help="Path to the bundler file.",
        required=True,
    )
    parser.add_argument(
        "bundler_im_list",
        type=str,
        help="Path to the bundler image list. If None is passed, it is assumed to be a file named 'bundler_list.txt' in the same directory of the bundler file.",
        default=None,
    )

    args = parser.parse_args()

    if args.bundler_im_list is None:
        args.bundler_im_list = args.bundler_file.parent / "bundler_list.txt"

    main(args)
