# 1. run matching with a feature-based approach (e.g., superpoint+lightglue)
# 2. run reconstruction with pycolmap and export the reconstruction as text file
# 3. run dense matching with a detector_free approach using the camera poses from the feature-based reconstruction (i.e., triangulate points with COLMAP trinaulator API)

import argparse
import shutil
import subprocess
from pathlib import Path

# sfm_dir = Path(
#     "output/example_easy_lowres_superpoint+lightglue_bruteforce/reconstruction"
# )
# dense_dir = Path("output/belvedere_stereo_roma_bruteforce/roma_dense")
# dense_feats_db = dense_dir / "database_pycolmap.db"


def empty_reconstruction_from_existing(sfm_dir: Path, dense_dir: Path, overwrite=False):
    """
    Create an empty COLMAP reconstruction from an existing one. This can be used to build a dense reconstruction starting from known camera poses.

    Args:
        reference_rec (Path): Path to the source reconstruction directory.
        target_rec (Path): Path to the target reconstruction directory.
        overwrite (bool, optional): If True, overwrite the target directory if it exists. Defaults to False.

    Raises:
        FileNotFoundError: If the source file or required files within it do not exist.
        OSError: If the target directory already exists and overwrite is set to False.
    """
    if not (sfm_dir := Path(sfm_dir)).exists():
        raise FileNotFoundError(f"Source file {sfm_dir} does not exist.")

    files = ["cameras.txt", "images.txt", "points3D.txt"]
    for f in files:
        if not (sfm_dir / f).exists():
            raise FileNotFoundError(
                f"Source file {sfm_dir / f} does not exist in {sfm_dir} directory."
            )

    if (dense_dir := Path(dense_dir)).exists() and not overwrite:
        raise OSError(
            f"Target directory {dense_dir} already exists and overwrite is set to False."
        )
    elif dense_dir.exists() and overwrite:
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True)

    # Make empty images.txt
    with open(sfm_dir / "images.txt", "r") as inp, open(
        dense_dir / "images.txt", "w"
    ) as out:
        lines = inp.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                out.write(line)
                out.write("\n")

    # Copy cameras.txt
    shutil.copy(sfm_dir / "cameras.txt", dense_dir / "cameras.txt")

    # Make empty points3D.txt
    with open(sfm_dir / "points3D.txt", "r") as inp, open(
        dense_dir / "points3D.txt", "w"
    ) as out:
        pass


def run_dense(
    dense_dir: Path,
    dense_database: str = "database.db",
    colmap_bin: str = "colmap",
    use_two_view_tracks: bool = True,
    options={},
):
    """
    Run dense reconstruction using COLMAP's point_triangulator.

    Raises:
        RuntimeError: If the triangulation process fails.
    """
    # Run dense reconstruction with detector_free matchers the camera poses from descriptor-based reconstruction
    images_path = dense_dir.parents[1] / "images"
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory {images_path} does not exist.")
    dense_database = dense_dir / dense_database
    if not dense_database.exists():
        raise FileNotFoundError(f"Dense database {dense_database} does not exist.")

    cmd = f"{colmap_bin} point_triangulator --database_path {dense_database} --image_path {images_path} --input_path {dense_dir} --output_path {dense_dir}"
    if use_two_view_tracks:
        cmd += " --Mapper.tri_ignore_two_view_tracks=0"

    for k, v in options.items():
        cmd += f" --{k}={v}"

    out = subprocess.run(cmd, shell=True, capture_output=True)

    if out.returncode != 0:
        print(out.stderr.decode("utf-8"))
        raise RuntimeError("Triangulation failed.")
    else:
        print(out.stdout.decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dense reconstruction using COLMAP."
    )
    parser.add_argument(
        "--sfm_dir",
        dest="sfm_dir",
        required=True,
        help="Path to the source feature-based reconstruction directory.",
    )
    parser.add_argument(
        "--dense_dir",
        dest="dense_dir",
        required=True,
        help="Path to the target dense reconstruction directory.",
    )
    parser.add_argument(
        "--database",
        dest="database",
        default="database.db",
        help="Name of the COLMAP database with respect to the 'dense_dir' containing features for dense matching.",
    )
    parser.add_argument(
        "--use-two-view-tracks",
        dest="use_two_view_tracks",
        action="store_true",
        default=True,
        help="Use two-view tracks for triangulation.",
    )
    parser.add_argument(
        "--colmap_bin",
        dest="colmap_path",
        default="colmap",
        help="Path to the COLMAP executable. Defaults to 'colmap' (works under linux if COLMAP is installed gloablly, specify your executable path otherwise').",
    )
    args = parser.parse_args()

    sfm_dir = Path(args.sfm_dir)
    dense_matching_dir = Path(args.dense_dir)
    colmap_path = args.colmap_path
    dense_database = args.database
    use_two_view_tracks = args.use_two_view_tracks

    options = {"Mapper.filter_min_tri_angle": 0.5}

    # Create empty reconstruction for dense matching
    dense_rec_dir = sfm_dir / "dense"

    # Make empty reconstruction for dense matching
    empty_reconstruction_from_existing(
        sfm_dir=sfm_dir / "reconstruction", dense_dir=dense_rec_dir, overwrite=True
    )
    # Copy database with dense features to the empty reconstruction
    shutil.copy(dense_matching_dir / dense_database, dense_rec_dir / dense_database)

    # Run dense matching
    run_dense(
        dense_dir=dense_rec_dir,
        dense_database=dense_database,
        colmap_bin=colmap_path,
        use_two_view_tracks=use_two_view_tracks,
        options=options,
    )
