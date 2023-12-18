# 1. run matching with a feature-based approach (e.g., superpoint+lightglue)
# 2. run reconstruction with pycolmap and export the reconstruction as text file
# 3. run dense matching with a detector_free approach using the camera poses from the feature-based reconstruction (i.e., triangulate points with COLMAP trinaulator API)


import shutil
import subprocess
from pathlib import Path

colmap_path = "colmap"
sfm_dir = Path("output/belvedere_stereo_superpoint+lightglue_bruteforce/reconstruction")
dense_dir = Path("output/belvedere_stereo_roma_bruteforce/roma_dense")
dense_feats_db = dense_dir / "database_pycolmap.db"


def empty_reconstruction_from_existing(
    reference_rec: Path, target_rec: Path, overwrite=False
):
    """
    empty_reconstruction_from_existing _summary_

    Args:
        reference_rec (Path): _description_
        target_rec (Path): _description_
        overwrite (bool, optional): _description_. Defaults to False.

    """
    if not (reference_rec := Path(reference_rec)).exists():
        raise FileNotFoundError(f"Source file {reference_rec} does not exist.")

    files = ["cameras.txt", "images.txt", "points3D.txt"]
    for f in files:
        if not (reference_rec / f).exists():
            raise FileNotFoundError(
                f"Source file {reference_rec / f} does not exist in {reference_rec} directory."
            )

    if (target_rec := Path(target_rec)).exists() and not overwrite:
        raise OSError(
            f"Target directory {target_rec} already exists and overwrite is set to False."
        )
    elif target_rec.exists() and overwrite:
        shutil.rmtree(target_rec)
    target_rec.mkdir(parents=True)

    # Make empty images.txt
    with open(reference_rec / "images.txt", "r") as inp, open(
        target_rec / "images.txt", "w"
    ) as out:
        lines = inp.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                out.write(line)
                out.write("\n")

    # Copy cameras.txt
    shutil.copy(reference_rec / "cameras.txt", target_rec / "cameras.txt")

    # Make empty points3D.txt
    with open(reference_rec / "points3D.txt", "r") as inp, open(
        target_rec / "points3D.txt", "w"
    ) as out:
        pass


def run_dense():
    """
    run_dense _summary_

    Raises:
        RuntimeError: _description_
    """
    # Make empty reconstruction for dense matching
    empty_reconstruction_from_existing(sfm_dir, dense_dir, overwrite=True)

    # Run dense reconstruction with detector_free matchers the camera poses from descriptor-based reconstruction
    images_path = dense_dir.parent / "images"
    database_path = dense_dir / "database.db"
    shutil.copy(dense_feats_db, database_path)

    cmd = f"{colmap_path} point_triangulator --database_path {database_path} --image_path {images_path} --input_path {dense_dir} --output_path {dense_dir}"
    cmd += " --Mapper.tri_ignore_two_view_tracks=0 --Mapper.filter_min_tri_angle=0.5"

    out = subprocess.run(cmd, shell=True, capture_output=True)

    if out.returncode != 0:
        print(out.stderr.decode("utf-8"))
        raise RuntimeError("Triangulation failed.")
    else:
        print(out.stdout.decode("utf-8"))


if __name__ == "__main__":
    run_dense()
