# 1. run matching with a feature-based approach (e.g., superpoint+lightglue)
# 2. run reconstruction with pycolmap and export the reconstruction as text file
# 3.

import shutil
import subprocess
from pathlib import Path


def empty_reconstruction_from_existing(
    reference_rec: Path, target_rec: Path, overwrite=False
):
    if not (reference_rec := Path(reference_rec)).exists():
        raise ValueError(f"Source file {reference_rec} does not exist.")

    files = ["cameras.txt", "images.txt", "points3D.txt"]
    for f in files:
        if not (reference_rec / f).exists():
            raise ValueError(
                f"Source file {reference_rec / f} does not exist in {reference_rec} directory."
            )

    if (target_rec := Path(target_rec)).exists() and not overwrite:
        raise ValueError(
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


# sfm_dir = Path("output/easy_small_superpoint+lightglue_sequential/reconstruction")
# dense_dir = Path("output/easy_small_superpoint+lightglue_sequential/roma_dense")
# roma_db = Path("output/easy_small_roma_bruteforce/database_pycolmap.db")


sfm_dir = Path("output/belvedere_stereo_superpoint+lightglue_bruteforce/reconstruction")
roma_db = Path("output/belvedere_stereo_roma_bruteforce/database_pycolmap.db")
dense_dir = Path("output/belvedere_stereo_roma_bruteforce/roma_dense")


# Make empty reconstruction for RoMa
empty_reconstruction_from_existing(sfm_dir, dense_dir, overwrite=True)

# Run dense reconstruction with RoMa using the camera poses from descriptor-based reconstruction
colmap_path = "colmap"
images_path = dense_dir.parent / "images"
database_path = dense_dir / "database.db"
shutil.copy(roma_db, database_path)

cmd = f"{colmap_path} point_triangulator --database_path {database_path} --image_path {images_path} --input_path {dense_dir} --output_path {dense_dir}"
cmd += " --Mapper.tri_ignore_two_view_tracks=0 --Mapper.filter_min_tri_angle=0.5"

out = subprocess.run(cmd, shell=True, capture_output=True)

if out.returncode != 0:
    print(out.stderr.decode("utf-8"))
    raise RuntimeError("RoMa triangulation failed.")
else:
    print(out.stdout.decode("utf-8"))

# import pycolmap
# from deep_image_matching.triangulation import OutputCapture

# verbose = True

# with OutputCapture(verbose):
#     with pycolmap.ostream():
#         reconstruction = pycolmap.triangulate_points(
#             reference_model, database_path, image_dir, model_path, options=options
#         )
