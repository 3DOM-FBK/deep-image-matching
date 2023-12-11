# 1. run matching with a feature-based approach (e.g., superpoint+lightglue)
# 2. run reconstruction with pycolmap and export the reconstruction as text file
# 3.

import subprocess
from pathlib import Path


def make_empty_image_file(source: Path, target: Path):
    if not source.exists():
        raise ValueError(f"Source file {source} does not exist.")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "r") as inp, open(target, "w") as out:
        lines = inp.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                out.write(line)
                out.write("\n")


sfm_dir = Path("output/easy_small_superpoint+lightglue_sequential/reconstruction")

dense_dir = Path("output/easy_small_superpoint+lightglue_sequential/roma_dense")

# Make empty images.txt
make_empty_image_file(sfm_dir / "images.txt", dense_dir / "images.txt")

# Run dense reconstruction with RoMa using the camera poses from descriptor-based reconstruction
colmap_path = "colmap"
database_path = dense_dir / "database.db"
images_path = dense_dir.parent / "images"

cmd = f"{colmap_path} point_triangulator --database_path {database_path} --image_path {images_path}"
out = subprocess.run(cmd, shell=True, check=True, capture_output=True)
