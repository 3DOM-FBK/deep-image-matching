# 1. run matching with a feature-based approach (e.g., superpoint+lightglue)
# 2. run reconstruction with pycolmap and export the reconstruction as text file
# 3.

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

out_dir = Path("output/easy_small_superpoint+lightglue_sequential/cleaned")

# Make empty images.txt
make_empty_image_file(sfm_dir / "images.txt", out_dir / "images.txt")

#
