from pathlib import Path

import pycolmap

rec_dir = Path(
    "/home/francesco/phd/deep-image-matching/datasets/belv_20230725/results_superpoint+lightglue_bruteforce_quality_high/reconstruction"
)
rec = pycolmap.Reconstruction(rec_dir)
rec.export_bundler(
    rec_dir / "bunlder.out",
    rec_dir / "bundler_list.txt",
    skip_distortion=True,
)
