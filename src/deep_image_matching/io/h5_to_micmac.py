from pathlib import Path

import h5py
import numpy as np


def export_features(
    feature_path: Path,
    match_path: Path,
    img_dir: Path,
    out_dir: Path,
):
    features = h5py.File(str(feature_path), "r")
    matches = h5py.File(str(match_path), "r")


if __name__ == "__main__":

    project_path = Path("datasets/cyprus")
    res = "results_superpoint+lightglue_matching_lowres_quality_high"
    out_dir = project_path / "micmac"

    feature_path = project_path / res / "features.h5"
    match_path = project_path / res / "matches.h5"

    out_feats_dir = out_dir / "Homol"
    out_feats_dir.mkdir(exist_ok=True, parents=True)

    features = h5py.File(str(feature_path), "r")
    matches = h5py.File(str(match_path), "r")

    for i0 in matches.keys():
        i0_dir = out_feats_dir / f"Pastis{i0}"
        i0_dir.mkdir(exist_ok=True, parents=True)
        for i1 in matches[i0].keys():
            with open(i0_dir / (i1 + ".txt"), "w") as f:
                # Get index of matches in both images
                matches0_idx = np.asarray(matches[i0][i1])[:, 0]
                matches1_idx = np.asarray(matches[i0][i1])[:, 1]

                # get index of sorted matches
                s0idx = np.argsort(matches0_idx)
                s1idx = np.argsort(matches1_idx)

                # Get coordinates of matches
                x0y0 = features[i0]["keypoints"][matches0_idx[s0idx]]
                x1y1 = features[i1]["keypoints"][matches1_idx[s1idx]]

                # Restore the original order
                x0y0 = x0y0[np.argsort(s0idx)]
                x1y1 = x1y1[np.argsort(s1idx)]

                for x0, y0, x1, y1 in zip(
                    x0y0[:, 0], x0y0[:, 1], x1y1[:, 0], x1y1[:, 1]
                ):
                    f.write(f"{x0:6f} {y0:6f} {x1:6f} {y1:6f} 1.000000\n")

    print("Done!")
