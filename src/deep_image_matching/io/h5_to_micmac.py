from itertools import permutations
from pathlib import Path
from typing import Tuple

import cv2
import h5py
import numpy as np
from deep_image_matching.visualization import viz_matches_cv2


def get_matches(matches, features, key0, key1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the matches between two images based on the given keys.

    Args:
        matches (h5py.File): The HDF5 file containing the matches.
        features (h5py.File): The HDF5 file containing the features.
        key0 (str): Name of the first image.
        key1 (str): Name of the second image.

    Returns:
        tuple: A tuple containing the coordinates of the matches in both images.
               The first element corresponds to the coordinates in the first image, and the second element corresponds to the coordinates in the second image.
               If the matches are not present, None is returned for both elements.
    """

    with h5py.File(str(feature_path), "r") as features, h5py.File(
        str(match_path), "r"
    ) as matches:

        # Check if the matches are present
        if key0 not in matches.keys() or key1 not in matches[key0].keys():
            return None, None

        # Get index of matches in both images
        matches0_idx = np.asarray(matches[key0][key1])[:, 0]
        matches1_idx = np.asarray(matches[key0][key1])[:, 1]

        # get index of sorted matches
        s0idx = np.argsort(matches0_idx)
        s1idx = np.argsort(matches1_idx)

        # Get coordinates of matches
        x0y0 = features[key0]["keypoints"][matches0_idx[s0idx]]
        x1y1 = features[key1]["keypoints"][matches1_idx[s1idx]]

        # Restore the original order
        x0y0 = x0y0[np.argsort(s0idx)]
        x1y1 = x1y1[np.argsort(s1idx)]

    return x0y0, x1y1


def write_matches(file, x0y0, x1y1):
    with open(file, "w") as f:
        for x0, y0, x1, y1 in zip(x0y0[:, 0], x0y0[:, 1], x1y1[:, 0], x1y1[:, 1]):
            f.write(f"{x0:6f} {y0:6f} {x1:6f} {y1:6f} 1.000000\n")


def export_tie_points(
    feature_path: Path,
    match_path: Path,
    out_dir: Path,
) -> None:
    """
    Export tie points from h5 databases containing the features on each images and the index of the matched features to text files in MicMac format.

    Args:
        feature_path (Path): Path to the features.h5 file.
        match_path (Path): Path to the matches.h5 file.
        out_dir (Path): Path to the output directory.

    Raises:
        FileNotFoundError: If the feature file or match file does not exist.

    Returns:
        None
    """

    feature_path = Path(feature_path)
    match_path = Path(match_path)
    out_dir = Path(out_dir)

    if not feature_path.exists():
        raise FileNotFoundError(f"File {feature_path} does not exist")
    if not match_path.exists():
        raise FileNotFoundError(f"File {match_path} does not exist")
    out_dir.mkdir(exist_ok=True, parents=True)

    with h5py.File(feature_path, "r") as features:
        keys = permutations(features.keys(), 2)

    with h5py.File(match_path, "r") as matches:
        for i0, i1 in keys:
            i0_dir = out_dir / f"Pastis{i0}"
            i0_dir.mkdir(exist_ok=True, parents=True)

            # Define the file to write the matches
            file = i0_dir / (i1 + ".txt")

            # Get the matches between the two images
            if i0 in matches.keys() and i1 in matches[i0].keys():
                x0y0, x1y1 = get_matches(features, matches, i0, i1)
            else:
                x1y1, x0y0 = get_matches(features, matches, i1, i0)

            if x0y0 is None or x1y1 is None:
                continue

            # threading.Thread(target=lambda: write_matches(file, x0y0, x1y1)).start()
            write_matches(file, x0y0, x1y1)


def show_micmac_matches(file: Path, image_dir: Path, out: Path = None) -> np.ndarray:
    """
    Display the tie points between two images matched by a MicMac from the matches text file.

    Args:
        file (Path): The path to the file containing the matches.
        image_dir (Path): The directory containing the images.
        out (Path, optional): The path to save the output image. Defaults to None.

    Returns:
        np.ndarray: The output image with the matches visualized.
    """

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist")

    # Get the image names
    i0 = file.parent.name.replace("Pastis", "")
    i1 = file.name.replace(".txt", "")

    # Read the matches
    data = np.loadtxt(file, dtype=np.float32)
    pts0 = data[:, :2]
    pts1 = data[:, 2:4]

    # Read the images
    image0 = cv2.imread(str(image_dir / i0))
    image1 = cv2.imread(str(image_dir / i1))

    out = viz_matches_cv2(image0, image1, pts0, pts1, out)

    return out


if __name__ == "__main__":

    project_path = Path("datasets/cyprus_micmac2")

    feature_path = project_path / "features.h5"
    match_path = project_path / "matches.h5"

    out_feats_dir = project_path / "Homol"
    export_tie_points(feature_path, match_path, out_feats_dir)

    # make match figures
    # match_figure_dir = project_path / "matches"
    # if match_figure_dir is not None:
    #     match_figure_dir = Path(match_figure_dir)
    #     match_figure_dir.mkdir(exist_ok=True, parents=True)
    #     file = project_path / "Homol" / f"Pastis{i0}" / f"{i1}.txt"
    #     matches_fig = match_figure_dir / f"{Path(i0).stem}_{Path(i1).stem}.png"
    #     threading.Thread(
    #         target=lambda: show_micmac_matches(file, project_path, matches_fig)
    #     ).start()

    print("Done!")
