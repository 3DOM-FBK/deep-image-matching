import shutil
import subprocess
from itertools import combinations, permutations
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import cv2
import h5py
import numpy as np
from deep_image_matching import IMAGE_EXT, logger
from deep_image_matching.visualization import viz_matches_cv2


def execute(cmd, cwd=None):
    if cwd is not None:
        cwd = Path(cwd).resolve()
        if not cwd.exists():
            raise FileNotFoundError(f"Directory {cwd} does not exist")
    print(" ".join(cmd))
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


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
                # If no matches are found, write a file with a single fake match with zeros image coordinates
                # NOTE: This is a workaround to avoid MicMac from crashing when no matches are found, these matches should be discarded as outliers during the bundle adjustment
                # x0y0 = np.zeros((1, 2))
                # x1y1 = np.zeros((1, 2))

            # threading.Thread(target=lambda: write_matches(file, x0y0, x1y1)).start()
            write_matches(file, x0y0, x1y1)

    logger.info(f"Exported tie points to {out_dir}")


def export_to_micmac(
    image_dir: Path,
    features_h5: Path,
    matches_h5: Path,
    out_dir: Path = "micmac",
    img_ext: str = IMAGE_EXT,
    run_Tapas: bool = False,
    micmac_path: Path = None,
):
    image_dir = Path(image_dir)
    feature_path = Path(features_h5)
    match_path = Path(matches_h5)
    out_dir = Path(out_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file {feature_path} does not exist")
    if not match_path.exists():
        raise FileNotFoundError(f"Matches file {match_path} does not exist")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Export the tie points
    homol_dir = out_dir / "Homol"
    export_tie_points(feature_path, match_path, homol_dir)

    # Check if some images have no matches and remove them
    images = sorted([e for e in image_dir.glob("*") if e.suffix in img_ext])
    for img in images:
        mtch_dir = homol_dir / f"Pastis{img.name}"
        if list(mtch_dir.glob("*.txt")):
            shutil.copyfile(img, out_dir / img.name)
        else:
            logger.info(f"No matches found for image {img.name}, removing it")
            shutil.rmtree(mtch_dir)

    # Check that the number of images is consistent with the number of matches
    images = sorted([e for e in out_dir.glob("*") if e.suffix in img_ext])
    matches = sorted([e for e in homol_dir.glob("*") if e.is_dir()])
    if len(images) != len(matches):
        raise Exception(
            f"The number of images ({len(images)}) is different from the number of matches ({len(matches)})"
        )

    logger.info(
        f"Succesfully exported images and tie points ready for MICMAC processing to {out_dir}"
    )

    if run_Tapas:
        # Try to run MicMac
        logger.info("Try to run relative orientation with MicMac...")
        try:
            # Try to find the MicMac executable
            if micmac_path is None:
                logger.info("MicMac path not specified, trying to find it...")
                micmac_path = shutil.which("mm3d")
                if not micmac_path:
                    raise FileNotFoundError("MicMac path not found")
                logger.info(f"Found MicMac executable at {micmac_path}")
                # Check if the executable is found and can be run
                subprocess.run(
                    [micmac_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
        except FileNotFoundError as e:
            logger.error(
                f"Unable to find MicMac executable, skipping reconstruction.Please manually specify the path to the MicMac executable. {e}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running MicMac Tapas, skipping reconstruction.\n{e}")

        # If micmac can be run correctly, try to run Tapas
        logger.info("Running MicMac Tapas...")
        img_list = " ".join([str(e) for e in images])
        cmd = [
            micmac_path,
            "Tapas",
            "RadialStd",
            ".*JPG",
            "Out=Calib",
            "ExpTxt=1",
        ]

        execution = execute(cmd, out_dir)
        for line in execution:
            print(line, end="")

        logger.info("Relative orientation with MicMac done!")


def show_micmac_matches(
    file: Path, image_dir: Path, out: Path = None, **kwargs
) -> np.ndarray:
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
    # x0y0 = []
    # x1y1 = []
    # with open(file, "r") as f:
    #     for line in f:
    #         line = line.split()
    #         x0y0.append(np.array([line[0],line[1]], dtype=np.float32))
    #         x1y1.append(np.array([line[2],line[3]], dtype=np.float32))

    # x0y0 = np.asarray(x0y0)
    # x1y1 = np.asarray(x1y1)

    data = np.loadtxt(file, dtype=np.float32)
    x0y0 = data[:, :2]
    x1y1 = data[:, 2:4]

    # Read the images
    image0 = cv2.imread(str(image_dir / i0))
    image1 = cv2.imread(str(image_dir / i1))

    out = viz_matches_cv2(image0, image1, x0y0, x1y1, out, **kwargs)

    return out


if __name__ == "__main__":

    project_path = Path("datasets/cyprus_micmac2")
    img_dir = project_path / "images"
    feature_path = project_path / "features.h5"
    match_path = project_path / "matches.h5"

    out_micmac = project_path / "micmac"
    if out_micmac.exists():
        shutil.rmtree(out_micmac, ignore_errors=True)
    export_to_micmac(img_dir, feature_path, match_path, out_micmac, run_Tapas=True)

    # Plot the matches
    images = sorted(out_micmac.glob("*.JPG"))
    matches_dir = out_micmac / "match_figs"
    matches_dir.mkdir(exist_ok=True, parents=True)
    with Pool() as p:
        p.starmap(
            show_micmac_matches,
            [
                (
                    out_micmac / "Homol" / f"Pastis{i0.name}" / f"{i1.name}.txt",
                    out_micmac,
                    matches_dir / f"matches_{i0.name}-{i1.name}.png",
                )
                for i0, i1 in combinations(images, 2)
            ],
        )

    print("Done!")
