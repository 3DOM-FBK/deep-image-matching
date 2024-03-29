#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
This module contains functions to export image features and matches to a COLMAP database.
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm

from ..utils.database import COLMAPDatabase, image_ids_to_pair_id

logger = logging.getLogger("dim")

default_camera_options = {
    "general": {
        "single_camera": False,
        "camera_model": "simple-radial",
    },
}


def export_to_colmap(
    img_dir: Path,
    feature_path: Path,
    match_path: Path,
    database_path: str = "database.db",
    camera_options: dict = default_camera_options,
):
    """
    Exports image features and matches to a COLMAP database.

    Args:
        img_dir (str): Path to the directory containing the source images.
        feature_path (Path): Path to the feature file (in HDF5 format) containing the extracted keypoints.
        match_path (Path): Path to the match file (in HDF5 format) containing the matches between keypoints.
        database_path (str, optional): Path to the COLMAP database file. Defaults to "colmap.db".
        camera_options (dict, optional): Flag indicating whether to use camera options. Defaults to default_camera_options.

    Returns:
        None

    Raises:
        IOError: If the image path is invalid.

    Warnings:
        If the database path already exists, it will be deleted and recreated.
        If a pair of images already has matches in the database, a warning will be raised.

    Example:
        export_to_colmap(
            img_dir="/path/to/images",
            feature_path=Path("/path/to/features.h5"),
            match_path=Path("/path/to/matches.h5"),
            database_path="colmap.db",
        )
    """
    database_path = Path(database_path)
    if database_path.exists():
        logger.warning(f"Database path {database_path} already exists - deleting it")
        database_path.unlink()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    fname_to_id = add_keypoints(db, feature_path, img_dir, camera_options)
    raw_match_path = match_path.parent / "raw_matches.h5"
    if raw_match_path.exists():
        add_raw_matches(
            db,
            raw_match_path,
            fname_to_id,
        )
    if match_path.exists():
        add_matches(
            db,
            match_path,
            fname_to_id,
        )

    db.commit()
    return


def get_focal(image_path: Path, err_on_default: bool = False) -> float:
    """
    Get the focal length of an image.

    Parameters:
        image_path (Path): The path to the image file.
        err_on_default (bool, optional): Whether to raise an error if the focal length cannot be determined from the image's EXIF data. Defaults to False.

    Returns:
        float: The focal length of the image.

    Raises:
        RuntimeError: If the focal length cannot be determined from the image's EXIF data and `err_on_default` is set to True.

    Note:
        This function calculates the focal length based on the maximum size of the image and the EXIF data. If the focal length cannot be determined from the EXIF data, it uses a default prior value.

    """
    image = Image.open(image_path)
    max_size = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35.0 * max_size

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def create_camera(db: Path, image_path: Path, camera_model: str):
    image = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

    if camera_model == "simple-pinhole":
        model = 0  # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    elif camera_model == "pinhole":
        model = 1  # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == "simple-radial":
        model = 2  # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == "opencv":
        model = 4  # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0])
    else:
        raise RuntimeError(f"Invalid camera model {camera_model}")

    return db.add_camera(model, width, height, param_arr)


def parse_camera_options(
    camera_options: dict,
    db: Path,
    image_path: Path,
) -> dict:
    """
    Parses camera options and creates camera entries in the COLMAP database.

    This function groups images by camera, assigns camera IDs, and attempts to
    initialize camera models in the provided COLMAP database.

    Args:
        camera_options (dict): A dictionary containing camera configuration options.
        db (Path): Path to the COLMAP database.
        image_path (Path): Path to the directory containing source images.

    Returns:
        dict: A dictionary mapping image filenames to their assigned camera IDs.
    """

    grouped_images = {}
    n_cameras = len(camera_options.keys()) - 1
    for camera in range(n_cameras):
        cam_opt = camera_options[f"cam{camera}"]
        images = cam_opt["images"].split(",")
        for i, img in enumerate(images):
            grouped_images[img] = {"camera_id": camera + 1}
            if i == 0:
                path = os.path.join(image_path, img)
                try:
                    create_camera(db, path, cam_opt["camera_model"])
                except:
                    logger.warning(f"Was not possible to load the first image to initialize cam{camera}")
    return grouped_images


def add_keypoints(db: Path, h5_path: Path, image_path: Path, camera_options: dict = {}) -> dict:
    """
    Adds keypoints from an HDF5 file to a COLMAP database.

    Reads keypoints from an HDF5 file, associates them with cameras (if necessary),
    and adds the image and keypoint information to the specified COLMAP database.

    Args:
        db (Path): Path to the COLMAP database.
        h5_path (Path): Path to the HDF5 file containing keypoints.
        image_path (Path): Path to the directory containing source images.
        camera_options (dict, optional): Camera configuration options (see `parse_camera_options`).
                                         Defaults to an empty dictionary.

    Returns:
        dict: A dictionary mapping image filenames to their corresponding image IDs in the database.
    """

    grouped_images = parse_camera_options(camera_options, db, image_path)

    with h5py.File(str(h5_path), "r") as keypoint_f:
        # camera_id = None
        fname_to_id = {}
        k = 0
        for filename in tqdm(list(keypoint_f.keys())):
            keypoints = keypoint_f[filename]["keypoints"].__array__()

            path = os.path.join(image_path, filename)
            if not os.path.isfile(path):
                raise IOError(f"Invalid image path {path}")

            if filename not in list(grouped_images.keys()):
                if camera_options["general"]["single_camera"] is False:
                    camera_id = create_camera(db, path, camera_options["general"]["camera_model"])
                elif camera_options["general"]["single_camera"] is True:
                    if k == 0:
                        camera_id = create_camera(db, path, camera_options["general"]["camera_model"])
                        single_camera_id = camera_id
                        k += 1
                    elif k > 0:
                        camera_id = single_camera_id
            else:
                camera_id = grouped_images[filename]["camera_id"]
            image_id = db.add_image(filename, camera_id)
            fname_to_id[filename] = image_id
            # print('keypoints')
            # print(keypoints)
            # print('image_id', image_id)
            if len(keypoints.shape) >= 2:
                db.add_keypoints(image_id, keypoints)
            # else:
            #    keypoints =
            #    db.add_keypoints(image_id, keypoints)

    return fname_to_id


def add_raw_matches(db: Path, h5_path: Path, fname_to_id: dict):
    """
    Adds raw feature matches from an HDF5 file to a COLMAP database.

    Reads raw matches from an HDF5 file, maps image filenames to their image IDs,
    and adds match information to the specified COLMAP database. Prevents duplicate
    matches from being added.

    Args:
        db (Path): Path to the COLMAP database.
        h5_path (Path): Path to the HDF5 file containing raw matches.
        fname_to_id (dict):  A dictionary mapping image filenames to their image IDs.
    """
    match_file = h5py.File(str(h5_path), "r")

    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue

                matches = group[key_2][()]
                db.add_matches(id_1, id_2, matches)
                # db.add_two_view_geometry(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)
    match_file.close()


def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(str(h5_path), "r")

    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue

                matches = group[key_2][()]
                # db.add_matches(id_1, id_2, matches)
                db.add_two_view_geometry(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)
    match_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", help=("Path to the directory with " "keypoints.h5 and matches.h5"))
    parser.add_argument("image_path", help="Path to source images")
    parser.add_argument(
        "--image-extension",
        default=".jpg",
        type=str,
        help="Extension of files in image_path",
    )
    parser.add_argument(
        "--database-path",
        default="database.db",
        help="Location where the COLMAP .db file will be created",
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help=(
            "Consider all photos to be made with a single camera (COLMAP "
            "will reduce the number of degrees of freedom"
        ),
    )
    parser.add_argument(
        "--camera-model",
        choices=["simple-pinhole", "pinhole", "simple-radial", "opencv"],
        default="simple-radial",
        help=(
            "Camera model to use in COLMAP. "
            "See https://github.com/colmap/colmap/blob/master/src/base/camera_models.h"
            " for explanations"
        ),
    )

    args = parser.parse_args()

    if args.camera_model == "opencv" and not args.single_camera:
        raise RuntimeError(
            "Cannot use --camera-model=opencv camera without "
            "--single-camera (the COLMAP optimisation will "
            "likely fail to converge)"
        )

    if os.path.exists(args.database_path):
        raise RuntimeError("database path already exists - will not modify it.")

    db = COLMAPDatabase.connect(args.database_path)
    db.create_tables()

    fname_to_id = add_keypoints(
        db,
        args.h5_path,
        args.image_path,
        args.image_extension,
        args.camera_model,
        args.single_camera,
    )
    add_matches(
        db,
        args.h5_path,
        fname_to_id,
    )

    db.commit()
