import contextlib
import io
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pycolmap

from .database import COLMAPDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputCapture:
    """
    Context manager for capturing standard output and suppressing it if necessary.

    Args:
        verbose (bool): If True, standard output is not suppressed.
                        If False, standard output is captured and
                        optionally logged as an error.
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                logger.error("Failed with output:\n%s", self.out.getvalue())
        sys.stdout.flush()


def get_pairs_from_file(pair_file: Path) -> list:
    """
    Reads image pairs from a text file.

    Args:
        pair_file (Path): Path to a text file containing image pairs,
                          one pair per line, separated by a space.

    Returns:
        list: A list of tuples where each tuple represents an image pair
              (image1 , image2).
    """
    pairs = []
    with open(pair_file, "r") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            im1, im2 = line.strip().split(" ", 1)
            pairs.append((im1, im2))
    return pairs


def to_homogeneous(p):
    return np.pad(p, ((0, 0),) * (p.ndim - 1) + ((0, 1),), constant_values=1)


def vector_to_cross_product_matrix(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def compute_epipolar_errors(j_from_i: pycolmap.Rigid3d, p2d_i, p2d_j):
    j_E_i = j_from_i.essential_matrix()
    l2d_j = to_homogeneous(p2d_i) @ j_E_i.T
    l2d_i = to_homogeneous(p2d_j) @ j_E_i
    dist = np.abs(np.sum(to_homogeneous(p2d_i) * l2d_i, axis=1))
    errors_i = dist / np.linalg.norm(l2d_i[:, :2], axis=1)
    errors_j = dist / np.linalg.norm(l2d_j[:, :2], axis=1)
    return errors_i, errors_j


def create_db_from_model(reconstruction: pycolmap.Reconstruction, database_path: Path) -> Dict[str, int]:
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()

    with COLMAPDatabase.connect(database_path) as db:
        db.create_tables()

        for i, camera in reconstruction.cameras.items():
            db.add_camera(
                camera.model.value,
                camera.width,
                camera.height,
                camera.params,
                camera_id=i,
                prior_focal_length=True,
            )

        for i, image in reconstruction.images.items():
            db.add_image(image.name, image.camera_id, image_id=i)

        db.commit()

    return {image.name: i for i, image in reconstruction.images.items()}
