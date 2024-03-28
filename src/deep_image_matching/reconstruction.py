import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap

from . import logger
from .triangulation import (
    OutputCapture,
)
from .utils.database import COLMAPDatabase


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_list=image_list or [],
            options=options,
        )


def update_cameras(database_path: Path, cameras: List[pycolmap.Camera]):
    if not all([isinstance(cam, pycolmap.Camera) for cam in cameras]):
        raise ValueError("cameras must be a list of pycolmap.Camera objects.")

    db = COLMAPDatabase.connect(database_path)

    num_cameras = len(db.execute("SELECT * FROM cameras;").fetchall())
    if num_cameras != len(cameras):
        raise ValueError(
            f"Number of cameras in the database ({num_cameras}) "
            f"does not match the number of cameras provided ({len(cameras)})."
        )

    for camera_id, cam in enumerate(cameras, start=1):
        db.update_camera(
            camera_id, cam.model.value, cam.width, cam.height, cam.params, True
        )
    db.commit()
    db.close()


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def pycolmap_reconstruction(
    database_path: Path,
    sfm_dir: Path,
    image_dir: Path,
    refine_intrinsics: bool = True,
    options: Optional[Dict[str, Any]] = {},
    export_text: bool = True,
    export_bundler: bool = True,
    export_ply: bool = True,
    verbose: bool = False,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")

    if not refine_intrinsics:
        options = {
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
            **options,
        }
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=options
            )

    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(
        f"Largest model is #{largest_index} " f"with {largest_num_images} images."
    )

    for index, model in reconstructions.items():
        if len(reconstructions) > 1:
            logger.info(f"Exporting model #{index}...")
            reconstruction_dir = sfm_dir / "reconstruction" / f"model_{index}"
        else:
            logger.info("Exporting model...")
            reconstruction_dir = sfm_dir / "reconstruction"
        reconstruction_dir.mkdir(exist_ok=True, parents=True)

        # Export reconstruction in Colmap format
        model.write(reconstruction_dir)

        # Export ply
        if export_ply:
            model.export_PLY(reconstruction_dir / "rec.ply")

        # Export reconstruction in text format
        if export_text:
            model.write_text(str(reconstruction_dir))

        # Export reconstruction in Bundler format
        if export_bundler:
            fname = "bundler"
            model.export_bundler(
                reconstruction_dir / (fname + ".out"),
                reconstruction_dir / (fname + "_list.txt"),
                skip_distortion=True,
            )

    return reconstructions[largest_index]
