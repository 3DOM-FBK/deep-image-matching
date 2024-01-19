import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap

from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
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


def update_cameras(database_path: Path, cameras: [List[pycolmap.Camera]]):
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
            camera_id, cam.model_id, cam.width, cam.height, cam.params, True
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
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    # options = {"ignore_two_view_track": False}
    if options is None:
        options = {}
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

    return reconstructions[largest_index]


def main(
    database: Path,
    image_dir: Path,
    feature_path: Path,
    match_path: Path,
    pair_path: Path,
    sfm_dir: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    cameras=None,
    skip_geometric_verification: bool = False,
    export_text: bool = True,
    export_bundler: bool = True,
    export_ply: bool = True,
    reconst_opts: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pycolmap.Reconstruction:
    # Create empty database
    create_empty_db(database)
    import_images(image_dir, database, camera_mode)

    # Update cameras intrinsics in the database
    if cameras:
        update_cameras(database, cameras)

    # Import features and matches
    image_ids = get_image_ids(database)
    import_features(image_ids, database, feature_path)
    import_matches(
        image_ids,
        database,
        match_path,
        skip_geometric_verification=skip_geometric_verification,
    )

    # Run geometric verification
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pair_path, verbose=verbose)

    # Run reconstruction
    model = pycolmap_reconstruction(
        sfm_dir=sfm_dir,
        database_path=database,
        image_dir=image_dir,
        verbose=verbose,
        options=reconst_opts,
    )
    if model is not None:
        logger.info(
            f"Reconstruction statistics:\n{model.summary()}"
            + f"\n\tnum_input_images = {len(image_ids)}"
        )

        # Copy images to sfm_dir (for debugging)
        # shutil.copytree(image_dir, sfm_dir / "images", dirs_exist_ok=True)

        # Create reconstruction directory
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

    else:
        logger.error("Pycolmap reconstruction failed")
    return model


if __name__ == "__main__":
    pass
