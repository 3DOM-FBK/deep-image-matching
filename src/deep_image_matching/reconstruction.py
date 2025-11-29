import logging
import multiprocessing
from pathlib import Path
from typing import Any, Optional, Union

import enlighten
import pycolmap

logger = logging.getLogger("dim")


def incremental_mapping_with_pbar(database_path, image_path, sfm_path):
    database = pycolmap.Database()
    database.open(database_path)
    num_images = database.num_images
    with enlighten.Manager() as manager:
        with manager.counter(total=num_images, desc="Images registered:") as pbar:
            pbar.update(0, force=True)
            reconstructions = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )
    return reconstructions


def incremental_reconstruction(
    database_path: Path,
    image_dir: Path,
    sfm_dir: Path,
    refine_intrinsics: bool = True,
    ignore_two_view_tracks: bool = True,
    filter_min_tri_angle: Union[float, None] = None,
    reconstruction_options: Optional[dict[str, Any]] = None,
    export_ply: bool = True,
    export_text: bool = False,
    export_bundler: bool = False,
) -> Union[pycolmap.Reconstruction, None]:
    logger.info("Running 3D reconstruction...")

    if not database_path.exists():
        logger.error(f"Database file {database_path} does not exist.")
        return None

    if reconstruction_options is None:
        reconstruction_options = {}

    try:
        num_threads = reconstruction_options.pop(
            "num_threads", multiprocessing.cpu_count()
        )
        pipeline_options = pycolmap.IncrementalPipelineOptions(
            num_threads=num_threads, **(reconstruction_options or {})
        )
    except TypeError:
        logger.error(
            "Invailid options for IncrementalPipelineOptions. Using default options."
        )
        pipeline_options = pycolmap.IncrementalPipelineOptions()

    if not refine_intrinsics:
        pipeline_options.ba_refine_focal_length = False
        pipeline_options.ba_refine_principal_point = False
        pipeline_options.ba_refine_extra_params = False

    if not ignore_two_view_tracks:
        pipeline_options.triangulation.ignore_two_view_tracks = False

    if filter_min_tri_angle is not None:
        pipeline_options.mapper.filter_min_tri_angle = filter_min_tri_angle

    sfm_dir.mkdir(exist_ok=True, parents=True)
    reconstructions = incremental_mapping_with_pbar(database_path, image_dir, sfm_dir)

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
    logger.info(f"Largest model is #{largest_index} with {largest_num_images} images.")

    # Exporting the models in other formats
    for index, model in reconstructions.items():
        reconstruction_dir = sfm_dir / f"{index}"

        # Export ply
        if export_ply:
            model.export_PLY(str(reconstruction_dir / "rec.ply"))

        # Export reconstruction in text format
        if export_text:
            model.write_text(str(reconstruction_dir))

        # Export reconstruction in Bundler format
        if export_bundler:
            logger.warning(
                "Exporting reconstruction in Bundler format is deprecated and not implemented anymore in pycolmap. Use the script export_to_bundler.py in the COLMAP repository: https://github.com/colmap/colmap/blob/main/scripts/python/export_to_bundler.py"
            )

    return reconstructions[largest_index]
