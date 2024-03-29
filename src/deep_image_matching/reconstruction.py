import contextlib
import io
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pycolmap

logger = logging.getLogger("dim")


class OutputCapture:
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
            reconstructions = pycolmap.incremental_mapping(database_path, image_dir, models_path, options=options)

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
    logger.info(f"Largest model is #{largest_index} " f"with {largest_num_images} images.")

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
