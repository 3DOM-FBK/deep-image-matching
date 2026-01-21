import os
import sys
import yaml
import deep_image_matching as dim

from pathlib import Path
from deep_image_matching import logger, timer
from deep_image_matching.utils.loftr_roma_to_multiview import LoftrRomaToMultiview


def run_matching(args):
    """
    Main processing function that performs image matching and reconstruction.

    Args:
        args: Configuration arguments (dict or namespace)
    """
    timer.start()

    # Build the configuration from the parameters or command line arguments
    config = dim.Config(args)
    imgs_dir = config.general["image_dir"]
    output_dir = config.general["output_dir"]

    # Initialize ImageMatcher class
    matcher = dim.ImageMatcher(config)

    # Run image matching
    feature_path, match_path = matcher.run()
    timer.update("Image Matching")

    # Export in colmap format
    database_path = output_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    dim.io.export_to_colmap(
        img_dir=imgs_dir,
        feature_path=feature_path,
        match_path=match_path,
        database_path=database_path,
        camera_config_path=config.general["camera_options"],
    )
    timer.update("Export to COLMAP")

    if matcher.matching in ["loftr", "se2loftr", "roma", "srif"]:
        images = os.listdir(imgs_dir)
        image_format = Path(images[0]).suffix
        LoftrRomaToMultiview(
            input_dir=feature_path.parent,
            output_dir=feature_path.parent,
            image_dir=imgs_dir,
            img_ext=image_format,
        )
        timer.update("LoftrRomaToMultiview")

    # Visualize view graph
    if config.general["graph"]:
        try:
            dim.graph.view_graph(database_path, output_dir, imgs_dir)
            timer.update("View Graph")
        except Exception as e:
            logger.error(f"Unable to visualize view graph: {e}")

    # If --skip_reconstruction is not specified, run reconstruction with pycolmap
    if not config.general["skip_reconstruction"]:
        reconst_opts = {}
        model = dim.reconstruction.incremental_reconstruction(
            database_path=output_dir / "database.db",
            image_dir=imgs_dir,
            sfm_dir=output_dir / "reconstruction",
            reconstruction_options=reconst_opts,
            refine_intrinsics=True,
            ignore_two_view_tracks=True,
            filter_min_tri_angle=None,
            export_ply=True,
            export_text=True,
            export_bundler=False,
        )
        timer.update("Reconstruction")

    # Export in openMVG format
    if config.general["openmvg_conf"]:
        with open(config.general["openmvg_conf"]) as file:
            openmvgcfg = yaml.safe_load(file)
        openmvg_sfm_bin = Path(openmvgcfg["general"]["path_to_binaries"])
        openmvg_database = Path(openmvgcfg["general"]["openmvg_database"])
        openmvg_out_path = output_dir / "openmvg"
        dim.io.export_to_openmvg(
            img_dir=imgs_dir,
            feature_path=feature_path,
            match_path=match_path,
            openmvg_out_path=openmvg_out_path,
            openmvg_sfm_bin=openmvg_sfm_bin,
            openmvg_database=openmvg_database,
            camera_config_path=Path(config.general["camera_options"]),
        )
        timer.update("Export to OpenMVG")

        # If skip_reconstruction is not specified, run OpenMVG reconstruction
        if not config.general["skip_reconstruction"]:
            from deep_image_matching.openmvg import openmvg_reconstruction

            openmvg_reconstruction(
                openmvg_out_path=openmvg_out_path,
                skip_reconstruction=config.general["skip_reconstruction"],
                openmvg_sfm_bin=openmvg_sfm_bin,
            )
            timer.update("OpenMVG Reconstruction")

    timer.print("Total execution time")


if __name__ == "__main__":
    """
    Main entry point that handles both CLI and interactive execution.
    """
    if len(sys.argv) > 1:
        # Parse command line arguments
        from deep_image_matching.parser import parse_cli

        args = parse_cli()
    else:
        # Define configuration for interactive use
        args = {
            "dir": "./assets/example_cyprus",
            "pipeline": "superpoint+lightglue",
            "strategy": "bruteforce",
            "quality": "medium",
            "tiling": "preselection",
            "camera_options": "./assets/example_cyprus/cameras.yaml",
            "openmvg": None,
            "force": True,
            "skip_reconstruction": False,
            "graph": True,
            "verbose": True,
        }

    run_matching(args)
