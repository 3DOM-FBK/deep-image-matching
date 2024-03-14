import os
import shutil
import subprocess
import sys
from pathlib import Path

from deep_image_matching import logger


def main(
    openmvg_out_path: Path,
    skip_reconstruction: bool,
    openmvg_sfm_bin: Path = None,
):
    openmvg_reconstruction_dir = openmvg_out_path / "reconstruction_sequential"
    openmvg_matches_dir = str(openmvg_out_path / "matches")

    if not skip_reconstruction:
        if not os.path.exists(openmvg_reconstruction_dir):
            os.mkdir(openmvg_reconstruction_dir)

        logger.debug("OpenMVG Sequential/Incremental reconstruction")

        if sys.platform in ["windows", "win32"]:
            if openmvg_sfm_bin is None:
                raise ValueError(
                    "openMVG binaries path is not provided. Please provide the path to openMVG binaries."
                )
            pRecons = subprocess.Popen(
                [
                    openmvg_sfm_bin / "openMVG_main_IncrementalSfM",
                    "-i",
                    openmvg_matches_dir + "/sfm_data.json",
                    "-m",
                    openmvg_matches_dir,
                    "-o",
                    openmvg_reconstruction_dir,
                ]
            )
        if sys.platform == "linux":
            if openmvg_sfm_bin is None:
                openmvg_sfm_bin = shutil.which("openMVG_main_SfM")
                if openmvg_sfm_bin is None:
                    raise FileNotFoundError(
                        "openMVG binaries path is not provided and DIM is not able to find it automatically. Please provide the path to openMVG binaries."
                    )
            openmvg_sfm_bin = Path(openmvg_sfm_bin)
            pRecons = subprocess.Popen(
                [
                    openmvg_sfm_bin / "openMVG_main_SfM",
                    "--sfm_engine",
                    "INCREMENTAL",
                    "-i",
                    openmvg_matches_dir + "/sfm_data.json",
                    "-m",
                    openmvg_matches_dir,
                    "-o",
                    openmvg_reconstruction_dir,
                ]
            )
        pRecons.wait()

    return


if __name__ == "__main__":
    pass
