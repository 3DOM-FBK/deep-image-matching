import os
import subprocess

from pathlib import Path
from deep_image_matching import logger
from typing import Union


def main(
        openmvg_out_path : Path,
        skip_reconstruction : bool,
        system_OS : str,
        openmvg_sfm_bin : Path,
        ):
    
    openmvg_reconstruction_dir = openmvg_out_path / "reconstruction_sequential"
    openmvg_matches_dir = str(openmvg_out_path / "matches")

    if not skip_reconstruction:
        if not os.path.exists(openmvg_reconstruction_dir):
            os.mkdir(openmvg_reconstruction_dir)

        logger.debug("OpenMVG Sequential/Incremental reconstruction")

        if system_OS == "windows":
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
        if system_OS == "linux":
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