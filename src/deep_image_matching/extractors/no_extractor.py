from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch

from .. import logger
from ..utils.image import Image
from .extractor_base import ExtractorBase


# TODO: skip the loading of hloc extractor, but implement it directly here.
class NoExtractor(ExtractorBase):
    default_conf = {}

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

    def extract(self, img: Union[Image, Path, str]) -> np.ndarray:
        """ovrewrite the base class method to always return an empty dict"""

        if isinstance(img, str):
            im_path = Path(img)
        elif isinstance(img, Image):
            im_path = img.path
        elif isinstance(img, Path):
            im_path = img
        else:
            raise TypeError(
                "Invalid image path. 'img' must be a string, a Path or an Image object"
            )
        if not im_path.exists():
            raise ValueError(f"Image {im_path} does not exist")

        output_dir = Path(self._config["general"]["output_dir"])
        feature_path = output_dir / "features.h5"
        output_dir.mkdir(parents=True, exist_ok=True)
        im_name = im_path.name

        # Create dummy empty features
        features = {}
        # features["feature_path"] = str(feature_path)
        # features["im_path"] = str(im_path)
        features["keypoints"] = np.array([])
        features["descriptors"] = np.array([])
        features["scores"] = np.array([])

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                if im_name in fd:
                    del fd[im_name]
                grp = fd.create_group(im_name)
                for k, v in features.items():
                    if k == "im_path" or k == "feature_path":
                        grp.create_dataset(k, data=str(v))
                    if isinstance(v, np.ndarray):
                        grp.create_dataset(k, data=v)
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    del grp, fd[im_name]
                raise error

        return feature_path

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        feats = {}
        feats["keypoints"] = np.array([])
        feats["descriptors"] = np.array([])
        feats["scores"] = np.array([])

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        """
        Convert a frame to a tensor.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cuda')
        """
        pass


if __name__ == "__main__":
    pass
