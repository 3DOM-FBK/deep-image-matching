import logging
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch

from ..utils.image import Image
from .extractor_base import ExtractorBase

logger = logging.getLogger("dim")


class NoExtractor(ExtractorBase):
    _default_conf = {}

    def __init__(self, config: dict):
        super().__init__(config)

    def extract(self, img: Union[Image, Path, str]) -> np.ndarray:
        """
        Extract features from an image. This is the main method of the feature extractor.

        Args:
                img: Image to extract features from. It can be either a path to an image or an Image object

        Returns:
                List of features extracted from the image. Each feature is a 2D NumPy array
        """

        if isinstance(img, str):
            im_path = Path(img)
        elif isinstance(img, Image):
            im_path = img.path
        elif isinstance(img, Path):
            im_path = img
        else:
            raise TypeError("Invalid image path. 'img' must be a string, a Path or an Image object")
        if not im_path.exists():
            raise ValueError(f"Image {im_path} does not exist")

        output_dir = Path(self.config["general"]["output_dir"])
        feature_path = output_dir / "features.h5"
        output_dir.mkdir(parents=True, exist_ok=True)
        im_name = im_path.name

        # Build fake features
        features = {}
        features["keypoints"] = np.array([])
        features["descriptors"] = np.array([])
        features["scores"] = np.array([])
        img_obj = Image(im_path)
        # img_obj.read_exif()
        features["image_size"] = np.array(img_obj.size)
        features["tile_idx"] = np.array([])
        features["im_path"] = im_path
        features["feature_path"] = feature_path

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
