import sys
from copy import copy

import numpy as np
import torch
from torch import nn

from ..thirdparty.accelerated_features.modules.model import XFeatModel
from .extractor_base import ExtractorBase


class XfeatExtractor(ExtractorBase):
    """
    Class: XfeatExtractor

    This class is a subclass of ExtractorBase and represents a feature extractor using the SuperPoint algorithm.

    Attributes:
        _default_conf (dict): Default configuration for the SuperPointExtractor.
        required_inputs (list): List of required inputs for the extract method.
        grayscale (bool): Flag indicating whether the input images should be converted to grayscale.
        descriptor_size (int): Size of the descriptors size

    Methods:
        __init__(self, config: dict): Initializes the SuperPointExtractor instance with a custom configuration.
        _extract(self, image: np.ndarray) -> dict: Extracts features from an image using the SuperPoint algorithm.
        _frame2tensor(self, image: np.ndarray, device: str = "cpu"): Converts an image to a tensor.
        _resize_image(self, quality: Quality, image: np.ndarray, interp: str = "cv2_area") -> Tuple[np.ndarray]: Resizes an image based on the specified quality.
        _resize_features(self, quality: Quality, features: FeaturesDict) -> Tuple[FeaturesDict]: Resizes features based on the specified quality.
        viz_keypoints(self, image: np.ndarray, keypoints: np.ndarray, output_dir: Path, im_name: str = "keypoints", resize_to: int = 2000, img_format: str = "jpg", jpg_quality: int = 90, ...): Visualizes keypoints on an image and saves the visualization to the specified output directory.
    """

    _default_conf = {
        "name": "superpoint",
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "fix_sampling": False,
    }
    required_inputs = ["image"]
    grayscale = False
    descriptor_size = 64

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self.config.get("extractor")
        self._extractor = XFeatModel(cfg).eval().to(self._device)

    @torch.inference_mode()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image using the SuperPoint model.

        Args:
            image (np.ndarray): The input image as a numpy array.

        Returns:
            np.ndarray: A dictionary containing the extracted features. The keys represent different feature types, and the values are numpy arrays.

        """
        # Convert image from numpy array to tensor
        image_ = self._frame2tensor(image, self._device)

        # Extract features
        feats = self._extractor({"image": image_})

        # Remove elements from list/tuple
        feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
        # Convert tensors to numpy arrays
        feats = {k: v.cpu().numpy() for k, v in feats.items()}

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        """
        Convert a frame to a tensor.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cuda')
        """
        if len(image.shape) == 2:
            image = image[None][None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)[None]
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)
