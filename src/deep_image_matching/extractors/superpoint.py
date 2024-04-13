import sys
from copy import copy

import numpy as np
import torch
from torch import nn

from ..thirdparty.SuperGluePretrainedNetwork.models import superpoint
from .extractor_base import ExtractorBase

# TODO: Use Superpoint implementation from LightGlue


# The original keypoint sampling is incorrect. We patch it here but
# we don't fix it upstream to not impact exisiting evaluations.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    _default_conf = {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "fix_sampling": True,
    }
    required_inputs = ["image"]
    detection_noise = 2.0

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self._default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)

    def _init(self, conf):
        if conf["fix_sampling"]:
            superpoint.sample_descriptors = sample_descriptors_fix_sampling
        self.net = superpoint.SuperPoint(conf)

    def _forward(self, data):
        return self.net(data)


class SuperPointExtractor(ExtractorBase):
    """
    Class: SuperPointExtractor

    This class is a subclass of ExtractorBase and represents a feature extractor using the SuperPoint algorithm.

    Attributes:
        _default_conf (dict): Default configuration for the SuperPointExtractor.
        required_inputs (list): List of required inputs for the SuperPointExtractor.
        grayscale (bool): Flag indicating whether the input images should be converted to grayscale.
        descriptor_size (int): Size of the descriptors extracted by the SuperPoint algorithm.
        detection_noise (float): Noise level for keypoint detection.

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
    grayscale = True
    descriptor_size = 256
    detection_noise = 2.0

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        SP_cfg = self.config.get("extractor")
        self._extractor = SuperPoint(SP_cfg).eval().to(self._device)

    @torch.no_grad()
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
