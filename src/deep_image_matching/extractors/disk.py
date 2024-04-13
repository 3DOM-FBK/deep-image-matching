import numpy as np
import torch

from ..thirdparty.hloc.extractors.disk import DISK
from .extractor_base import ExtractorBase

# TODO: use Kornia implementation of DISK


class DiskExtractor(ExtractorBase):
    _default_conf = {
        "weights": "depth",
        "max_keypoints": 2000,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }
    required_inputs = ["image"]
    grayscale = False
    descriptor_size = 128

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        disk_cfg = self.config.get("extractor")

        # Load extractor
        self._extractor = DISK(disk_cfg).eval().to(self._device)

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        # Convert image from numpy array to tensor
        image_ = self._frame2tensor(image, self._device)

        # Extract features
        feats = self._extractor({"image": image_})

        # Remove elements from list/tuple
        feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
        # Convert tensors to numpy arrays
        feats = {k: v.cpu().numpy() for k, v in feats.items()}

        # Rename keys name 'keypoint_scores' to 'scores'
        feats["scores"] = feats.pop("keypoint_scores")

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        """
        Convert a frame to a tensor.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cuda')
        """
        if len(image.shape) != 3:
            raise ValueError("The input image must have at least 3 channels.")

        # Add batch dimension
        image = image.transpose(2, 0, 1)[None]
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)
