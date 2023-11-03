import logging

import numpy as np
import torch

from ..hloc.extractors.superpoint import SuperPoint
from .extractor_base import ExtractorBase

logger = logging.getLogger(__name__)


# TODO: skip the loading of hloc extractor, but implement it directly here.
class AlikeExtractor(ExtractorBase):
    default_conf = (
        {
            "model": "alike-s",
            "device": "cuda",
            "top_k": 15000,
            "scores_th": 0.2,
            "n_limit": 15000,
            "subpixel": True,
        },
    )
    required_inputs = []
    grayscale = True
    descriptor_size = 256
    detection_noise = 2.0

    def __init__(self, **config: dict):
        # Init the base class
        super().__init__(**config)

        # TODO: improve configuration management!
        SP_cfg = {**self.default_conf, **self._config["SuperPoint"]}

        # Load extractor
        self._extractor = SuperPoint(SP_cfg).eval().to(self._device)

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        # Convert image from numpy array to tensor
        image_ = self._frame2tensor(image, self._device)

        # Extract features
        feats = self._extractor({"image": image_})

        # Remove elements from list/tuple
        feats = {
            k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()
        }
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
