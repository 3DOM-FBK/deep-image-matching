import logging
from importlib import import_module

import numpy as np
import torch

from .extractor_base import ExtractorBase

logger = logging.getLogger(__name__)


class DiskExtractor(ExtractorBase):
    def __init__(self, **config: dict):
        # Init the base class
        super().__init__(**config)

        # TODO: improve configuration management!
        disk_cfg = self._config["DISK"]

        # Load extractor
        extractors = import_module("deep_image_matching.hloc.extractors.disk")
        self._extractor = extractors.DISK(disk_cfg).eval().to(self._device)

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
