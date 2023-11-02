import logging
from importlib import import_module

import numpy as np
import torch

from .extractor_base import ExtractorBase

logger = logging.getLogger(__name__)


# TODO: skip the loading of hloc extractor, but implement it directly here.
class SuperPointExtractor(ExtractorBase):
    def __init__(self, **config: dict):
        # Init the base class
        super().__init__(**config)

        # TODO: improve configuration management!
        SP_cfg = self._config["SuperPoint"]

        # Load extractor
        extractors = import_module("deep_image_matching.hloc.extractors.superpoint")
        self._extractor = extractors.SuperPoint(SP_cfg).eval().to(self._device)

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
