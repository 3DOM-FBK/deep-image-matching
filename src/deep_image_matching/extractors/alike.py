import numpy as np
import torch

from ..thirdparty.alike.alike import ALike, configs
from .extractor_base import ExtractorBase


class AlikeExtractor(ExtractorBase):
    _default_conf = {
        "name:": "alike",
        "model": "alike-s",
        "device": "cuda",
        "top_k": 15000,
        "scores_th": 0.2,
        "n_limit": 15000,
        "subpixel": True,
    }
    required_inputs = []
    grayscale = False
    descriptor_size = 96

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self.config.get("extractor")
        self._extractor = ALike(
            **configs[cfg["model"]],
            device=cfg["device"],
            top_k=cfg["top_k"],
            scores_th=cfg["scores_th"],
            n_limit=cfg["n_limit"],
        )

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        # Extract features
        feats = self._extractor(image, sub_pixel=True)

        # Transpose descriptors
        feats["descriptors"] = feats["descriptors"].T

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
