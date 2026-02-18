import torch
import yaml
import numpy as np
import torch
from pathlib import Path

from ..thirdparty.liftfeat.models.liftfeat_wrapper import MODEL_PATH, LiftFeat
from .extractor_base import ExtractorBase


class XfeatExtractor(ExtractorBase):
    _default_conf = {
        "name": "xfeat",
        "max_num_keypoints": 4000,
        "detect_threshold": 0.05,
    }
    required_inputs = []
    grayscale = False
    descriptor_size = 128
    

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self.config.get("extractor")
        detct_threshold = cfg.get("detect_threshold", self._default_conf["detect_threshold"])

        self._extractor = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = self._default_conf["max_num_keypoints"])
        self.max_num_keypoints = cfg.get("max_num_keypoints", self._default_conf["max_num_keypoints"])


    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        # Extract features using LiftFeat's extract method (expects numpy array)
        feats = self._extractor.detectAndCompute(image, top_k = self.max_num_keypoints)[0]

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

    def _rbd(self, data: dict) -> dict:
        """Remove batch dimension from elements in data"""
        return {
            k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()
        }


if __name__ == "__main__":
    pass
