import yaml
import numpy as np
import torch
from pathlib import Path

from ..thirdparty.rdd.RDD import RDD
from .extractor_base import ExtractorBase


class RDDSparseExtractor(ExtractorBase):
    _default_conf = {
        "name": "aliked",
        "max_num_keypoints": 4000,
    }
    required_inputs = []
    grayscale = False
    descriptor_size = 128
    

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self.config.get("extractor")
        config_path = Path(__file__).parent.parent / 'thirdparty' / 'rdd' / 'configs' / 'default.yaml'
        weights_path = Path(__file__).parent.parent / 'thirdparty' / 'rdd' / 'RDD' / 'weights' / 'RDD-v2.pth'
        
        with open(config_path, 'r') as f:
            network_config = yaml.safe_load(f)
        self._extractor = RDD.build(config=network_config, weights=str(weights_path))
        self.max_num_keypoints = cfg.get("max_num_keypoints", self._default_conf["max_num_keypoints"])

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        image_ = self._frame2tensor(image, self._device)

        # Extract features using RDD's extract method
        feats_list = self._extractor.extract(image_)
        
        # Get the first batch element (batch size is 1)
        feats = feats_list[0]

        # Convert tensors to numpy arrays
        feats = {k: v.cpu().numpy() for k, v in feats.items()}

        feats["keypoints"] = feats["keypoints"][:self.max_num_keypoints, :]
        feats["descriptors"] = feats["descriptors"][:self.max_num_keypoints, :]

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
