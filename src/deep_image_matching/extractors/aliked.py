import numpy as np
import torch

from ..thirdparty.LightGlue.lightglue.aliked import ALIKED
from .extractor_base import ExtractorBase

# TODO: replace LightGlue implementation of ALIKED with the original ALIKED code


class AlikedExtractor(ExtractorBase):
    # config from original ALIKED implementation
    # _default_conf = {
    #     "name:": "aliked",
    #     "model": "aliked-n16rot",
    #     "device": "cuda",
    #     "top_k": 2000,
    #     "scores_th": 0.2,
    #     "n_limit": 8000,
    # }

    # config from LightGlue implementation of ALIKED
    _default_conf = {
        "name:": "aliked",
        "model": "aliked-n16rot",
        "device": "cuda",
        "max_num_keypoints": 4000,
        "detection_threshold": 0.2,
        "nms_radius": 2,
    }
    required_inputs = []
    grayscale = False
    descriptor_size = 128

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self.config.get("extractor")
        if self._device == "cuda" and torch.cuda.is_available():
            self._extractor = ALIKED(**cfg).cuda()
        else:
            self._extractor = ALIKED(**cfg)

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        image_ = self._frame2tensor(image, self._device)

        # Extract features
        feats = self._extractor({"image": image_})

        # Remove batch dimension from elements in feats
        feats = self._rbd(feats)

        # Convert tensors to numpy arrays
        feats = {k: v.cpu().numpy() for k, v in feats.items()}

        # Transpose descriptors
        feats["descriptors"] = feats["descriptors"].T

        # Rename 'keypoint_scores' to 'scores'
        feats["scores"] = feats.pop("keypoint_scores")

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
        return {k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v for k, v in data.items()}


if __name__ == "__main__":
    pass
