import sys
from copy import copy
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..thirdparty.accelerated_features.modules.interpolator import InterpolateSparse2d
from ..thirdparty.accelerated_features.modules.model import XFeatModel
from .extractor_base import ExtractorBase


class XFeatExtractor(ExtractorBase):
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
        _preprocess_input(self, image: np.ndarray, device: str = "cpu"): Converts an image to a tensor.
        _resize_image(self, quality: Quality, image: np.ndarray, interp: str = "cv2_area") -> Tuple[np.ndarray]: Resizes an image based on the specified quality.
        _resize_features(self, quality: Quality, features: FeaturesDict) -> Tuple[FeaturesDict]: Resizes features based on the specified quality.
        viz_keypoints(self, image: np.ndarray, keypoints: np.ndarray, output_dir: Path, im_name: str = "keypoints", resize_to: int = 2000, img_format: str = "jpg", jpg_quality: int = 90, ...): Visualizes keypoints on an image and saves the visualization to the specified output directory.
    """

    _default_conf = {
        "name": "xfeat",
        "top_k": 4000,
    }
    grayscale = False
    descriptor_size = 64
    weights_url = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        self._extractor = XFeatModel().to(self._device).eval()
        self._extractor.load_state_dict(
            torch.hub.load_state_dict_from_url(self.weights_url, map_location=self._device, progress=True)
        )

        self.interpolator = InterpolateSparse2d("bicubic")

    @torch.inference_mode()
    def _extract(self, image: Union[np.ndarray, torch.Tensor]) -> dict:
        """
        Extract features from an image.

        Args:
            image [np.ndarray, torch.Tensor]: Image to extract features from. Shape: [H, W, C] or [B, C, H, W]

        Returns:
            dict: A dictionary containing the extracted features with the following keys:
                - 'keypoints' -> np.ndarray: keypoints (top_k, 2): keypoints
                - 'scores' -> np.ndarray: keypoint scores (top_k,): keypoint scores
                - 'descriptors' -> np.ndarray: local features (64, top_k): local features

        """
        top_k = self.config.extractor["top_k"]

        image_, rh1, rw1 = self._preprocess_input(image)

        B, _, _H1, _W1 = image_.shape

        # Extract features
        M1, K1, H1 = self._extractor(image_)
        M1 = F.normalize(M1, dim=1)

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=0.05, kernel_size=5)

        # Compute reliability scores
        _nearest = InterpolateSparse2d("nearest")
        _bilinear = InterpolateSparse2d("bilinear")
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        # Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)

        # Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0

        out = [
            {"keypoints": mkpts[b][valid[b]], "scores": scores[b][valid[b]], "descriptors": feats[b][valid[b]]}
            for b in range(B)
        ]

        # Convert tensors to numpy arrays and remove batch dimension
        out = {k: v.cpu().numpy() for k, v in out[0].items()}

        # Transpose descriptors
        out["descriptors"] = out["descriptors"].T

        return out

    def _preprocess_input(self, x: Union[np.ndarray, torch.Tensor]):
        """Guarantee that image is divisible by 32 to avoid aliasing artifacts."""
        if isinstance(x, np.ndarray) and x.ndim == 3:
            x = torch.tensor(x).permute(2, 0, 1)[None]
        elif isinstance(x, np.ndarray) and x.ndim == 2:
            x = torch.tensor(x)[None][None]

        x = x.to(self._device).float()

        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        rh, rw = H / _H, W / _W

        x = F.interpolate(x, (_H, _W), mode="bilinear", align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def NMS(self, x, threshold=0.05, kernel_size=5):
        B, _, H, W = x.shape
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        # Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, : len(pos_batched[b]), :] = pos_batched[b]

        return pos
