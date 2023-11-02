import logging
from importlib import import_module

import numpy as np
import torch

from .matcher_base import FeaturesDict, MatcherBase

logger = logging.getLogger(__name__)


class LightGlueMatcher(MatcherBase):
    default_config = (
        {
            "descriptor_dim": 256,
            "n_layers": 9,
            "num_heads": 4,
            "flash": True,  # enable FlashAttention if available.
            "mp": False,  # enable mixed precision
            "depth_confidence": 0.95,  # early stopping, disable with -1
            "width_confidence": 0.99,  # point pruning, disable with -1
            "filter_threshold": 0.1,  # match threshold
            "weights": None,
        },
    )

    def __init__(self, local_features="superpoint", **config) -> None:
        """Initializes a LightGlueMatcher"""

        self._localfeatures = local_features
        super().__init__(**config)

        # load the LightGlue module
        LG = import_module("deep_image_matching.thirdparty.LightGlue.lightglue")

        # load the matcher
        sg_cfg = self._config["LightGlue"]
        self._matcher = (
            LG.LightGlue(self._localfeatures, **sg_cfg).eval().to(self._device)
        )

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
        def featuresDict_2_lightglue(feats: FeaturesDict, device: torch.device) -> dict:
            # Remove elements from list/tuple
            feats = {
                k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()
            }
            # Move descriptors dimension to last
            if "descriptors" in feats.keys():
                if feats["descriptors"].shape[-1] != 256:
                    feats["descriptors"] = feats["descriptors"].T
            # Add batch dimension
            feats = {k: v[None] for k, v in feats.items()}
            # Convert to tensor
            feats = {
                k: torch.tensor(v, dtype=torch.float, device=device)
                for k, v in feats.items()
            }
            # Check device
            feats = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in feats.items()
            }

            return feats

        feats0 = featuresDict_2_lightglue(feats0, self._device)
        feats1 = featuresDict_2_lightglue(feats1, self._device)

        # match the features
        match_res = self._matcher({"image0": feats0, "image1": feats1})

        # remove batch dimension and convert to numpy
        mfeats0, mfeats1, matches01 = [
            self._rbd(x) for x in [feats0, feats1, match_res]
        ]
        match_res = {
            k: v.cpu().numpy()
            for k, v in matches01.items()
            if isinstance(v, torch.Tensor)
        }

        # get matching array (indices of matched keypoints in image0 and image1)
        matches01_idx = match_res["matches"]

        return matches01_idx

    def _rbd(self, data: dict) -> dict:
        """Remove batch dimension from elements in data"""
        return {
            k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()
        }
