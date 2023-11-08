import logging

import numpy as np
import torch

from ..thirdparty.LightGlue.lightglue import LightGlue
from .matcher_base import FeaturesDict, MatcherBase

logger = logging.getLogger(__name__)


class LightGlueMatcher(MatcherBase):
    default_conf = {
        "name": "kornia_matcher",
        "match_mode": "snn",
        "th": 0.8,
    }
    required_inputs = []
    min_matches = 20
    max_feat_no_tiling = 200000

    def __init__(self, local_features="superpoint", **config) -> None:
        """Initializes a LightGlueMatcher"""

        self._localfeatures = local_features
        super().__init__(**config)

        # load the matcher
        sg_cfg = {**self.default_conf, **self._config.get("kornia_matcher", {})}
        self._matcher = LightGlue(self._localfeatures, **sg_cfg).eval().to(self._device)

        if self._localfeatures == "disk":
            self.max_feat_no_tiling = 50000

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
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
