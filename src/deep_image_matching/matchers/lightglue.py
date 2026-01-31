import numpy as np
import torch

from ..thirdparty.LightGlue.lightglue import LightGlue
from .matcher_base import FeaturesDict, MatcherBase


def featuresDict2Lightglue(feats: dict, device: torch.device) -> dict:
    """
    Convert a feature dictionary to LightGlue-compatible format.

    Ensures:
        keypoints   : (B, N, 2)
        descriptors : (B, N, D)
        scores      : (B, N) (if present)
    """

    # 1. Unwrap list / tuple
    feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}

    # 2. Sanity checks
    if "keypoints" not in feats or "descriptors" not in feats:
        raise KeyError("features must contain 'keypoints' and 'descriptors'")

    kpts = feats["keypoints"]          # (N, 2)
    desc = feats["descriptors"]        # (N, D) or (D, N)

    if kpts.ndim != 2 or kpts.shape[1] != 2:
        raise ValueError(f"Invalid keypoints shape: {kpts.shape}")

    N = kpts.shape[0]

    # 3. Fix descriptor layout using keypoint count
    if desc.ndim != 2:
        raise ValueError(f"Invalid descriptors shape: {desc.shape}")

    # Case A: (D, N) → transpose
    if desc.shape[1] == N and desc.shape[0] != N:
        desc = desc.T

    # Case B: (N, D) → OK
    elif desc.shape[0] == N:
        pass

    else:
        raise ValueError(
            f"Descriptor / keypoint mismatch: "
            f"descriptors={desc.shape}, keypoints={kpts.shape}"
        )

    feats["descriptors"] = desc  # now guaranteed (N, D)

    # 4. Remove unused keys (LightGlue don't use)
    feats.pop("feature_path", None)
    feats.pop("im_path", None)

    # 5. Add batch dimension
    feats = {k: v[None] for k, v in feats.items()}

    # 6. Convert to torch.Tensor
    feats = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in feats.items()
    }

    return feats


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


class LightGlueMatcher(MatcherBase):
    _default_conf = {
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
    required_inputs = []
    min_matches = 20
    max_feat_no_tiling = 200000

    def __init__(self, config, local_features="superpoint") -> None:
        """Initializes a LightGlueMatcher"""
        self._localfeatures = local_features
        super().__init__(config)

        # load the matcher
        cfg = {**self._default_conf, **self.config.get("matcher", {})}
        self._matcher = LightGlue(self._localfeatures, **cfg).eval().to(self._device)

        if self._localfeatures == "disk":
            self.max_feat_no_tiling = 50000

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
        feats0 = featuresDict2Lightglue(feats0, self._device)
        feats1 = featuresDict2Lightglue(feats1, self._device)

        # match the features
        match_res = self._matcher({"image0": feats0, "image1": feats1})

        # remove batch dimension and convert to numpy
        mfeats0, mfeats1, matches01 = [rbd(x) for x in [feats0, feats1, match_res]]
        match_res = {
            k: v.cpu().numpy()
            for k, v in matches01.items()
            if isinstance(v, torch.Tensor)
        }

        # get matching array (indices of matched keypoints in image0 and image1)
        matches01_idx = match_res["matches"]

        return matches01_idx
