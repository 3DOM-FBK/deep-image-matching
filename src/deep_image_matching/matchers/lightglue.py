from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from deep_image_matching.config import Config
from deep_image_matching.matchers.matcher_base import FeaturesDict, MatcherBase


def featuresDict2Lightglue(feats: FeaturesDict, device: torch.device) -> dict:
    # Remove elements from list/tuple
    feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
    # Move descriptors dimension to last
    if "descriptors" in feats.keys():
        if feats["descriptors"].shape[-1] != 256:
            feats["descriptors"] = feats["descriptors"].T

    if "feature_path" in feats.keys():
        del feats["feature_path"]
    if "im_path" in feats.keys():
        del feats["im_path"]
    # if "time" in feats.keys():
    #    del feats["time"]

    # Add batch dimension
    feats = {k: v[None] for k, v in feats.items()}

    # Convert to tensor
    feats = {k: torch.tensor(v, dtype=torch.float, device=device) for k, v in feats.items()}
    # Check device
    feats = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}

    return feats


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v for k, v in data.items()}

    # hw1 = torch.tensor(image1.shape[2:], device=device)
    # hw2 = torch.tensor(image2.shape[2:], device=device)
    # lg_matcher = KF.LightGlueMatcher("dedodeb").eval().to(device)
    # with torch.inference_mode():
    #     dists, idxs = lg_matcher(
    #         descriptors1[0],
    #         descriptors2[0],
    #         KF.laf_from_center_scale_ori(keypoints1),
    #         KF.laf_from_center_scale_ori(keypoints2),
    #         hw1=hw1,
    #         hw2=hw2,
    #     )

    # mkpts1, mkpts2 = get_matching_keypoints(keypoints1[0], keypoints2[0], idxs.detach().cpu())

    # Fm, inliers = cv2.findFundamentalMat(
    #     mkpts1.detach().cpu().numpy(), mkpts2.detach().cpu().numpy(), cv2.USAC_MAGSAC, 1.5, 0.999, 100000
    # )
    # inliers = inliers > 0


class LightGlueMatcher(MatcherBase):
    _default_conf: ClassVar[Dict[str, Any]] = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
    local_features: ClassVar[List[str]] = [
        "aliked",
        "dedodeb",
        "dedodeg",
        "disk",
        "dog_affnet_hardnet",
        "doghardnet",
        "keynet_affnet_hardnet",
        "sift",
        "superpoint",
    ]

    required_inputs: ClassVar[List[str]] = ["image0", "image1"]
    min_matches = 20
    max_feat_no_tiling = 200000

    def __init__(self, config: Config, local_features="superpoint") -> None:
        """Initializes a LightGlueMatcher"""
        self._localfeatures = local_features
        super().__init__(config)

        # load the matcher
        if self.config.extractor["name"] != "dedode":
            feat_name = self.config.extractor["name"]
        else:
            if "G" in self.config.extractor["descriptor_weights"]:
                feat_name = "dedodeg"
            elif "B" in self.config.extractor["descriptor_weights"]:
                feat_name = "dedodeb"
        if not feat_name in self.local_features:
            raise ValueError(f"Feature {feat_name} not supported by LightGlueMatcher")

        self._matcher = KF.LightGlueMatcher(feature_name=feat_name).eval().to(self._device)

        if self._localfeatures == "disk":
            self.max_feat_no_tiling = 50000

    @torch.inference_mode()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
        """Match the features in the two images.

        Args:
            feats0 (dict): Features of the first image.
            feats1 (dict): Features of the second image.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.

        """

        # LightGlueMatcher.forward() expects the following arguments:
        # Args:
        #     desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        #     desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        #     lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
        #     lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        # Return:
        #     - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        #     - Long tensor indexes of matching descriptors in desc1 and desc2,
        #         shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.

        # LightGlue takes the following inputs:
        # Input (dict):
        #     image0: dict
        #         keypoints: [B x M x 2]
        #         descriptors: [B x M x D]
        #         image: [B x C x H x W] or image_size: [B x 2]
        #     image1: dict
        #         keypoints: [B x N x 2]
        #         descriptors: [B x N x D]
        #         image: [B x C x H x W] or image_size: [B x 2]
        # Output (dict):
        #     log_assignment: [B x M+1 x N+1]
        #     matches0: [B x M]
        #     matching_scores0: [B x M]
        #     matches1: [B x N]
        #     matching_scores1: [B x N]
        #     matches: List[[Si x 2]], scores: List[[Si]]
        # """

        # Convert features to LightGlue format
        # Check if keypoints outside the image
        idx = (
            (feats0["keypoints"][:, 0] >= 0)
            & (feats0["keypoints"][:, 0] < feats0["image_size"][1])
            & (feats0["keypoints"][:, 1] >= 0)
            & (feats0["keypoints"][:, 1] < feats0["image_size"][0])
        )
        if not idx.all():
            raise ValueError("Some keypoints are outside the image")

        feats0 = featuresDict2Lightglue(feats0, self._device)
        feats1 = featuresDict2Lightglue(feats1, self._device)

        # match the features
        lafs0 = KF.laf_from_center_scale_ori(feats0["keypoints"])
        lafs1 = KF.laf_from_center_scale_ori(feats1["keypoints"])
        dist, idx = self._matcher(
            feats0["descriptors"][0],
            feats1["descriptors"][0],
            lafs0,
            lafs1,
            hw1=feats0["image_size"],
            hw2=feats1["image_size"],
        )

        matches01_idx = idx.detach().cpu().numpy()

        return matches01_idx


if __name__ == "__main__":
    from pathlib import Path

    import h5py

    import deep_image_matching as dim

    SKIP_EXTRACTION = False

    params = {
        "dir": "./assets/example_cyprus",
        "pipeline": "superpoint+lightglue",
        "strategy": "bruteforce",
        "quality": "medium",
        "tiling": "none",
        "camera_options": "./assets/example_cyprus/cameras.yaml",
        "openmvg": None,
        "force": False if SKIP_EXTRACTION else True,
    }
    config = dim.Config(params)

    # Get image list
    img_list = list((Path(params["dir"]) / "images").rglob("*"))

    img0 = img_list[0]
    img1 = img_list[1]

    # Manually load the desired extractor and matcher
    # extractor = dim.extractors.DeDoDeExtractor(config)
    extractor = dim.extractors.SuperPointExtractor(config)
    matcher = LightGlueMatcher(config)

    # Extract features from the first image
    if not SKIP_EXTRACTION:
        feat_path = extractor.extract(img0)
        feat_path = extractor.extract(img1)
    else:
        feat_path = Path("./assets/example_cyprus/results_superpoint+lightglue_bruteforce_quality_medium/features.h5")
        assert feat_path.exists()

    # Open the feature file and print its contents
    with h5py.File(feat_path, "r") as f:
        for k, v in f.items():
            print(k, v)

    # match the features
    matches = matcher.match(feat_path, feat_path, img0, img1)

    print("Done")
