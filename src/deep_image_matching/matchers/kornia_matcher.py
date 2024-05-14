import numpy as np
import torch
from kornia import feature as KF

from .matcher_base import FeaturesDict, MatcherBase


# Refer to https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.DescriptorMatcher for more information
class KorniaMatcher(MatcherBase):
    _default_conf = {
        "name": "kornia_matcher",
        "match_mode": "smnn",
        "th": 0.8,
    }
    required_inputs = []
    min_matches = 20
    max_feat_no_tiling = 200000

    def __init__(self, config) -> None:
        super().__init__(config)

        # load the matcher
        cfg = {**self._default_conf, **self.config.get("matcher", {})}
        self._matcher = KF.DescriptorMatcher(cfg["match_mode"], cfg["th"])

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
        # print(feats0["descriptors"][:5,:5])
        # print(feats1["descriptors"][:5,:5])
        # print(feats0["descriptors"][0,:]-feats1["descriptors"][0,:])

        desc1 = feats0["descriptors"].T
        desc2 = feats1["descriptors"].T

        desc1 = torch.tensor(desc1, dtype=torch.float).to(self._device)
        desc2 = torch.tensor(desc2, dtype=torch.float).to(self._device)

        # print(desc1.shape)
        # print(desc2.shape)

        # match the features
        dist, idx = self._matcher(desc1, desc2)

        # get matching array (indices of matched keypoints in image0 and image1)
        matches01_idx = idx.cpu().numpy()
        # print(dist)
        # print(matches01_idx)

        return matches01_idx


if __name__ == "__main__":
    pass
    # For debugging
    # from pathlib import Path

    # import cv2

    # from deep_image_matching.visualization import viz_matches_cv2

    # image1_path = Path("data/easy_small/01_Camera1.jpg")
    # image2_path = Path("data/easy_small/02_Camera1.jpg")

    # kpts1 = feats0["keypoints"]
    # kpts2 = feats1["keypoints"]
    # mkpts1 = kpts1[matches01_idx[:, 0]]
    # mkpts2 = kpts2[matches01_idx[:, 1]]

    # img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
    # img = viz_matches_cv2(img1, img2, mkpts1, mkpts2)

    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # cv2.imshow("image", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
