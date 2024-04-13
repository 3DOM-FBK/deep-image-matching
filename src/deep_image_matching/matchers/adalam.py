import numpy as np
import torch
from kornia import feature as KF

from .matcher_base import FeaturesDict, MatcherBase


# Refer to https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.DescriptorMatcher for more information
class AdalamMatcher(MatcherBase):
    _default_conf = {
        "name": "adalam",
        "match_mode": "adalam",
        "th": 0.8,
    }
    required_inputs = []
    min_matches = 20
    max_feat_no_tiling = 200000

    def __init__(self, config) -> None:
        super().__init__(config)

        # load the matcher
        cfg = {**self._default_conf, **self.config.get("adalam", {})}
        self._matcher = KF.GeometryAwareDescriptorMatcher(cfg["match_mode"], cfg["th"])

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
        if "lafs" not in feats0.keys() or "lafs" not in feats1.keys():
            raise ValueError("LAFs not found in features. Unable to match with Adalam.")
        desc1 = feats0["descriptors"].T
        desc2 = feats1["descriptors"].T
        lafs1 = feats0["lafs"]
        lafs2 = feats1["lafs"]

        desc1 = torch.tensor(desc1, dtype=torch.float).to(self._device)
        desc2 = torch.tensor(desc2, dtype=torch.float).to(self._device)
        lafs1 = torch.tensor(lafs1, dtype=torch.float).to(self._device)
        lafs2 = torch.tensor(lafs2, dtype=torch.float).to(self._device)

        # match the features
        dist, idx = self._matcher(desc1, desc2, lafs1, lafs2)

        # get matching array (indices of matched keypoints in image0 and image1)
        matches01_idx = idx.cpu().numpy()

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
