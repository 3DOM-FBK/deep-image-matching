import logging

import cv2
import numpy as np

from .extractor_base import ExtractorBase

logger = logging.getLogger(__name__)


class ORBExtractor(ExtractorBase):
    default_conf = {
        "scaleFactor": 1.2,
        "nlevels": 1,
        "edgeThreshold": 1,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": 0,
        "patchSize": 31,
        "fastThreshold": 0,
    }
    required_inputs = []
    grayscale = True
    descriptor_size = 256
    detection_noise = 2.0

    def __init__(self, **config: dict):
        # Init the base class
        # super().__init__(**config)

        # TODO: improve configuration management!
        self._config = {**self.default_conf, **self._config.get("ORB", {})}

        # Load extractor
        self._extractor = cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.orb_cfg["scaleFactor"],
            nlevels=self.orb_cfg["nlevels"],
            edgeThreshold=self.orb_cfg["edgeThreshold"],
            firstLevel=self.orb_cfg["firstLevel"],
            WTA_K=self.orb_cfg["WTA_K"],
            scoreType=self.orb_cfg["scoreType"],
            patchSize=self.orb_cfg["patchSize"],
            fastThreshold=self.orb_cfg["fastThreshold"],
        )

    def _extract(self, image: np.ndarray) -> np.ndarray:
        kp = self._extractor.detect(image, None)
        kp, des = self._extractor.compute(image, kp)
        kpts = cv2.KeyPoint_convert(kp)
        des = des.astype(float)

        # Convert tensors to numpy arrays
        feats = {"keypoints": kpts, "descriptors": des}

        return feats


if __name__ == "__main__":
    pass
