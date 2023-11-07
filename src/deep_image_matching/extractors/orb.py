import logging

import cv2
import numpy as np

from .extractor_base import ExtractorBase

logger = logging.getLogger(__name__)


class ORBExtractor(ExtractorBase):
    default_conf = {
        "n_features": 1000,
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
    grayscale = False
    descriptor_size = 256
    detection_noise = 2.0

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self._config.get("extractor")
        self._extractor = cv2.ORB_create(
            nfeatures=cfg["n_features"],
            scaleFactor=cfg["scaleFactor"],
            nlevels=cfg["nlevels"],
            edgeThreshold=cfg["edgeThreshold"],
            firstLevel=cfg["firstLevel"],
            WTA_K=cfg["WTA_K"],
            scoreType=cfg["scoreType"],
            patchSize=cfg["patchSize"],
            fastThreshold=cfg["fastThreshold"],
        )

    def _extract(self, image: np.ndarray) -> np.ndarray:
        kp = self._extractor.detect(image, None)
        kp, des = self._extractor.compute(image, kp)
        kpts = cv2.KeyPoint_convert(kp)
        des = des.astype(float)

        # Convert tensors to numpy arrays
        feats = {"keypoints": kpts, "descriptors": des}

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        pass


if __name__ == "__main__":
    pass
