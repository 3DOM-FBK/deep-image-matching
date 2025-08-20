import cv2
import numpy as np

from .extractor_base import ExtractorBase, FeaturesDict


class ORBExtractor(ExtractorBase):
    _default_conf = {
        "name:": "orb",
        "n_features": 4000,
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
    as_float = False
    descriptor_size = 32

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self.config.get("extractor")
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
        if kp:
            kpts = cv2.KeyPoint_convert(kp)
            des = des.astype(float).T
        else:
            kpts = np.array([], dtype=np.float32).reshape(0, 2)
            des = np.array([], dtype=np.float32).reshape(
                self.descriptor_size,
                0,
            )

        # Convert tensors to numpy arrays
        feats = FeaturesDict(keypoints=kpts, descriptors=des)

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        pass


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    from deep_image_matching import GeometricVerification, Quality, TileSelection
    from deep_image_matching.io.h5 import get_features

    image_path = Path("data/easy_small/01_Camera1.jpg")
    cfg = {
        "general": {
            "quality": Quality.HIGH,
            "tile_selection": TileSelection.PRESELECTION,
            "tile_grid": [3, 3],
            "tile_overlap": 50,
            "geom_verification": GeometricVerification.PYDEGENSAC,
            "output_dir": "sandbox",
        },
        "extractor": {"name": "orb"},
    }
    pprint(cfg)

    extractor = ORBExtractor(cfg)
    feats_path = extractor.extract(image_path)

    features = get_features(feats_path, image_path.name)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    kp = cv2.KeyPoint_convert(features["keypoints"])
    img = cv2.drawKeypoints(img, kp, img, color=(0, 0, 255), flags=0)

    window = cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("done")
