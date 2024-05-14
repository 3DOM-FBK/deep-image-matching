import logging
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

logger = logging.getLogger("dim")

try:
    import pycolmap
except ImportError:
    print("pycolmap not installed. Please install it to use this extractor.")
    exit(1)

from deep_image_matching.extractors import ExtractorBase, FeaturesDict


class SIFTpycolmapExtractor(ExtractorBase):
    # Pycolmap SIFT extractor options default values
    # print(pycolmap.SiftExtractionOptions().summary())
    _default_conf = {
        "name": "sift_pycolmap",
        "num_threads": -1,
        "max_image_size": 3200,
        "max_num_features": 8192,
        "first_octave": -1,
        "num_octaves": 4,
        "octave_resolution": 3,
        "peak_threshold": 0.006666666666666667,
        "edge_threshold": 10.0,
        "estimate_affine_shape": False,
        "max_num_orientations": 2,
        "upright": False,
        "darkness_adaptivity": False,
        "domain_size_pooling": False,
        "dsp_min_scale": 0.16666666666666666,
        "dsp_max_scale": 3.0,
        "dsp_num_scales": 10,
        "normalization": pycolmap.Normalization.L1_ROOT,
    }
    _pycolmap_cfg_keys = [
        "num_threads",
        "max_image_size",
        "max_num_features",
        "first_octave",
        "num_octaves",
        "octave_resolution",
        "peak_threshold",
        "edge_threshold",
        "estimate_affine_shape",
        "max_num_orientations",
        "upright",
        "darkness_adaptivity",
        "domain_size_pooling",
        "dsp_min_scale",
        "dsp_max_scale",
        "dsp_num_scales",
        "normalization",
    ]

    required_inputs = []
    grayscale = True
    as_float = False
    descriptor_size = 128
    features_as_half = False

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Keep only the configuation that are in the pycolmap options
        # This is to avoid passing invalid options to the pycolmap extractor that will raise an error
        cfg = self.config.extractor
        cfg = {k: cfg[k] for k in cfg if k in self._pycolmap_cfg_keys}

        # Load extractor
        # Optional parameters:
        # - options: dict or pycolmap.SiftExtractionOptions
        # - device: default pycolmap.Device.auto uses the GPU if available
        self._extractr = pycolmap.Sift(cfg, device=pycolmap.Device.auto)

    def _extract(self, image: Union[np.ndarray, torch.Tensor]) -> dict:
        # Input should be HxW float array grayscale image with range [0, 1].
        image = Image.fromarray(image)
        image = ImageOps.grayscale(image)
        image = np.array(image).astype(np.float32) / 255.0

        # Extract keypoints and descriptors
        # Returns:
        # - keypoints: Nx4 array; format: x (j), y (i), scale, orientation
        # - descriptors: Nx128 array; L2-normalized descriptors
        kpts, des = self._extractr.extract(image)

        # Convert tensors to numpy arrays
        # NOTE: only the keypoints coordinates are stored in the features dict
        # TODO: store the scale and orientation and use them for the matching
        feats = FeaturesDict(keypoints=kpts[:, :2], descriptors=des.T)

        return feats

    def _preprocess_input(self, image: np.ndarray, device: str = "cuda"):
        pass


if __name__ == "__main__":
    from pathlib import Path

    import h5py
    import numpy as np
    import pycolmap
    from PIL import ImageOps

    import deep_image_matching as dim

    params = {
        "dir": "./assets/example_cyprus",
        "pipeline": "superpoint+lightglue",
        "strategy": "bruteforce",
        "quality": "high",
        "tiling": "none",
        "camera_options": "./assets/example_cyprus/cameras.yaml",
        "openmvg": None,
        "force": True,
    }
    config = dim.Config(params)

    # Get image list
    img_list = list((Path(params["dir"]) / "images").rglob("*"))
    img_name = img_list[0]

    # Extractor
    extractor = SIFTpycolmapExtractor(config)

    feat_path = extractor.extract(img_name)

    # Open the feature file and print its contents
    with h5py.File(feat_path, "r") as f:
        for k, v in f.items():
            print(k, v)

    f = h5py.File(feat_path, "r")
    f_img = f[img_name.name]
    f_kpts = f_img["keypoints"].__array__()
    f_des = f_img["descriptors"].__array__()

    print("Done")
