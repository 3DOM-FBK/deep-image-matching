from pathlib import Path

import h5py
import pytest
import torch
from deep_image_matching import Config
from deep_image_matching.extractors import SuperPointExtractor


def test_superpoint(data_dir):
    prm = {"dir": data_dir, "pipeline": "superpoint+lightglue", "strategy": "bruteforce", "skip_reconstruction": True}
    config = Config(prm)
    config.extractor["keypoint_threshold"] = 0.01
    config.extractor["max_keypoints"] = 1000

    img_list = list(config.general["image_dir"].rglob("*"))
    fname = img_list[0]
    extractor = SuperPointExtractor(config)
    with torch.inference_mode():
        feat_path = extractor.extract(fname)

    assert Path(feat_path).exists()
    with h5py.File(feat_path, "r") as f:
        # check that the key of the image is present
        assert fname.name in f.keys()
        # Check that the keypoint and descriptor arrays are present
        assert "keypoints" in f[fname.name]
        assert "descriptors" in f[fname.name]
        # Check that the keypoint and descriptor arrays have the correct shapes
        assert f[fname.name]["keypoints"].shape[1] == 2
        assert f[fname.name]["descriptors"].shape[0] == 256

    # Test extraction with no keypoints
    config.extractor["keypoint_threshold"] = 1.0
    with torch.inference_mode():
        feat_path = extractor.extract(fname)

    assert Path(feat_path).exists()
    with h5py.File(feat_path, "r") as f:
        # check that the key of the image is present
        assert fname.name in f.keys()
        # Check that the keypoint and descriptor arrays are present
        assert "keypoints" in f[fname.name]
        assert "descriptors" in f[fname.name]
        # Check that the keypoint and descriptor arrays have the correct shapes
        assert f[fname.name]["keypoints"].shape[1] == 2
        assert f[fname.name]["descriptors"].shape[0] == 256
