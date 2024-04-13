from pathlib import Path

import pytest
import yaml
from deep_image_matching import Config, GeometricVerification, Quality, TileSelection


def create_config_file(config: dict, path: Path) -> Path:
    def tuple_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(tuple, tuple_representer)

    with open(path, "w") as f:
        yaml.dump(config, f)
        return Path(path)


# Config object is created successfully with valid input arguments
def test_valid_basic_arguments(data_dir):
    cfg = {
        "extractor": {
            "name": "superpoint",
            "max_keypoints": 20000,
        }
    }
    config_file = create_config_file(cfg, Path(data_dir) / "temp.yaml")

    args = {
        "gui": False,
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "config_file": config_file,
        "quality": "high",
        "tiling": "preselection",
        "strategy": "matching_lowres",
        "upright": False,
        "skip_reconstruction": False,
        "force": True,
        "verbose": False,
        "openmvg": None,
        "camera_options": None,
    }
    config = Config(args)

    assert isinstance(config, Config)

    # Check that config dictionary contains at least the specified keys with the correct values
    expected_general = {
        "quality": Quality.HIGH,
        "tile_selection": TileSelection.PRESELECTION,
        "tile_size": (2400, 2000),
        "tile_overlap": 10,
        "tile_preselection_size": 1000,
        "min_matches_per_tile": 10,
        "geom_verification": GeometricVerification.PYDEGENSAC,
        "gv_threshold": 4,
        "gv_confidence": 0.99999,
        "min_inliers_per_pair": 15,
        "min_inlier_ratio_per_pair": 0.25,
        "try_match_full_images": False,
    }
    # Check that the config object contains the expected keys
    assert all(key in config.general for key in expected_general)

    # Check that the default values are as expected
    assert all(config.general[key] == expected_general[key] for key in expected_general)

    # Check the extractor parameters
    expected_extractor = {
        "name": "superpoint",
        "nms_radius": 3,
        "keypoint_threshold": 0.0005,
        "max_keypoints": 20000,
    }
    assert all(
        key in config.extractor and config.extractor[key] == expected_extractor[key] for key in expected_extractor
    )

    # Check the matcher parameters
    expected_matcher = {
        "name": "lightglue",
        "n_layers": 9,
        "mp": False,
        "flash": True,
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "filter_threshold": 0.1,
    }
    assert all(key in config.matcher and config.matcher[key] == expected_matcher[key] for key in expected_matcher)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
