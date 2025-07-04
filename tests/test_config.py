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

    # Test that required sections exist
    assert hasattr(config, "general")
    assert hasattr(config, "extractor")
    assert hasattr(config, "matcher")
    assert isinstance(config.general, dict)
    assert isinstance(config.extractor, dict)
    assert isinstance(config.matcher, dict)

    # Test required general configuration keys and their types
    required_general_keys = {
        "quality": Quality,
        "tile_selection": TileSelection,
        "tile_size": tuple,
        "tile_overlap": (int, float),
        "tile_preselection_size": (int, float),
        "min_matches_per_tile": int,
        "geom_verification": GeometricVerification,
        "gv_threshold": (int, float),
        "gv_confidence": (int, float),
        "min_inliers_per_pair": int,
        "min_inlier_ratio_per_pair": (int, float),
        "try_match_full_images": bool,
    }

    for key, expected_type in required_general_keys.items():
        assert key in config.general, f"Missing required key: {key}"
        if isinstance(expected_type, tuple):
            assert isinstance(config.general[key], expected_type), (
                f"Key '{key}' should be of type {expected_type}, got {type(config.general[key])}"
            )
        else:
            assert isinstance(config.general[key], expected_type), (
                f"Key '{key}' should be of type {expected_type}, got {type(config.general[key])}"
            )

    # Test tile_size is a tuple of exactly 2 positive numbers
    tile_size = config.general["tile_size"]
    assert len(tile_size) == 2, (
        f"tile_size should have 2 elements, got {len(tile_size)}"
    )
    assert all(isinstance(x, (int, float)) and x > 0 for x in tile_size), (
        "tile_size elements should be positive numbers"
    )

    # Test value ranges for critical parameters
    assert 0 < config.general["gv_confidence"] <= 1, (
        f"gv_confidence should be in (0, 1], got {config.general['gv_confidence']}"
    )
    assert 0 <= config.general["min_inlier_ratio_per_pair"] <= 1, (
        f"min_inlier_ratio_per_pair should be in [0, 1], got {config.general['min_inlier_ratio_per_pair']}"
    )
    assert config.general["gv_threshold"] > 0, (
        f"gv_threshold should be positive, got {config.general['gv_threshold']}"
    )
    assert config.general["min_matches_per_tile"] >= 0, (
        f"min_matches_per_tile should be non-negative, got {config.general['min_matches_per_tile']}"
    )
    assert config.general["min_inliers_per_pair"] >= 0, (
        f"min_inliers_per_pair should be non-negative, got {config.general['min_inliers_per_pair']}"
    )

    # Test required extractor configuration keys and their types
    required_extractor_keys = {
        "name": str,
        "nms_radius": (int, float),
        "keypoint_threshold": (int, float),
        "max_keypoints": int,
    }

    for key, expected_type in required_extractor_keys.items():
        assert key in config.extractor, f"Missing required extractor key: {key}"
        if isinstance(expected_type, tuple):
            assert isinstance(config.extractor[key], expected_type), (
                f"Extractor key '{key}' should be of type {expected_type}, got {type(config.extractor[key])}"
            )
        else:
            assert isinstance(config.extractor[key], expected_type), (
                f"Extractor key '{key}' should be of type {expected_type}, got {type(config.extractor[key])}"
            )

    # Test extractor value ranges
    assert config.extractor["max_keypoints"] > 0, (
        f"max_keypoints should be positive, got {config.extractor['max_keypoints']}"
    )
    assert config.extractor["keypoint_threshold"] >= 0, (
        f"keypoint_threshold should be non-negative, got {config.extractor['keypoint_threshold']}"
    )
    assert config.extractor["nms_radius"] >= 0, (
        f"nms_radius should be non-negative, got {config.extractor['nms_radius']}"
    )
    assert config.extractor["name"] != "", "extractor name should not be empty"

    # Test required matcher configuration keys and their types
    required_matcher_keys = {
        "name": str,
        "n_layers": int,
        "mp": bool,
        "flash": bool,
        "depth_confidence": (int, float),
        "width_confidence": (int, float),
        "filter_threshold": (int, float),
    }

    for key, expected_type in required_matcher_keys.items():
        assert key in config.matcher, f"Missing required matcher key: {key}"
        if isinstance(expected_type, tuple):
            assert isinstance(config.matcher[key], expected_type), (
                f"Matcher key '{key}' should be of type {expected_type}, got {type(config.matcher[key])}"
            )
        else:
            assert isinstance(config.matcher[key], expected_type), (
                f"Matcher key '{key}' should be of type {expected_type}, got {type(config.matcher[key])}"
            )

    # Test matcher value ranges
    assert config.matcher["n_layers"] > 0, (
        f"n_layers should be positive, got {config.matcher['n_layers']}"
    )
    assert 0 <= config.matcher["depth_confidence"] <= 1, (
        f"depth_confidence should be in [0, 1], got {config.matcher['depth_confidence']}"
    )
    assert 0 <= config.matcher["width_confidence"] <= 1, (
        f"width_confidence should be in [0, 1], got {config.matcher['width_confidence']}"
    )
    assert config.matcher["filter_threshold"] >= 0, (
        f"filter_threshold should be non-negative, got {config.matcher['filter_threshold']}"
    )
    assert config.matcher["name"] != "", "matcher name should not be empty"

    # Test that enums have valid values
    assert config.general["quality"] in Quality, (
        f"quality should be a valid Quality enum value, got {config.general['quality']}"
    )
    assert config.general["tile_selection"] in TileSelection, (
        f"tile_selection should be a valid TileSelection enum value, got {config.general['tile_selection']}"
    )
    assert config.general["geom_verification"] in GeometricVerification, (
        f"geom_verification should be a valid GeometricVerification enum value, got {config.general['geom_verification']}"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
