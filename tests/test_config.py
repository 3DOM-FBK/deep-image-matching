from pathlib import Path

import pytest
from deep_image_matching.config import Config

import os
from pathlib import Path

import pytest


@pytest.fixture
def data_dir():
    return Path(__file__).parents[0].parents[0] / "assets/pytest"

@pytest.fixture
def default_args():
    return {
        "gui": False,
        "dir": Path(__file__).parents[0].parents[0] / "assets/example_cyprus",
        "images": None,
        "outs": None,
        "config": "superpoint+lightglue",
        "quality": "high",
        "tiling": "none",
        "strategy": "matching_lowres",
        "pairs": None,
        "overlap": None,
        "retrieval": None,
        "db_path": None,
        "upright": False,
        "skip_reconstruction": False,
        "force": True,
        "verbose": False,
    }


def test_config_initialization(default_args):
    config = Config(default_args)
    assert isinstance(config, Config)


if __name__ == "__main__":
    pytest.main()
