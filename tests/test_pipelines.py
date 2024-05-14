import platform
import subprocess
from pathlib import Path

import deep_image_matching as dim
import pytest
import torch
import yaml

# def run_pipeline(cmd, verbose: bool = False) -> None:
#     # Run the script using subprocess
#     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = process.communicate()

#     if verbose:
#         print(stdout.decode("utf-8"))

#     # Check if the process exited successfully (return code 0)
#     assert process.returncode == 0, f"Script execution failed with error: {stderr.decode('utf-8')}"


# @pytest.fixture
# def script():
#     return (Path(__file__).parents[1] / "main.py").resolve()


@pytest.fixture
def config_file_tiling(data_dir):
    config = {"general": {"tile_size": (200, 200)}}
    config_file = Path(data_dir) / "config.yaml"
    return create_config_file(config, config_file)


def create_config_file(config: dict, path: Path) -> Path:
    def tuple_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(tuple, tuple_representer)

    with open(path, "w") as f:
        yaml.dump(config, f)
        return Path(path)


# Test matching strategies
def test_sp_lg_bruteforce(data_dir):
    prm = {"dir": data_dir, "pipeline": "superpoint+lightglue", "strategy": "bruteforce", "skip_reconstruction": True}
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_sp_lg_sequential(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_sp_lg_matching_lowres(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "matching_lowres",
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


# Test using a custom configuration file
def test_sp_lg_custom_config(data_dir):
    config = {
        "extractor": {
            "name": "superpoint",
            "max_keypoints": 1000,
        }
    }
    config_file = Path(__file__).parents[1] / "temp.yaml"
    config_file = create_config_file(config, config_file)
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "config_file": config_file,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()
    config_file.unlink()


# Test pycolmap reconstruction
def test_pycolmap(data_dir):
    if platform.system() == "Windows":
        pytest.skip("Pycolmap is not available on Windows. Please use WSL or Docker to run this test.")
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": False,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    dim.io.export_to_colmap(
        img_dir=config.general["image_dir"],
        feature_path=feature_path,
        match_path=match_path,
        database_path=config.general["output_dir"] / "database.db",
        camera_config_path=config.general["camera_options"],
    )
    model = dim.reconstruction.pycolmap_reconstruction(
        database_path=config.general["output_dir"] / "database.db",
        sfm_dir=config.general["output_dir"],
        image_dir=config.general["image_dir"],
        refine_intrinsics=False,
    )
    assert feature_path.exists()
    assert match_path.exists()
    assert model is not None


# Test Quality
def test_sp_lg_quality_medium(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


# Test tiling
def test_tiling_preselection(data_dir, config_file_tiling):
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "tiling": "preselection",
        "config_file": config_file_tiling,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()
    config_file_tiling.unlink()


def test_tiling_grid(data_dir, config_file_tiling):
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "tiling": "grid",
        "config_file": config_file_tiling,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()
    config_file_tiling.unlink()


def test_tiling_exhaustive(data_dir, config_file_tiling):
    prm = {
        "dir": data_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "tiling": "exhaustive",
        "config_file": config_file_tiling,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()
    config_file_tiling.unlink()


# Test different matching methods with sequential strategy (faster)
def test_disk_lg(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "disk+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_aliked_lg(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "aliked+lightglue",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_orb(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "orb+kornia_matcher",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_sift(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "sift+kornia_matcher",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_keynet(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "keynetaffnethardnet+kornia_matcher",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_dedode_nn(data_dir):
    if not torch.cuda.is_available():
        pytest.skip("DeDoDe is not available without CUDA GPU.")
    prm = {
        "dir": data_dir,
        "pipeline": "dedode+kornia_matcher",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


# Test semi-dense matchers
def test_loftr(data_dir):
    prm = {
        "dir": data_dir,
        "pipeline": "loftr",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_roma(data_dir):
    if not torch.cuda.is_available():
        pytest.skip("ROMA is not available without CUDA GPU.")
    prm = {
        "dir": data_dir,
        "pipeline": "roma",
        "strategy": "sequential",
        "overlap": 1,
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()


def test_roma_tiling(data_dir, config_file_tiling):
    if not torch.cuda.is_available():
        pytest.skip("ROMA is not available without CUDA GPU.")
    prm = {
        "dir": data_dir,
        "pipeline": "roma",
        "strategy": "sequential",
        "overlap": 1,
        "tiling": "preselection",
        "skip_reconstruction": True,
    }
    config = dim.Config(prm)
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()
    assert feature_path.exists()
    assert match_path.exists()
    config_file_tiling.unlink()


if __name__ == "__main__":
    pytest.main([f"{__file__}"])
