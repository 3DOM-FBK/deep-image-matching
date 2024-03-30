import platform
import subprocess
from pathlib import Path

import pytest
import torch
import yaml


def run_pipeline(cmd, verbose: bool = False) -> None:
    # Run the script using subprocess
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if verbose:
        print(stdout.decode("utf-8"))

    # Check if the process exited successfully (return code 0)
    assert process.returncode == 0, f"Script execution failed with error: {stderr.decode('utf-8')}"


def create_config_file(config: dict, path: Path) -> Path:
    def tuple_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(tuple, tuple_representer)

    with open(path, "w") as f:
        yaml.dump(config, f)
        return Path(path)


@pytest.fixture
def script():
    return (Path(__file__).parents[1] / "main.py").resolve()


@pytest.fixture
def config_file_tiling(data_dir):
    config = {"general": {"tile_size": (200, 200)}}
    config_file = Path(data_dir) / "config.yaml"
    return create_config_file(config, config_file)


# Test matching strategies
def test_sp_lg_bruteforce(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --skip_reconstruction --force"
    )


def test_sp_lg_sequential(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_sp_lg_matching_lowres(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy matching_lowres --skip_reconstruction --force"
    )


# Test using a custom configuration file
def test_sp_lg_custom_config(data_dir, script):
    config = {
        "extractor": {
            "name": "superpoint",
            "max_keypoints": 20000,
        }
    }
    config_file = Path(__file__).parents[1] / "temp.yaml"
    config_file = create_config_file(config, config_file)
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --config_file {config_file} --strategy sequential --overlap 1 --skip_reconstruction --force"
    )
    config_file.unlink()


# Test pycolmap reconstruction
def test_pycolmap(data_dir, script):
    if platform.system() == "Windows":
        pytest.skip("Pycolmap is not available on Windows. Please use WSL or Docker to run this test.")
    run_pipeline(f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy matching_lowres --force")


# Test different matching methods with sequential strategy (faster)
def test_disk_lg(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline disk+lightglue --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_aliked_lg(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline aliked+lightglue --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_orb(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline orb+kornia_matcher --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_sift(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline sift+kornia_matcher --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_keynet(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline keynetaffnethardnet+kornia_matcher --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_dedode_nn(data_dir, script):
    if not torch.cuda.is_available():
        pytest.skip("DeDoDe is not available without CUDA GPU.")
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline dedode+kornia_matcher --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


# Test Quality
def test_sp_lg_quality_medium(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy sequential --overlap 1 --quality medium --skip_reconstruction --force"
    )


# Test tiling
def test_tiling_preselection(data_dir, script, config_file_tiling):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --tiling preselection --config {config_file_tiling} --skip_reconstruction --force",
    )
    config_file_tiling.unlink()


def test_tiling_grid(data_dir, script, config_file_tiling):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --tiling grid --config {config_file_tiling} --skip_reconstruction --force",
    )
    config_file_tiling.unlink()


def test_tiling_exhaustive(data_dir, script, config_file_tiling):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --tiling exhaustive --config {config_file_tiling} --skip_reconstruction --force",
    )
    config_file_tiling.unlink()


# Test semi-dense matchers
def test_loftr(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline loftr --strategy bruteforce --skip_reconstruction --force"
    )


def test_roma(data_dir, script):
    if not torch.cuda.is_available():
        pytest.skip("ROMA is not available without CUDA GPU.")
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline roma --strategy bruteforce --skip_reconstruction --force"
    )


def test_roma_tiling(data_dir, script, config_file_tiling):
    if not torch.cuda.is_available():
        pytest.skip("ROMA is not available without CUDA GPU.")
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline roma --strategy bruteforce --config {config_file_tiling} --tiling preselection --skip_reconstruction --force"
    )
    config_file_tiling.unlink()


if __name__ == "__main__":
    pytest.main([f"{__file__}"])
