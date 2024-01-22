import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def script():
    return (Path(__file__).parents[1] / "main.py").resolve()


@pytest.fixture
def config_file():
    config = {"general": {"tile_size": (400, 400)}}

    def tuple_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(tuple, tuple_representer)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
        yaml.dump(config, file)
        return file.name


def run_pipeline(cmd, verbose: bool = False) -> None:
    # Run the script using subprocess
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if verbose:
        print(stdout.decode("utf-8"))

    # Check if the process exited successfully (return code 0)
    assert (
        process.returncode == 0
    ), f"Script execution failed with error: {stderr.decode('utf-8')}"


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
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --config_file config.yaml --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


# Test pycolmap reconstruction
def test_pycolmap(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy matching_lowres --force"
    )


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


# Test Quality
def test_sp_lg_quality_medium(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy sequential --overlap 1 --quality medium --skip_reconstruction --force"
    )


# Test tiling
def test_tiling_preselection(data_dir, script, config_file):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --tiling preselection --config {config_file} --skip_reconstruction --force",
        verbose=True,
    )


def test_tiling_grid(data_dir, script, config_file):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --tiling grid --config {config_file} --skip_reconstruction --force",
        verbose=True,
    )


def test_tiling_exhaustive(data_dir, script, config_file):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --tiling exhaustive --config {config_file} --skip_reconstruction --force",
        verbose=True,
    )


# Test semi-dense matchers
def test_loftr(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline loftr --strategy bruteforce --skip_reconstruction --force"
    )


def test_roma(data_dir, script):
    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline roma --strategy bruteforce --skip_reconstruction --force"
    )


# def test_roma_tiling(data_dir, script):
#     run_pipeline(
#         f"python {script} --dir {data_dir} --pipeline roma --strategy bruteforce --tiling preselection --skip_reconstruction --force"
#     )


if __name__ == "__main__":
    pytest.main([f"{__file__}"])
