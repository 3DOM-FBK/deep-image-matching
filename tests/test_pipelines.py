import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def data_dir():
    return Path(__file__).parents[0].parents[0] / "assets/pytest"


@pytest.fixture
def main():
    return Path(__file__).parents[1] / "main.py"


def run_pipeline(cmd) -> None:
    # Run the script using subprocess
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    # Check if the process exited successfully (return code 0)
    assert (
        process.returncode == 0
    ), f"Script execution failed with error: {stderr.decode('utf-8')}"


# Test matching strategies
def test_sp_lg_bruteforce(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config superpoint+lightglue --strategy bruteforce --skip_reconstruction --force"
    )


def test_sp_lg_sequential(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config superpoint+lightglue --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_sp_lg_matching_lowres(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config superpoint+lightglue --strategy matching_lowres --skip_reconstruction --force"
    )


# Test pycolmap reconstruction
def test_pycolmap(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config superpoint+lightglue --strategy matching_lowres --force"
    )


# Test different matching methods with sequential strategy (faster)
def test_disk_lg(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config disk+lightglue --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_aliked_lg(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config aliked+lightglue --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_orb(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config orb+kornia_matcher --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_sift(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config sift+kornia_matcher --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


def test_keynet(data_dir, main):
    run_pipeline(
        f"python {main} --dir {data_dir} --config keynetaffnethardnet+kornia_matcher --strategy sequential --overlap 1 --skip_reconstruction --force"
    )


if __name__ == "__main__":
    pytest.main()
