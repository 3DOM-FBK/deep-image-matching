import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def script():
    return (Path(__file__).parents[1] / "main.py").resolve()


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


@pytest.fixture
def config_file():
    config = {"general": {"tile_size": (400, 400), "tile_overlap": 10}}
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(yaml.dump(config))
        return f.name


# Test matching strategies
def test_tiling(script):
    def tuple_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(tuple, tuple_representer)
    data_dir = Path(__file__).parents[1] / "sandbox" / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    config = {"general": {"tile_size": (400, 400), "tile_overlap": 10}}
    with open(data_dir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    run_pipeline(
        f"python {script} --dir {data_dir} --pipeline superpoint+lightglue --strategy bruteforce --tiling preselection --config {config_file} --skip_reconstruction --force"
    )


if __name__ == "__main__":
    pytest.main([f"{__file__}"])
