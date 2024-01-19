import os
import shutil
import tempfile
from pathlib import Path

import pytest


# Create a temporary directory with the test images for each test
@pytest.fixture
def data_dir():
    assets = (Path(__file__).parents[0].parents[0] / "assets/pytest/images").resolve()

    tempdir = tempfile.mkdtemp()
    shutil.copytree(assets, Path(tempdir) / "images")

    return tempdir


if __name__ == "__main__":
    print(Path(os.path.split(__file__)[0]).parents[0] / "assets")
