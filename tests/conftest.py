import os
from pathlib import Path

import pytest


@pytest.fixture
def data_dir():
    return Path(__file__).parents[0].parents[0] / "assets/pytest"


# @pytest.fixture
# def log_dir():
#     dirpath = tempfile.mkdtemp()
#     return dirpath


# @pytest.fixture
# def cfg_file(data_dir):
#     return data_dir / "config.yaml"


# # @pytest.fixture
# # def epoch_dict(data_dir):

if __name__ == "__main__":
    print(Path(os.path.split(__file__)[0]).parents[0] / "assets")
