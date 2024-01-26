import shutil
import tempfile
from pathlib import Path

import pytest


# # Create a temporary directory with the test images for each test
@pytest.fixture
def data_dir(request):
    assets = (Path(__file__).parents[0].parents[0] / "assets/pytest/images").resolve()
    tempdir = tempfile.mkdtemp()
    shutil.copytree(assets, Path(tempdir) / "images")

    def cleanup():
        # Cleanup code: remove the temporary directory and its contents
        shutil.rmtree(tempdir)

    # Add finalizer to the fixture
    request.addfinalizer(cleanup)

    return tempdir
