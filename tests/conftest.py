import gc
import shutil
import tempfile
import time
from pathlib import Path

import pytest


# # Create a temporary directory with the test images for each test
@pytest.fixture
def data_dir(request):
    assets = (Path(__file__).parents[0].parents[0] / "assets/pytest/images").resolve()
    tempdir = tempfile.mkdtemp()
    shutil.copytree(assets, Path(tempdir) / "images")

    def cleanup():
        # Force garbage collection and add delay for Windows
        gc.collect()
        time.sleep(0.5)  # Give Windows time to release file handles

        # Cleanup code: remove the temporary directory and its contents
        try:
            shutil.rmtree(tempdir)
        except PermissionError:
            # On Windows, try again after a longer delay
            time.sleep(2.0)
            try:
                shutil.rmtree(tempdir)
            except PermissionError as e:
                # Log the error but don't fail the test
                print(f"Warning: Could not clean up temp directory {tempdir}: {e}")

    # Add finalizer to the fixture
    request.addfinalizer(cleanup)

    return tempdir
