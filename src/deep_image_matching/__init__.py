__version__ = "1.2.4"

import logging
from time import time
from collections import OrderedDict

time_dict = OrderedDict()
time_dict["start"] = time()

# Check if pycolmap is installed
try:
    import pycolmap

    NO_PYCOLMAP = False
except ImportError:
    logging.warning(
        "pycolmap is not installed, some advanced features may not work, but you will be able to run deep-image-matching and export the matched features in a sqlite3 database to be opened in COLMAP GUI."
    )
    NO_PYCOLMAP = True

# Import submodules
from . import extractors
from . import matchers
from . import io
from . import utils
from . import visualization
from . import thirdparty

if not NO_PYCOLMAP:
    # Import submodules that require pycolmap
    from . import reconstruction
    from . import triangulation
try:
    from . import graph
except ImportError:
    logging.warning("pyvis is not available. Unable to visualize view graph.")

# Import functions
from .parser import parse_cli

# Import classes and variables
from .image_matching import ImageMatcher
from .pairs_generator import PairsGenerator
from .constants import *
from .config import Config

print("Deep Image Matching loaded in {:.3f} seconds.".format(time() - time_dict["start"]))
