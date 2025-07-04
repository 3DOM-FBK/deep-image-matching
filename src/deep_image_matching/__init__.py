__version__ = "1.2.4"

import logging
from collections import OrderedDict
from time import time

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
from . import extractors, io, matchers, thirdparty, utils, visualization

if not NO_PYCOLMAP:
    # Import submodules that require pycolmap
    from . import reconstruction, triangulation
try:
    from . import graph
except ImportError:
    logging.warning("pyvis is not available. Unable to visualize view graph.")

# Import functions
from .config import Config
from .constants import *

# Import classes and variables
from .image_matching import ImageMatcher
from .pairs_generator import PairsGenerator
from .parser import parse_cli

print(
    "Deep Image Matching loaded in {:.3f} seconds.".format(time() - time_dict["start"])
)
