__version__ = "1.3.0"

import importlib
import logging
from collections import OrderedDict
from time import time

time_dict = OrderedDict()
time_dict["start"] = time()

try:
    importlib.import_module("pycolmap")
    NO_PYCOLMAP = False
except ImportError:
    logging.warning(
        "pycolmap is not installed. Some functionalities will be unavailable. "
        "Use the `skip-reconstruction` parameter to run DIM without reconstruction. "
        "For installing pycolmap, follow the instructions at https://colmap.github.io/pycolmap/index.html."
    )
    NO_PYCOLMAP = True

# Import submodules
from . import (
    extractors,
    graph,
    io,
    matchers,
    thirdparty,
    utils,
    visualization,
)

if not NO_PYCOLMAP:
    # If pycolmap is available, import reconstruction and triangulation module
    from . import reconstruction, triangulation

# Import Config class and constants
from .config import Config
from .constants import *

# Import classes and variables
from .image_matching import ImageMatcher
from .pairs_generator import PairsGenerator
from .parser import parse_cli

print(
    "Deep Image Matching loaded in {:.3f} seconds.".format(time() - time_dict["start"])
)
