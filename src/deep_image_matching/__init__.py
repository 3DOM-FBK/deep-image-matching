__version__ = "1.1.0"

import logging

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
from . import io
from . import utils
from . import reconstruction
from . import extractors
from . import matchers

if not NO_PYCOLMAP:
    from . import triangulation  # the triangulation module strictly requires pycolmap

# Import functions
from .parser import parse_cli

# Import classes and variables
from .constants import *
from .image_matching import ImageMatcher
from .config import Config
from .pairs_generator import PairsGenerator
