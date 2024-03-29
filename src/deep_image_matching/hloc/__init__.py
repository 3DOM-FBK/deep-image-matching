import logging

__version__ = "1.5"

try:
    import pycolmap
except ImportError:
    logging.warning("pycolmap is not installed, some features may not work.")
