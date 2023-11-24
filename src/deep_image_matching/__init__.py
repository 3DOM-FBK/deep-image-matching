from .utils.consts import *  # noqa: F401, F403
from .utils.logger import change_logger_level, get_logger, setup_logger  # noqa: F401
from .utils.timer import Timer, timeit  # noqa: F401

__version__ = "0.0.1"

logger = setup_logger(name="deep-image-matching", log_level="info")
timer = Timer(logger=logger)
