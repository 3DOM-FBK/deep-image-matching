from enum import Enum
from typing import Tuple

from .utils.logger import change_logger_level, get_logger, setup_logger  # noqa: F401
from .utils.timer import Timer, timeit  # noqa: F401

__version__ = "0.0.4"

logger = setup_logger(name="deep-image-matching", log_level="info")
timer = Timer(logger=logger)


class TileSelection(Enum):
    """Enumeration for tile selection methods."""

    NONE = 0
    EXHAUSTIVE = 1
    GRID = 2
    PRESELECTION = 3


class GeometricVerification(Enum):
    """Enumeration for geometric verification methods."""

    NONE = 0
    PYDEGENSAC = 1
    MAGSAC = 2


class Quality(Enum):
    """Enumeration for matching quality."""

    LOWEST = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4


def get_size_by_quality(
    quality: Quality,
    size: Tuple[int, int],  # usually (width, height)
):
    quality_size_map = {
        Quality.HIGHEST: 2,
        Quality.HIGH: 1,
        Quality.MEDIUM: 1 / 2,
        Quality.LOW: 1 / 4,
        Quality.LOWEST: 1 / 8,
    }
    f = quality_size_map[quality]
    return (int(size[0] * f), int(size[1] * f))
