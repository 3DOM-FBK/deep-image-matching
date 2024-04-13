import inspect
from enum import Enum
from typing import Tuple

from .utils import Timer, setup_logger

logger = setup_logger(name="dim", log_level="info")
timer = Timer(logger=logger)


# def get_extractor_classes(root):
#     classes = inspect.getmembers(root, inspect.isclass)
#     classes = [c[0] for c in classes if issubclass(c[1], root.ExtractorBase)]
#     return classes


# def get_matcher_classes(root):
#     classes = inspect.getmembers(root, inspect.isclass)
#     classes = [c[0] for c in classes if issubclass(c[1], root.MatcherBase)]
#     return classes


class Pipeline(Enum):
    """Enumeration for pipeline approaches."""

    SUPERPOINT_LIGHTGLUE = 0
    SUPERPOINT_SUPERGLUE = 1
    DISK_LIGHTGLUE = 2
    ALIKED_LIGHTGLUE = 3
    ORB_KORNIA_MATCHER = 4
    SIFT_KORNIA_MATCHER = 5
    LOFTR = 6
    SE2LOFTR = 7
    ROMA = 8
    KEYNETAFFNETHARDNET_KORNIA_MATCHER = 9
    DEDODE_KORNIA_MATCHER = 10


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
    RANSAC = 3
    LMEDS = 4
    RHO = 5
    USAC_DEFAULT = 6
    USAC_PARALLEL = 7
    USAC_FM_8PTS = 8
    USAC_FAST = 9
    USAC_ACCURATE = 10
    USAC_PROSAC = 11
    USAC_MAGSAC = 12


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
