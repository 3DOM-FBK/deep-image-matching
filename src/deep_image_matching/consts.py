from enum import Enum


class TileSelection(Enum):
    """Enumeration for tile selection methods."""

    NONE = 0
    EXHAUSTIVE = 1
    GRID = 2
    PRESELECTION = 3


class GeometricVerification(Enum):
    """Enumeration for geometric verification methods."""

    NONE = 1
    PYDEGENSAC = 2
    MAGSAC = 3


class Quality(Enum):
    """Enumeration for matching quality."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4
