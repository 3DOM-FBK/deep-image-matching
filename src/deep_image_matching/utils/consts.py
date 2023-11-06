from enum import Enum


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

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4


def_cfg_general = {
    "max_keypoints": 4096,
    "tile_selection": TileSelection.NONE,
    "tiling_grid": [1, 1],
    "output_dir": "results",
    "tiling_overlap": 0,
    "force_cpu": False,
    "do_viz": False,
    "fast_viz": True,
    "hide_matching_track": True,
    "do_viz_tiles": False,
}
