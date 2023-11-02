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
    "tile_selection": TileSelection.NONE,
    "max_keypoints": 4096,
    "force_cpu": False,
    "output_dir": "results",
    "do_viz": False,
    "fast_viz": True,
    # "interactive_viz": False,
    "hide_matching_track": True,
    "do_viz_tiles": False,
    "tiling_grid": [1, 1],
    "tiling_overlap": 0,
    "min_matches_per_tile": 5,
}
