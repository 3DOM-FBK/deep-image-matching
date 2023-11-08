from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class Tiler:
    """
    Class for dividing an image into tiles.
    """

    def __init__(
        self,
        grid: List[int] = [1, 1],
        overlap: int = 0,
        origin: List[int] = [0, 0],
        max_length: int = 2000,
    ) -> None:
        """
        Initialize class.

        Parameters:
        - image (Image): The input image.
        - grid (List[int], default=[1, 1]): List containing the number of rows and number of columns in which to divide the image ([nrows, ncols]).
        - overlap (int, default=0): Number of pixels of overlap between adjacent tiles.
        - origin (List[int], default=[0, 0]): List of coordinates [x, y] of the pixel from which the tiling starts (top-left corner of the first tile).

        Returns:
        None
        """
        self._origin = origin
        self._overlap = overlap
        self._nrow = grid[0]
        self._ncol = grid[1]
        self._limits = None
        self._tiles = None

    @property
    def grid(self) -> List[int]:
        """
        Get the grid size.

        Returns:
        List[int]: The number of rows and number of columns in the grid.
        """
        return [self._nrow, self._ncol]

    @property
    def origin(self) -> List[int]:
        """
        Get the origin of the tiling.

        Returns:
        List[int]: The coordinates [x, y] of the pixel from which the tiling starts (top-left corner of the first tile).
        """
        return self._origin

    @property
    def overlap(self) -> int:
        """
        Get the overlap size.

        Returns:
        int: The number of pixels of overlap between adjacent tiles.
        """
        return self._overlap

    @property
    def limits(self) -> Dict[int, tuple]:
        """
        Get the tile limits.

        Returns:
        dict: A dictionary containing the index of each tile and its bounding box coordinates.
        """
        return self._limits

    def compute_grid_size(self, max_length: int) -> None:
        """
        Compute the best number of rows and columns for the grid based on the maximum length of each tile.

        Parameters:
        - max_length (int): The maximum length of each tile.

        Returns:
        None

        NOTE: NOT WORKING. NEEDS TO BE TESTED.
        """
        self._nrow = int(np.ceil(self._h / (max_length - self._overlap)))
        self._ncol = int(np.ceil(self._w / (max_length - self._overlap)))

    def compute_limits_by_grid(self, image: np.ndarray) -> List[int]:
        """
        Compute the limits of each tile (i.e., xmin, ymin, xmax, ymax) given the number of rows and columns in the tile grid.

        Returns:
        List[int]: A list containing the bounding box coordinates of each tile as: [xmin, ymin, xmax, ymax]
        List[int]: The coordinates [x, y] of the pixel from which the tiling starts (top-left corner of the first tile).
        """

        self._image = image
        self._w = image.shape[1]
        self._h = image.shape[0]

        DX = round((self._w - self._origin[0]) / self._ncol / 10) * 10
        DY = round((self._h - self._origin[1]) / self._nrow / 10) * 10

        self._limits = {}
        for col in range(self._ncol):
            for row in range(self._nrow):
                tile_idx = np.ravel_multi_index(
                    (row, col), (self._nrow, self._ncol), order="C"
                )
                xmin = max(self._origin[0], col * DX - self._overlap)
                ymin = max(self._origin[1], row * DY - self._overlap)
                xmax = xmin + DX + self._overlap - 1
                ymax = ymin + DY + self._overlap - 1
                self._limits[tile_idx] = (xmin, ymin, xmax, ymax)

        return self._limits, self._origin

    def extract_patch(self, image: np.ndarray, limits: List[int]) -> np.ndarray:
        """Extract image patch
        Parameters
        __________
        - limits (List[int]): List containing the bounding box coordinates as: [xmin, ymin, xmax, ymax]
        __________
        Return: patch (np.ndarray)
        """
        patch = image[
            limits[1] : limits[3],
            limits[0] : limits[2],
        ]
        return patch

    def read_all_tiles(self) -> None:
        """
        Read all tiles and store them in the class instance.

        Returns:
        None
        """
        self._tiles = {}
        for idx, limit in self._limits.items():
            self._tiles[idx] = self.extract_patch(self._image, limit)

    def read_tile(self, idx) -> np.ndarray:
        """
        Extract and return a tile given its index.

        Parameters:
        - idx (int): The index of the tile.

        Returns:
        np.ndarray: The extracted tile.
        """
        if self._tiles is None:
            self._tiles = {}
        return self.extract_patch(self._image, self._limits[idx])

    def remove_tiles(self, tile_idx=None) -> None:
        """
        Remove tiles from the class instance.

        Parameters:
        - tile_idx: The index of the tile to be removed. If None, remove all tiles.

        Returns:
        None
        """
        if tile_idx is None:
            self._tiles = {}
        else:
            self._tiles[tile_idx] = []

    def display_tiles(self) -> None:
        """
        Display all the stored tiles.

        Returns:
        None
        """
        for idx, tile in self._tiles.items():
            plt.subplot(self.grid[0], self.grid[1], idx + 1)
            plt.imshow(tile)
        plt.show()
