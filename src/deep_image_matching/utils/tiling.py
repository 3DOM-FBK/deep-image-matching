from enum import Enum
from typing import Dict, List, Tuple, Union

import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch

# TODO: add possibility to specify the number of rows and columns in the grid
# TODO: add auto tiling mode
# TODO: add possibility to export tensors directly


class TilingMode(Enum):
    AUTO = 0
    SIZE = 1
    GRID = 2


class Tiler:
    """
    Class for dividing an image into tiles.
    """

    def __init__(
        self,
        tiling_mode=TilingMode.SIZE,
    ) -> None:
        """
        Initialize class.

        Parameters:
        - tiling_mode (TilingMode or str, default=TilingMode.SIZE): The tiling mode to use. Can be a TilingMode enum or a string with the name of the enum.

        Returns:
        None
        """
        if isinstance(tiling_mode, str):
            tiling_mode = TilingMode[tiling_mode.upper()]
        elif not isinstance(tiling_mode, TilingMode):
            raise TypeError(
                "tiling_mode must be a TilingMode enum or a string with the name of the enum"
            )
        self._tiling_mode = tiling_mode

    def compute_tiles_by_size(
        self,
        input: Union[np.ndarray, torch.Tensor],
        window_size: Union[int, Tuple[int, int]],
        overlap: Union[int, Tuple[int, int]] = 0,
    ) -> Tuple[
        Dict[int, np.ndarray], Dict[int, Tuple[int, int]], Tuple[int, int, int, int]
    ]:
        """
        Compute tiles by specifying the window size and overlap.

        Parameters:
        - input (np.ndarray or torch.Tensor): The input image.
        - window_size (int or Tuple[int, int]): The size of each tile. If int, the same size is used for both height and width. If Tuple[int, int], the first element represents the x coordinate (horizontal) and the second element represents the y coordinate (vertical).
        - overlap (int or Tuple[int, int], default=0): The overlap between adjacent tiles. If int, the same overlap is used for both height and width. If Tuple[int, int], the first element represents the overlap in the horizontal direction and the second element represents the overlap in the vertical direction.

        Returns:
        Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, int]]]: A tuple containing two dictionaries. The first dictionary contains the extracted tiles, where the key is the index of the tile and the value is the tile itself. The second dictionary contains the x, y coordinates of the top-left corner of each tile in the original image (before padding), where the key is the index of the tile and the value is a tuple of two integers representing the x and y coordinates.

        Raises:
        - TypeError: If the input is not a numpy array or a torch tensor.
        - TypeError: If the window_size is not an integer or a tuple of integers.
        - TypeError: If the overlap is not an integer or a tuple of integers.

        Note:
        - If the input is a numpy array, it is assumed to be in the format (H, W, C). If C > 1, it is converted to (C, H, W).
        - The output tiles are in the format (H, W, C).
        - The output origins are expressed in x, y coordinates, where x is the horizontal axis and y is the vertical axis (pointing down, as in OpenCV).
        """
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        elif isinstance(window_size, tuple) or isinstance(window_size, List):
            # transpose to be (H, W)
            window_size = (window_size[1], window_size[0])
        else:
            raise TypeError("window_size must be an integer or a tuple of integers")

        if isinstance(overlap, int):
            overlap = (overlap, overlap)
        elif not isinstance(overlap, tuple) or isinstance(window_size, List):
            raise TypeError("overlap must be an integer or a tuple of integers")
        overlap = overlap

        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
            # If input is a numpy array, it is assumed to be in the format (H, W, C). If C>1, it is converted to (C, H, W)
            if input.dim() > 2:
                input = input.permute(2, 0, 1)

        # Add dimensions to the tensor to be (B, C, H, W)
        if input.dim() == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        if input.dim() == 3:
            input = input.unsqueeze(0)

        H, W = input.shape[2:]

        # Compute padding to make the image divisible by the window size
        # This returns a tuple of 4 int (top, bottom, left, right)
        padding = K.contrib.compute_padding((H, W), window_size)

        stride = [w - o for w, o in zip(window_size, overlap)]
        patches = K.contrib.extract_tensor_patches(
            input, window_size, stride=stride, padding=padding
        )

        # Remove batch dimension
        patches = patches.squeeze(0)

        # compute x,y coordinates of the top-left corner of each tile in the original image (before padding)
        origins = {}
        n_rows = (H + padding[0] + padding[1] - window_size[0]) // stride[0] + 1
        n_cols = (W + padding[2] + padding[3] - window_size[1]) // stride[1] + 1
        for row in range(n_rows):
            for col in range(n_cols):
                tile_idx = np.ravel_multi_index((row, col), (n_rows, n_cols), order="C")
                x = -padding[2] + col * stride[1]
                y = -padding[0] + row * stride[0]
                origins[tile_idx] = (x, y)

        # Convert patches to numpy array (H, W, C)
        patches = patches.permute(0, 2, 3, 1).numpy()

        # arrange patches in a dictionary with the index of the patch as key
        patches = {i: patches[i] for i in range(patches.shape[0])}

        return patches, origins, padding


class Tiler_old:
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
        - grid (List[int], default=[1, 1]): List containing the number of rows and number of columns in which to divide the image ([nrows, ncols]).
        - overlap (int, default=0): Number of pixels of overlap between adjacent tiles.
        - origin (List[int], default=[0, 0]): List of coordinates [x, y] of the pixel from which the tiling starts (top-left corner of the first tile).
        - max_length (int, default=2000): The maximum length of each tile.

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

    def compute_limits_by_grid(
        self, image: np.ndarray
    ) -> Tuple[Dict[int, tuple], Tuple[int, int]]:
        """
        Compute the limits of each tile (i.e., xmin, ymin, xmax, ymax) given the number of rows and columns in the tile grid.

        Parameters:
        - image (np.ndarray): The input image.

        Returns:
        Tuple[Dict[int, tuple], Tuple[int, int]]: A tuple containing two elements. The first element is a dictionary containing the index of each tile and its bounding box coordinates. The second element is a tuple containing the x, y coordinates of the pixel from which the tiling starts (top-left corner of the first tile).

        Note:
        - The input image should have shape (H, W, C).
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
        """
        Extract an image patch given the bounding box coordinates.

        Parameters:
        - image (np.ndarray): The input image.
        - limits (List[int]): The bounding box coordinates as [xmin, ymin, xmax, ymax].

        Returns:
        np.ndarray: The extracted image patch.
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


if __name__ == "__main__":
    from pathlib import Path

    import cv2

    c, w, h = 1, 10, 8
    img = torch.arange(0, h * w).reshape(1, h, w).float()
    img = img.repeat(1, c, 1, 1)

    tile_size = (4, 5)
    overlap = 2

    tiler = Tiler()
    tiles, origins, padding = tiler.compute_tiles_by_size(
        input=img, window_size=tile_size, overlap=overlap
    )
    print(origins)

    img_path = Path("data/belv_lingua_easy/DJI_20220728115852_0003.JPG")
    img = cv2.imread(str(img_path))

    tile_size = (2048, 2730)
    overlap = 50
    tiler = Tiler()
    tiles, origins, padding = tiler.compute_tiles_by_size(
        input=img, window_size=tile_size, overlap=overlap
    )

    print("done")
