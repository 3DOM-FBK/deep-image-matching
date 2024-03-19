from enum import Enum
from typing import Dict, List, Tuple, Union

import kornia as K
import numpy as np
import torch


def konria_071(base_version: str = "0.7.1"):
    try:
        from packaging import version
    except ImportError:
        return False
    return version.parse(K.__version__) == version.parse(base_version)


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

    def compute_tiles(self, input: Union[np.ndarray, torch.Tensor], **kwargs):
        if self._tiling_mode == TilingMode.SIZE:
            return self.compute_tiles_by_size(input=input, **kwargs)
        elif self._tiling_mode == TilingMode.GRID:
            return self.compute_tiles_by_grid(input=input, **kwargs)
        else:
            return self.compute_tiles_auto(input=input, **kwargs)

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
            input (np.ndarray or torch.Tensor): The input image.
            window_size (int or Tuple[int, int]): The size of each tile. If int, the same size is used for both height and width. If Tuple[int, int], the first element represents the x coordinate (horizontal) and the second element represents the y coordinate (vertical).
            overlap (int or Tuple[int, int], default=0): The overlap between adjacent tiles. If int, the same overlap is used for both height and width. If Tuple[int, int], the first element represents the overlap in the horizontal direction and the second element represents the overlap in the vertical direction.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, int]]]: A tuple containing two dictionaries. The first dictionary contains the extracted tiles, where the key is the index of the tile and the value is the tile itself. The second dictionary contains the x, y coordinates of the top-left corner of each tile in the original image (before padding), where the key is the index of the tile and the value is a tuple of two integers representing the x and y coordinates.

        Raises:
            TypeError: If the input is not a numpy array or a torch tensor.
            TypeError: If the window_size is not an integer or a tuple of integers.
            TypeError: If the overlap is not an integer or a tuple of integers.

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
        elif isinstance(overlap, tuple) or isinstance(window_size, List):
            # transpose to be (H, W)
            overlap = (overlap[1], overlap[0])
        elif not isinstance(overlap, tuple) or isinstance(window_size, List):
            raise TypeError("overlap must be an integer or a tuple of integers")

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

        # Compute padding to make the image divisible by the window size.
        # This returns a tuple of 2 int (vertical, horizontal)
        # NOTE: from version 0.7.1 compute_padding() returns a tuple of 2 int and not 4 ints (top, bottom, left, right) anymore.
        padding = K.contrib.compute_padding((H, W), window_size)
        stride = [w - o for w, o in zip(window_size, overlap)]
        patches = K.contrib.extract_tensor_patches(
            input, window_size, stride=stride, padding=padding
        )

        # Remove batch dimension
        patches = patches.squeeze(0)

        # Compute number of rows and columns
        if konria_071():
            n_rows = (H + 2 * padding[0] - window_size[0]) // stride[0] + 1
            n_cols = (W + 2 * padding[1] - window_size[1]) // stride[1] + 1
        else:
            n_rows = (H + padding[0] + padding[1] - window_size[0]) // stride[0] + 1
            n_cols = (W + padding[2] + padding[3] - window_size[1]) // stride[1] + 1

        # compute x,y coordinates of the top-left corner of each tile in the original image (before padding)
        origins = {}
        for row in range(n_rows):
            for col in range(n_cols):
                tile_idx = np.ravel_multi_index((row, col), (n_rows, n_cols), order="C")
                if konria_071():
                    x = -padding[1] + col * stride[1]
                    y = -padding[0] + row * stride[0]
                else:
                    x = -padding[2] + col * stride[1]
                    y = -padding[0] + row * stride[0]
                origins[tile_idx] = (x, y)

        # Convert patches to numpy array (H, W, C)
        patches = patches.permute(0, 2, 3, 1).numpy()

        # arrange patches in a dictionary with the index of the patch as key
        patches = {i: patches[i] for i in range(patches.shape[0])}

        return patches, origins, padding

    def compute_tiles_by_grid(
        self,
        input: Union[np.ndarray, torch.Tensor],
        grid: List[int] = [1, 1],
        overlap: int = 0,
        origin: List[int] = [0, 0],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, int]]]:
        raise NotImplementedError(
            "compute_tiles_by_grid is not fully implemented yet (need to add padding and testing.)"
        )

        if not isinstance(grid, list) or len(grid) != 2:
            raise TypeError("grid must be a list of two integers")

        if not isinstance(input, np.ndarray):
            raise TypeError(
                "input must be a numpy array. Tile selection by grid is not implemented for torch tensors yet."
            )

        H, W = input.shape[:2]
        n_rows = grid[0]
        n_cols = grid[1]

        DX = round(W / n_cols / 10) * 10
        DY = round(H / n_rows / 10) * 10

        origins = {}
        for col in range(n_cols):
            for row in range(n_rows):
                tile_idx = np.ravel_multi_index((row, col), (n_rows, n_cols), order="C")
                xmin = col * DX - overlap
                ymin = row * DY - overlap
                origins[tile_idx] = (xmin, ymin)

        patches = {}
        for idx, origin in origins.items():
            xmin, ymin = origin
            xmax = xmin + DX + overlap - 1
            ymax = ymin + DY + overlap - 1
            patches[idx] = input[ymin:ymax, xmin:xmax]

        return patches, origins

    def compute_tiles_auto(self, input: Union[np.ndarray, torch.Tensor]):
        raise NotImplementedError("compute_tiles_auto is not implemented yet")


if __name__ == "__main__":
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

    print("done")
