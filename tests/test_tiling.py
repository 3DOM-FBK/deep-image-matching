import kornia
import numpy as np
import pytest
import torch
from deep_image_matching.utils import Tiler


@pytest.fixture
def tiler():
    return Tiler()


def konria_071(base_version: str = "0.7.1"):
    try:
        from packaging import version
    except ImportError:
        return False
    return version.parse(kornia.__version__) == version.parse(base_version)


def test_compute_tiles_by_size_no_overlap_no_padding(tiler):
    # Create a numpy array with shape (100, 100, 3)
    input_shape = (100, 100, 3)
    input_image = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    window_size = 50
    overlap = 0

    tiles, origins, padding = tiler.compute_tiles_by_size(input_image, window_size, overlap)

    # Assert the output types and shapes
    assert isinstance(tiles, dict)
    assert isinstance(origins, dict)
    assert isinstance(padding, tuple)
    if konria_071():
        assert len(padding) == 2
    else:
        assert len(padding) == 4

    # Assert the number of tiles and origins
    assert len(tiles) == 4
    assert len(origins) == 4

    # Assert the shape of the tiles
    for tile in tiles.values():
        assert tile.shape == (window_size, window_size, 3)

    # Assert the padding values
    if konria_071():
        assert padding == (0, 0)
    else:
        assert padding == (0, 0, 0, 0)


def test_compute_tiles_by_size_no_overlap_padding(tiler):
    # Create a numpy array with shape (100, 100, 3)
    input_shape = (100, 100, 3)
    input_image = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    window_size = 40
    overlap = 0

    tiles, origins, padding = tiler.compute_tiles_by_size(input_image, window_size, overlap)

    # Assert the output types and shapes
    assert isinstance(tiles, dict)
    assert isinstance(origins, dict)
    assert isinstance(padding, tuple)
    if konria_071():
        assert len(padding) == 2
    else:
        assert len(padding) == 4
    # Assert the number of tiles and origins
    assert len(tiles) == 9
    assert len(origins) == 9

    # Assert the shape of the tiles
    for tile in tiles.values():
        assert tile.shape == (window_size, window_size, 3)

    # Assert the padding values
    if konria_071():
        assert padding == (10, 10)
    else:
        assert padding == (10, 10, 10, 10)


def test_compute_tiles_by_size_overlap_no_padding(tiler):
    # Create a numpy array with shape (100, 100, 3)
    input_shape = (100, 100, 3)
    input_image = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    window_size = 50
    overlap = 10

    tiles, origins, padding = tiler.compute_tiles_by_size(input_image, window_size, overlap)

    # Assert the output types and shapes
    assert isinstance(tiles, dict)
    assert isinstance(origins, dict)
    assert isinstance(padding, tuple)
    if konria_071():
        assert len(padding) == 2
    else:
        assert len(padding) == 4
    # Assert the number of tiles and origins
    assert len(tiles) == 4
    assert len(origins) == 4

    # Assert the shape of the tiles
    for tile in tiles.values():
        assert tile.shape == (window_size, window_size, 3)

    # Assert the padding values
    if konria_071():
        assert padding == (0, 0)
    else:
        assert padding == (0, 0, 0, 0)


def test_compute_tiles_by_size_with_torch_tensor(tiler):
    # Create a torch tensor with shape (3, 100, 100)
    channels = 3
    input_shape = (channels, 100, 100)
    input_image = torch.randint(0, 255, input_shape, dtype=torch.uint8)
    window_size = (50, 50)
    overlap = (0, 0)

    tiles, origins, padding = tiler.compute_tiles_by_size(input_image, window_size, overlap)

    # Assert the output types and shapes
    assert isinstance(tiles, dict)
    assert isinstance(origins, dict)
    assert isinstance(padding, tuple)
    if konria_071():
        assert len(padding) == 2
    else:
        assert len(padding) == 4
    # Assert the number of tiles and origins
    assert len(tiles) == 4
    assert len(origins) == 4

    # Assert the shape of the tiles
    for tile in tiles.values():
        assert tile.shape == (window_size[0], window_size[1], channels)

    # Assert the padding values
    if konria_071():
        assert padding == (0, 0)
    else:
        assert padding == (0, 0, 0, 0)


def test_compute_tiles_by_size_with_invalid_input(tiler):
    # Create an invalid window_size (a string)
    input_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    window_size = "32"
    overlap = 8

    with pytest.raises(TypeError):
        tiler.compute_tiles_by_size(input_image, window_size, overlap)

    # Create an invalid overlap (a float)
    window_size = 32
    overlap = 8.0

    with pytest.raises(TypeError):
        tiler.compute_tiles_by_size(input_image, window_size, overlap)


if __name__ == "__main__":
    pytest.main([__file__])
