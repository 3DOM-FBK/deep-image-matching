from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from deep_image_matching.utils.image import Image, ImageList, read_image


@pytest.fixture
def image_dir():
    return (Path(__file__).parents[0].parents[0] / "assets/pytest/images").resolve()


@pytest.fixture
def image_path():
    assets = (Path(__file__).parents[0].parents[0] / "assets/pytest/images").resolve()
    return list(assets.glob("*"))[0]


@pytest.fixture
def image(image_path):
    return Image(image_path)


def test_reads_image(image_path):
    image = read_image(image_path)
    assert isinstance(image, np.ndarray)


def test_can_read_image_as_color_or_grayscale(image_path):
    color_image = read_image(image_path, color=True)
    grayscale_image = read_image(image_path, color=False)
    assert isinstance(color_image, np.ndarray)
    assert isinstance(grayscale_image, np.ndarray)


def test_initialized_with_valid_path(image_path):
    image = Image(image_path)
    assert image.path == Path(image_path)


def test_initialized_with_invalid_path():
    path = "invalid/path"
    with pytest.raises(ValueError):
        Image(path)


def test_initialized_with_non_image_file():
    path = __file__
    with pytest.raises(ValueError):
        Image(path)


def test_read_exif(image):
    assert isinstance(image.name, str)
    assert isinstance(image.stem, str)
    assert isinstance(image.path, Path)
    assert isinstance(image.parent, Path)
    assert isinstance(image.extension, str)
    assert isinstance(image.height, int)
    assert isinstance(image.width, int)
    assert isinstance(image.size, tuple)
    assert isinstance(image.exif, dict)
    assert isinstance(image.date, str)
    assert isinstance(image.time, str)
    assert isinstance(image.datetime, datetime)
    assert isinstance(image.focal_length, float)

    img = image.read()
    assert isinstance(img, np.ndarray)
    assert img.shape == (image.height, image.width, 3)


def test_create_image_list_from_valid_directory(image_dir):
    image_list = ImageList(image_dir)
    assert len(image_list) > 0
    assert isinstance(image_list[0], Image)
    assert len(image_list) == 3


def test_access_image_object_by_index(image_dir):
    image_list = ImageList(image_dir)
    image = image_list[0]
    assert isinstance(image, Image)


def test_raise_value_error_for_nonexistent_directory():
    with pytest.raises(ValueError):
        ImageList(Path("nonexistent_directory"))


def test_raise_value_error_for_directory_without_valid_image_files():
    dir = Path(__file__).parent
    with pytest.raises(ValueError):
        ImageList(dir)


def test_raise_index_error_for_invalid_index(image_dir):
    image_list = ImageList(image_dir)
    with pytest.raises(IndexError):
        image = image_list[10]


if __name__ == "__main__":
    pytest.main([__file__])
