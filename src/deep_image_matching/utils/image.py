import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import cv2
import exifread
import numpy as np
import PIL
from PIL import Image

from .sensor_width_database import SensorWidthDatabase

logger = logging.getLogger("dim")


IMAGE_EXT = [".jpg", ".JPG", ".png", ".PNG", ".tif", "TIF"]

Image.MAX_IMAGE_PIXELS = None


def read_image(
    path: Union[str, Path],
    color: bool = True,
) -> np.ndarray:
    """
    Reads image with OpenCV and returns it as a NumPy array.

    Args:
        path (Union[str, Path]): The path of the image.
        color (bool, optional): Whether to read the image as color (RGB) or grayscale. Defaults to True.

    Returns:
        np.ndarray: The image as a NumPy array.
    """

    if not Path(path).exists():
        raise ValueError(f"File {path} does not exist")

    if color:
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_GRAYSCALE

    image = cv2.imread(str(path), flag)

    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],  # Destination size (width, height)
    interp: str = "cv2_area",
) -> np.ndarray:
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class Image:
    """A class representing an image.

    Attributes:
        _path (Path): The path to the image file.
        _value_array (np.ndarray): Numpy array containing pixel values. If available, it can be accessed with `Image.value`.
        _width (int): The width of the image in pixels.
        _height (int): The height of the image in pixels.
        _exif_data (dict): The EXIF metadata of the image, if available.
        _date_time (datetime): The date and time the image was taken, if available.

    """

    IMAGE_EXT = IMAGE_EXT
    DATE_FMT = "%Y-%m-%d"
    TIME_FMT = "%H:%M:%S"
    DATETIME_FMT = "%Y:%m:%d %H:%M:%S"
    DATE_FORMATS = [DATETIME_FMT, DATE_FMT, TIME_FMT]

    def __init__(self, path: Union[str, Path], id: int = None) -> None:
        """
        __init__ Create Image object as a lazy loader for image data

        Args:
            path (Union[str, Path]): path to the image
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"File {path} does not exist")

        if path.suffix not in self.IMAGE_EXT:
            raise ValueError(f"File {path} is not a valid image file")

        self._path = path
        self._id = id
        self._width = None
        self._height = None
        self._exif_data = None
        self._date_time = None
        self._focal_length = None

        try:
            self.read_exif()
        except Exception:
            img = PIL.Image.open(path)
            self._width, self._height = img.size

    def __repr__(self) -> str:
        """Returns a string representation of the image"""
        return f"Image {self._path}"

    def __str__(self) -> str:
        """Returns a string representation of the image"""
        return f"Image {self._path}"

    @property
    def id(self) -> int:
        """Returns the id of the image"""
        if self._id is None:
            logger.error(f"Image id not available for {self.name}. Set it first")
        return self._id

    @property
    def name(self) -> str:
        """Returns the name of the image (including extension)"""
        return self._path.name

    @property
    def stem(self) -> str:
        """Returns the name of the image (excluding extension)"""
        return self._path.stem

    @property
    def path(self) -> Path:
        """Path of the image"""
        return self._path

    @property
    def parent(self) -> str:
        """Path to the parent folder of the image"""
        return self._path.parent

    @property
    def extension(self) -> str:
        """Returns the extension  of the image"""
        return self._path.suffix

    @property
    def height(self) -> int:
        """Returns the height of the image in pixels"""
        if self._height is None:
            logger.error(f"Image height not available for {self.name}. Try to read it from the image file.")
            try:
                img = PIL.Image.open(self._path)
                self._width, self._height = img.size
            except Exception as e:
                logger.error(f"Unable to read image size for {self.name}: {e}")
                return None
        return int(self._height)

    @property
    def width(self) -> int:
        """Returns the width of the image in pixels"""
        if self._width is None:
            logger.error(f"Image width not available for {self.name}. Try to read it from the image file.")
            try:
                img = PIL.Image.open(self._path)
                self._width, self._height = img.size
            except Exception as e:
                logger.error(f"Unable to read image size for {self.name}: {e}")
                return None

        return int(self._width)

    @property
    def size(self) -> tuple:
        """Returns the size of the image in pixels as a tuple (width, height)"""
        if self._width is None or self._height is None:
            logger.warning(f"Image size not available for {self.name}. Trying to read it from the image file.")
            try:
                img = PIL.Image.open(self._path)
                self._width, self._height = img.size
            except Exception as e:
                logger.error(f"Unable to read image size for {self.name}: {e}")
                return None

        return (int(self._width), int(self._height))

    @property
    def exif(self) -> dict:
        """exif Returns the exif of the image"""
        if self._exif_data is None:
            logger.error(f"No exif data available for {self.name}.")
            return None
        return self._exif_data

    @property
    def date(self) -> str:
        """Returns the date and time of the image in a string format."""
        if self._date_time is None:
            logger.error(f"No exif data available for {self.name}.")
            return None
        return self._date_time.strftime(self.DATE_FMT)

    @property
    def time(self) -> str:
        """time Returns the time of the image from exif as a string"""
        if self._date_time is None:
            logger.error(f"No exif data available for {self.name}.")
            return None
        return self._date_time.strftime(self.TIME_FMT)

    @property
    def datetime(self) -> datetime:
        """Returns the date and time of the image as datetime object."""
        if self._date_time is None:
            logger.error(f"No exif data available for {self.name}.")
            return None
        return self._date_time

    @property
    def timestamp(self) -> str:
        """Returns the date and time of the image in a string format."""
        if self._date_time is None:
            logger.error(f"No exif data available for {self.name}.")
            return None
        return self._date_time.strftime(self.DATETIME_FMT)

    @property
    def focal_length(self) -> float:
        """Returns the focal length of the image in mm."""
        if self._focal_length is None:
            logger.error(f"Focal length not available in exif data for {self.name}.")
            return None
        return self._focal_length

    def read(self) -> np.ndarray:
        """Returns the image (pixel values) as numpy array"""
        return read_image(self._path)

    def read_exif(self) -> None:
        """
        Read image exif with exifread and store them in a dictionary

        Raises:
            IOError: If there is an error reading the image file.
            InvalidExif: If the exif data is invalid.
            ExifNotFound: If no exif data is found for the image.

        Returns:
            None

        """
        from exifread.exceptions import ExifNotFound, InvalidExif

        try:
            with open(self._path, "rb") as f:
                exif = exifread.process_file(f, details=False, debug=False)
        except IOError as e:
            logger.info(f"{e}. Unable to read exif data for image {self.name}.")
            raise InvalidExif("Exif error")
        except InvalidExif as e:
            logger.info(f"Unable to read exif data for image {self.name}. {e}")
            raise ValueError("Exif error")
        except ExifNotFound as e:
            logger.info(f"Unable to read exif data for image {self.name}. {e}")
            raise ValueError("Exif error")

        if len(exif) == 0:
            logger.info(f"No exif data available for image {self.name} (this will probably not affect the matching).")
            raise ValueError("Exif error")

        # Get image size
        if "Image ImageWidth" in exif.keys() and "Image ImageLength" in exif.keys():
            self._width = exif["Image ImageWidth"].printable
            self._height = exif["Image ImageLength"].printable
        elif "EXIF ExifImageWidth" in exif.keys() and "EXIF ExifImageLength" in exif.keys():
            self._width = exif["EXIF ExifImageWidth"].printable
            self._height = exif["EXIF ExifImageLength"].printable

        # Get Image Date and Time
        if "Image DateTime" in exif.keys():
            date_str = exif["Image DateTime"].printable
        elif "EXIF DateTimeOriginal" in exif.keys():
            date_str = exif["EXIF DateTimeOriginal"].printable
        else:
            logger.info(f"Date not available in exif for {self.name}")
            date_str = None
        if date_str is not None:
            for format in self.DATE_FORMATS:
                try:
                    self._date_time = datetime.strptime(date_str, format)
                    break
                except ValueError:
                    continue

        # Get Focal Length
        if "EXIF FocalLength" in exif.keys():
            try:
                focal_length_str = exif["EXIF FocalLength"].printable

                # Check if it's a ratio
                if "/" in focal_length_str:
                    numerator, denominator = focal_length_str.split("/")
                    self._focal_length = float(numerator) / float(denominator)
                else:
                    self._focal_length = float(focal_length_str)
            except ValueError:
                logger.info(f"Unable to get focal length from exif for image {self.name}")

        # Store exif data
        self._exif_data = exif

        # TODO: Get GPS coordinates from exif

    def get_intrinsics_from_exif(self) -> np.ndarray:
        """Constructs the camera intrinsics from exif tag.

        Equation: focal_px=max(w_px,h_px)*focal_mm / ccdw_mm

        Note:
            References for this functions can be found:

            * https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
            * https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
            * https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from # noqa: E501

        Returns:
            K (np.ndarray): intrinsics matrix (3x3 numpy array).
        """
        if self._exif_data is None or len(self._exif_data) == 0:
            try:
                self.read_exif()
            except OSError:
                logger.error("Unable to read exif data.")
                return None
        try:
            focal_length_mm = float(self._exif_data["EXIF FocalLength"].printable)
        except OSError:
            logger.error("Focal length non found in exif data.")
            return None
        try:
            sensor_width_db = SensorWidthDatabase()
            sensor_width_mm = sensor_width_db.lookup(
                self._exif_data["Image Make"].printable,
                self._exif_data["Image Model"].printable,
            )
        except OSError:
            logger.error("Unable to get sensor size in mm from sensor database")
            return None

        img_w_px = self.width
        img_h_px = self.height
        focal_length_px = max(img_h_px, img_w_px) * focal_length_mm / sensor_width_mm
        center_x = img_w_px / 2
        center_y = img_h_px / 2
        K = np.array(
            [
                [focal_length_px, 0.0, center_x],
                [0.0, focal_length_px, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return K


class ImageList:
    """
    Represents a collection of Image objects

    Attributes:
        IMAGE_EXT (tuple): Supported image file extensions.
    """

    IMAGE_EXT = IMAGE_EXT

    def __init__(self, img_dir: Path):
        """
        Initializes an ImageList object

        Args:
            img_dir (Path): The path to the directory containing the images.

        Raises:
            ValueError: If the directory does not exist, is not a directory, or
                does not contain any valid images.
        """
        if not img_dir.exists():
            raise ValueError(f"Directory {img_dir} does not exist")

        if not img_dir.is_dir():
            raise ValueError(f"{img_dir} is not a directory")

        self.images = []
        self.current_idx = 0
        i = 0
        all_imgs = [image for image in img_dir.glob("*") if image.suffix in self.IMAGE_EXT]
        all_imgs.sort()

        if len(all_imgs) == 0:
            raise ValueError(f"{img_dir} does not contain any image")

        for image in all_imgs:
            self.add_image(image, i)
            i += 1

    def __len__(self):
        return len(self.images)

    def __repr__(self) -> str:
        return f"ImageList with {len(self.images)} images"

    def __getitem__(self, img_id):
        return self.images[img_id]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.images):
            raise StopIteration
        cur = self.current_idx
        self.current_idx += 1
        return self.images[cur]

    def add_image(self, path: Path, img_id: int):
        """
        Adds a new Image object to the ImageList.

        Args:
            path (Path): The path to the image file.
            img_id (int): The ID to assign to the image.
        """
        new_image = Image(path, img_id)
        self.images.append(new_image)

    @property
    def img_names(self):
        """
        Returns a list of image names in the ImageList.

        Returns:
            list: A list of image names (strings).
        """
        return [im.name for im in self.images]

    @property
    def img_paths(self):
        """
        Returns a list of image paths in the ImageList

        Returns:
            list: A list of image paths (Path objects).
        """
        return [im.path for im in self.images]


if __name__ == "__main__":
    image_path = "data/easy_small/01_Camera1.jpg"

    img = Image(image_path)

    image_dir = "data/easy_small"

    img_list = ImageList(image_dir)

    print("done")
