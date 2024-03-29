import inspect
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, TypedDict, Union

import cv2
import h5py
import numpy as np
import torch

from ..config import Config
from ..constants import Quality, TileSelection, get_size_by_quality
from ..utils.image import Image, resize_image
from ..utils.tiling import Tiler

logger = logging.getLogger("dim")


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray]
    lafs: Optional[np.ndarray]
    tile_idx: Optional[np.ndarray]


def extractor_loader(root, model):
    """
    Load and return the specified extractor class from the given root module.

    Args:
        root (module): The root module where the extractor module is located.
        model (str): The name of the extractor module.

    Returns:
        class: The specified extractor class.

    Raises:
        AssertionError: If no or multiple extractor classes are found.

    """
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], ExtractorBase)]
    assert len(classes) == 1, classes
    return classes[0][1]
    # return getattr(module, 'Model')


def save_features_h5(feature_path: Path, features: FeaturesDict, im_name: str, as_half: bool = True):
    # If as_half is True then the features are converted to float16.
    if as_half:
        feat_dtype = np.float16
        for k in features:
            if not isinstance(features[k], np.ndarray):
                continue
            dt = features[k].dtype
            if (dt == np.float32) and (dt != feat_dtype):
                features[k] = features[k].astype(feat_dtype)
    else:
        feat_dtype = np.float32

    with h5py.File(str(feature_path), "a", libver="latest") as fd:
        try:
            if im_name in fd:
                del fd[im_name]
            grp = fd.create_group(im_name)
            for k, v in features.items():
                if k == "im_path" or k == "feature_path":
                    grp.create_dataset(k, data=str(v))
                if isinstance(v, np.ndarray):
                    grp.create_dataset(
                        k,
                        data=v,
                        dtype=feat_dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                else:
                    raise TypeError(f"Features data must be of type np.ndarray, not {type(v)}")

        except OSError as error:
            if "No space left on device" in error.args[0]:
                logger.error(
                    "Out of disk space: storing features on disk can take "
                    "significant space, did you enable the as_half flag?"
                )
                del grp, fd[im_name]
            raise error


class ExtractorBase(metaclass=ABCMeta):
    _default_general_conf = {
        "quality": Quality.HIGH,
        "tile_selection": TileSelection.NONE,
        "tile_size": (1024, 1024),  # (x, y) or (width, height)
        "tile_overlap": 0,  # in pixels
        "force_cpu": False,
        "do_viz": False,
    }
    _default_conf = {}
    required_inputs = []
    grayscale = True
    as_float = True
    interp = "cv2_area"  # "cv2_area", "cv2_linear", or "pil_bilinear" (more accurate but slower)
    descriptor_size = 128
    features_as_half = True

    def __init__(self, custom_config: Config) -> None:
        """
        Initialize the instance with a custom config. This is the method to be called by subclasses

        Args:
            custom_config: A Config object with custom configuration parameters
        """
        # If a custom config is passed, update the default config
        if not isinstance(custom_config, Config):
            raise TypeError("Invalid config object. 'custom_config' must be a Config object")

        # Update default config with custom config
        # NOTE: This is done to keep backward compatibility with the old config format that was a dictionary, it should be replaced with the new config object
        self.config = {
            "general": {
                **self._default_general_conf,
                **custom_config.general,
            },
            "extractor": {
                **self._default_conf,
                **custom_config.extractor,
            },
        }

        # Get main processing parameters and save them as class members
        # NOTE: this is used for backward compatibility, it should be removed
        self._quality = self.config["general"]["quality"]
        self._tiling = self.config["general"]["tile_selection"]
        logger.debug(f"Matching options: Quality: {self._quality.name} - Tiling: {self._tiling.name}")
        logger.debug(f"Saving directory: {self.config['general']['output_dir']}")

        # Get device
        self._device = "cuda" if torch.cuda.is_available() and not self.config["general"]["force_cpu"] else "cpu"
        logger.debug(f"Running inference on device {self._device}")

    def extract(self, img: Union[Image, Path, str]) -> np.ndarray:
        """
        Extract features from an image. This is the main method of the feature extractor.

        Args:
                img: Image to extract features from. It can be either a path to an image or an Image object

        Returns:
                List of features extracted from the image. Each feature is a 2D NumPy array
        """

        if isinstance(img, str):
            im_path = Path(img)
        elif isinstance(img, Image):
            im_path = img.path
        elif isinstance(img, Path):
            im_path = img
        else:
            raise TypeError("Invalid image path. 'img' must be a string, a Path or an Image object")
        if not im_path.exists():
            raise ValueError(f"Image {im_path} does not exist")

        # Define feature path
        feature_path = self.config["general"]["output_dir"] / "features.h5"

        # Load image
        image = cv2.imread(str(im_path))
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.as_float:
            image = image.astype(np.float32)

        # Resize images if needed
        image_ = self._resize_image(self._quality, image, interp=self.interp)

        if self.config["general"]["tile_selection"] == TileSelection.NONE:
            # Extract features from the whole image
            features = self._extract(image_)
            # features["feature_path"] = str(feature_path)
            # features["im_path"] = str(im_path)
            features["tile_idx"] = np.zeros(features["keypoints"].shape[0], dtype=np.float32)

        else:
            # Extract features by tiles
            features = self._extract_by_tile(image_, select_unique=True)
            # features["feature_path"] = str(feature_path)
            # features["im_path"] = str(im_path)
        logger.debug(f"Extracted {len(features['keypoints'])} keypoints")

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        features = self._resize_features(self._quality, features)

        # Add the image_size to the features (if not already present)
        features["image_size"] = np.array(image.shape[:2])

        # Save features to disk in h5 format
        save_features_h5(
            feature_path,
            features,
            im_path.name,
            as_half=self.features_as_half,
        )

        # For debug: visualize keypoints and save to disk
        if self.config["general"]["verbose"]:
            viz_dir = self.config["general"]["output_dir"] / "debug" / "keypoints"
            viz_dir.mkdir(parents=True, exist_ok=True)
            image = cv2.imread(str(im_path))
            self.viz_keypoints(
                image,
                features["keypoints"],
                viz_dir,
                im_path.stem,
                img_format="jpg",
                jpg_quality=70,
            )

        return feature_path

    @abstractmethod
    def _extract(self, image: np.ndarray) -> dict:
        """
        Extract features from an image. This is called by ` extract ` method to extract features from the image. This method must be implemented by subclasses.

        Args:
            image: A NumPy array of shape ( height width 3 )

        Returns:
            A dictionary of extracted features
        """
        raise NotImplementedError("Subclasses should implement _extract method!")

    @abstractmethod
    def _frame2tensor(self, image: np.ndarray, device: str = "cpu"):
        """
        Convert a frame to a tensor. This is a low - level method to be used by subclasses that need to convert an image to a tensor with the required format. This method must be implemented by subclasses.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cpu')
        """
        raise NotImplementedError(
            "Subclasses should implement _frame2tensor method to adapt the input image to the required format!"
        )

    def _extract_by_tile(self, image: np.ndarray, select_unique: bool = True):
        """
        Extract features from an image by tiles. This is called by :meth:`extract` to extract features from the image.

        Args:
            image: The image to extract from. Must be a 2D array
            select_unique: If True the unique values of keypoints are selected
        """
        # Compute tiles limits
        tile_size = self.config["general"]["tile_size"]
        overlap = self.config["general"]["tile_overlap"]
        tiler = Tiler(tiling_mode="size")
        tiles, tiles_origins, padding = tiler.compute_tiles_by_size(input=image, window_size=tile_size, overlap=overlap)

        # Initialize empty arrays
        kpts_full = np.array([], dtype=np.float32).reshape(0, 2)
        descriptors_full = np.array([], dtype=np.float32).reshape(self.descriptor_size, 0)
        scores_full = np.array([], dtype=np.float32)
        tile_idx_full = np.array([], dtype=np.float32)

        # Extract features from each tile
        for idx, tile in tiles.items():
            logger.debug(f"  - Extracting features from tile: {idx}")

            # Extract features in tile
            feat_tile = self._extract(tile)
            kp_tile = feat_tile["keypoints"]
            des_tile = feat_tile["descriptors"]
            if "scores" in feat_tile:
                scor_tile = feat_tile["scores"]
            else:
                scor_tile = None

            # For debug: visualize keypoints and save to disk
            if self.config["general"]["verbose"]:
                tile = np.uint8(tile)
                viz_dir = self.config["general"]["output_dir"] / "debug" / "tiles"
                viz_dir.mkdir(parents=True, exist_ok=True)
                self.viz_keypoints(
                    tile,
                    kp_tile,
                    viz_dir,
                    f"tile_{idx}",
                    img_format="jpg",
                    jpg_quality=70,
                )

            # get keypoints in original image coordinates
            kp_tile += np.array(tiles_origins[idx])

            # Check if any keypoints are outside the original image (non-padded) or too close to the border
            border_thr = 2  # Adjust this threshold as needed
            mask = (
                (kp_tile[:, 0] >= border_thr)
                & (kp_tile[:, 0] < image.shape[1] - border_thr)
                & (kp_tile[:, 1] >= border_thr)
                & (kp_tile[:, 1] < image.shape[0] - border_thr)
            )
            kp_tile = kp_tile[mask]
            des_tile = des_tile[:, mask]
            if scor_tile is not None:
                scor_tile = scor_tile[mask]

            if len(kp_tile) > 0:
                kpts_full = np.vstack((kpts_full, kp_tile))
                descriptors_full = np.hstack((descriptors_full, des_tile))
                tile_idx = np.full(len(kp_tile), idx, dtype=np.float32)
                tile_idx_full = np.concatenate((tile_idx_full, tile_idx))
                if scor_tile is not None:
                    scores_full = np.concatenate((scores_full, scor_tile))
                else:
                    scores_full = None

        if scores_full is None:
            logger.warning("No scores found in features")
            scores_full = np.ones(kpts_full.shape[0], dtype=np.float32)

        # Select unique keypoints
        if select_unique is True:
            kpts_full, unique_idx = np.unique(kpts_full, axis=0, return_index=True)
            descriptors_full = descriptors_full[:, unique_idx]
            tile_idx_full = tile_idx_full[unique_idx]
            scores_full = scores_full[unique_idx]

        # Make FeaturesDict object
        features = FeaturesDict(
            keypoints=kpts_full,
            descriptors=descriptors_full,
            scores=scores_full,
            tile_idx=tile_idx_full,
        )

        return features

    def _resize_image(self, quality: Quality, image: np.ndarray, interp: str = "cv2_area") -> Tuple[np.ndarray]:
        """
        Resize images based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            image (np.ndarray): The first image.

        Returns:
            Tuple[np.ndarray]: Resized images.

        """
        # If quality is HIGHEST, force interpolation to cv2_cubic
        if quality == Quality.HIGHEST:
            interp = "cv2_cubic"
        if quality == Quality.HIGH:
            return image  # No resize
        new_size = get_size_by_quality(quality, image.shape[:2])
        return resize_image(image, (new_size[1], new_size[0]), interp=interp)

    def _resize_features(self, quality: Quality, features: FeaturesDict) -> Tuple[FeaturesDict]:
        """
        Resize features based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            features (FeaturesDict): The features to be resized.

        Returns:
            Tuple[FeaturesDict]: Resized features.

        """
        if quality == Quality.HIGHEST:
            features["keypoints"] /= 2
        elif quality == Quality.HIGH:
            pass
        elif quality == Quality.MEDIUM:
            features["keypoints"] *= 2
        elif quality == Quality.LOW:
            features["keypoints"] *= 4
        elif quality == Quality.LOWEST:
            features["keypoints"] *= 8

        return features

    def viz_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        output_dir: Path,
        im_name: str = "keypoints",
        resize_to: int = 2000,
        img_format: str = "jpg",
        jpg_quality: int = 90,
    ):
        """
        Visualizes keypoints on an image and saves the result to a file.

        Args:
            image (np.ndarray): The input image.
            keypoints (np.ndarray): The keypoints to visualize.
            output_dir (Path): The directory to save the output image.
            im_name (str, optional): The name of the output image file. Defaults to "keypoints".
            resize_to (int, optional): The maximum size (in pixels) to resize the image. Defaults to 2000.
            img_format (str, optional): The format of the output image file. Defaults to "jpg".
            jpg_quality (int, optional): The JPEG quality of the output image (only applicable if img_format is "jpg"). Defaults to 90.
        """
        if resize_to > 0:
            size = image.shape[:2][::-1]
            scale = resize_to / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = cv2.resize(image, size_new)
            keypoints = keypoints * scale

        kk = [cv2.KeyPoint(x, y, 1) for x, y in keypoints]
        out = cv2.drawKeypoints(
            image,
            kk,
            0,
            (0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )
        out_path = str(output_dir / f"{im_name}.{img_format}")
        if img_format == "jpg":
            cv2.imwrite(
                out_path,
                out,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality],
            )
        else:
            cv2.imwrite(out_path, out)
