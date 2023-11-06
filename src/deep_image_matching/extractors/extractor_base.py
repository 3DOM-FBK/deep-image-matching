import inspect
import logging
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, TypedDict, Union

import cv2
import h5py
import numpy as np
import torch

from ..utils.consts import Quality, TileSelection, def_cfg_general
from ..utils.image import Image
from ..utils.tiling import Tiler

logger = logging.getLogger(__name__)


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray]
    tile_idx: Optional[np.ndarray]


def featuresDict_2_tensor(features: FeaturesDict, device: torch.device) -> FeaturesDict:
    return {
        k: torch.tensor(v, dtype=torch.float, device=device)
        for k, v in features.items()
    }


def extractor_loader(root, model):
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


class ExtractorBase(metaclass=ABCMeta):
    default_conf = {"general": def_cfg_general}
    required_inputs = []
    grayscale = True
    descriptor_size = 128

    def __init__(self, **custom_config: dict):
        """
        Initialize the instance with a custom config. This is the method to be called by subclasses

        Args:
                custom_config: a dictionary of options to
        """
        # Set default config
        self._config = self.default_conf

        # If a custom config is passed, update the default config
        if not isinstance(custom_config, dict):
            raise TypeError("opt must be a dictionary")
        self._update_config(custom_config)

    def extract(self, img: Union[Image, Path]) -> np.ndarray:
        """
        Extract features from an image. This is the main method of the feature extractor.

        Args:
                img: Image to extract features from.

        Returns:
                List of features extracted from the image. Each feature is a 2D NumPy array
        """
        # Load image

        im_path = img if isinstance(img, Path) else img.absolute_path
        image = cv2.imread(str(im_path)).astype(np.float32)

        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize images if needed
        image_ = self._resize_image(self._quality, image)

        if self._config["general"]["tile_selection"] == TileSelection.NONE:
            # Extract features from the whole image
            features = self._extract(image_)
            features.tile_idx = np.zeros(features.keypoints.shape[0], dtype=np.float32)
        else:
            # Extract features by tiles
            features = self._extract_by_tile(image_, select_unique=True)

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        features = self._resize_features(self._quality, features)

        # Add the image_size to the features (if not already present)
        features["image_size"] = np.array(image.shape[:2])

        # Save features to disk in h5 format (TODO: MOVE it to another method)
        # def save_features_to_h5(self)
        as_half = True  # TODO: add this to the config
        output_dir = Path(self._config["general"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_path = output_dir / "features.h5"
        im_name = im_path.name

        # If as_half is True then the features are converted to float32 or float16.
        if as_half:
            for k in features:
                if not isinstance(features[k], np.ndarray):
                    continue
                dt = features[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    features[k] = features[k].astype(np.float16)

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                if im_name in fd:
                    del fd[im_name]
                grp = fd.create_group(im_name)
                for k, v in features.items():
                    if isinstance(v, np.ndarray):
                        grp.create_dataset(k, data=v)
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    del grp, fd[im_name]
                raise error

        # Save also keypoints and descriptors separately
        with h5py.File(str(output_dir / "keypoints.h5"), "a", libver="latest") as fd:
            if im_name in fd:
                del fd[im_name]
            fd[im_name] = features["keypoints"]

        desc_dim = features["descriptors"].shape[0]
        with h5py.File(str(output_dir / "descriptors.h5"), "a", libver="latest") as fd:
            if im_name in fd:
                del fd[im_name]
            fd[im_name] = features["descriptors"].reshape(-1, desc_dim)

        return feature_path

    @abstractmethod
    def _extract(self, image: np.ndarray) -> dict:
        """
        Extract features from an image. This is called by : meth : ` extract ` to extract features from the image. This method must be implemented by subclasses.

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

    def _update_config(self, config: dict):
        """
        Update the config dictionary. This is called by : meth : ` update_config ` to allow subclasses to perform additional checks before and after configuration is updated.

        Args:
           config: The configuration dictionary to update in place. It is assumed that the keys and values are valid
        """

        # Make a deepcopy of the default config and update it with the custom config
        new_config = deepcopy(self._config)
        for key in config:
            if key not in new_config:
                new_config[key] = config[key]
            else:
                new_config[key] = {**new_config[key], **config[key]}

        # Check general config
        required_keys_general = [
            "quality",
            "tile_selection",
            "force_cpu",
        ]
        missing_keys = [
            key for key in required_keys_general if key not in new_config["general"]
        ]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in 'general' config: {', '.join(missing_keys)}."
            )
        if not isinstance(new_config["general"]["quality"], Quality):
            raise TypeError("quality must be a Quality enum")
        if not isinstance(new_config["general"]["tile_selection"], TileSelection):
            raise TypeError("tile_selection must be a TileSelection enum")

        # Update the current config with the custom config
        self._config = new_config

        # Get main processing parameters and save them as class members
        self._quality = self._config["general"]["quality"]
        self._tiling = self._config["general"]["tile_selection"]
        logger.debug(
            f"Matching options: Quality: {self._quality.name} - Tiling: {self._tiling.name}"
        )

        # Define saving directory
        output_dir = self._config["general"]["output_dir"]
        if output_dir is not None:
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._output_dir = None
        logger.debug(f"Saving directory: {self._output_dir}")

        # Get device
        self._device = (
            "cuda"
            if torch.cuda.is_available() and not self._config["general"]["force_cpu"]
            else "cpu"
        )
        logger.debug(f"Running inference on device {self._device}")

    def _extract_by_tile(self, image: np.ndarray, select_unique: bool = True):
        """
        Extract features from an image by tiles. This is called by :meth:`extract` to extract features from the image.

        Args:
            image: The image to extract from. Must be a 2D array
            select_unique: If True the unique values of keypoints are selected
        """
        # Compute tiles limits
        grid = self._config["general"]["tiling_grid"]
        overlap = self._config["general"]["tiling_overlap"]
        tiler = Tiler(grid=grid, overlap=overlap)
        t_lims, t_origin = tiler.compute_limits_by_grid(image)

        # Initialize empty arrays
        kpts_full = np.array([], dtype=np.float32).reshape(0, 2)
        descriptors_full = np.array([], dtype=np.float32).reshape(
            self.descriptor_size, 0
        )
        scores_full = np.array([], dtype=np.float32)
        tile_idx_full = np.array([], dtype=np.float32)

        # Extract features from each tile
        for idx, lim in t_lims.items():
            logger.debug(f"  - Extracting features from tile: {idx}")

            # Extract features in tile
            tile = tiler.extract_patch(image, lim)
            feat_tile = self._extract(tile)

            # append features
            kpts_full = np.vstack(
                (kpts_full, feat_tile["keypoints"] + np.array(lim[0:2]))
            )
            descriptors_full = np.hstack((descriptors_full, feat_tile["descriptors"]))
            scores_full = np.concatenate((scores_full, feat_tile["scores"]))
            tile_idx_full = np.concatenate(
                (
                    tile_idx_full,
                    np.ones(feat_tile["keypoints"].shape[0], dtype=np.float32) * idx,
                )
            )
        if select_unique is True:
            kpts_full, unique_idx = np.unique(kpts_full, axis=0, return_index=True)
            descriptors_full = descriptors_full[:, unique_idx]
            scores_full = scores_full[unique_idx]
            tile_idx_full = tile_idx_full[unique_idx]

        # Make FeaturesDict object
        features = FeaturesDict(
            keypoints=kpts_full,
            descriptors=descriptors_full,
            scores=scores_full,
            tile_idx=tile_idx_full,
        )

        return features

    def _resize_image(self, quality: Quality, image: np.ndarray) -> Tuple[np.ndarray]:
        """
        Resize images based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            image (np.ndarray): The first image.

        Returns:
            Tuple[np.ndarray]: Resized images.

        """
        if quality == Quality.HIGHEST:
            image_ = cv2.pyrUp(image)
        elif quality == Quality.HIGH:
            image_ = image
        elif quality == Quality.MEDIUM:
            image_ = cv2.pyrDown(image)
        elif quality == Quality.LOW:
            image_ = cv2.pyrDown(cv2.pyrDown(image))
        return image_

    def _resize_features(
        self, quality: Quality, features: FeaturesDict
    ) -> Tuple[FeaturesDict]:
        """
        Resize features based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            features0 (FeaturesDict): The features of the first image.
            features1 (FeaturesDict): The features of the second image.

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

        return features
