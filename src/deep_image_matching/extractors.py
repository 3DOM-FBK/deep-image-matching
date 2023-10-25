import cv2
import numpy as np
from pathlib import Path
import logging
from copy import deepcopy
from typing import Tuple, Union, TypedDict, Optional
import torch
import h5py

from .image import Image
from .consts import Quality, TileSelection
from .tiling import Tiler

logger = logging.getLogger(__name__)


# TODO: move to another file to share it with matchers.py
# @dataclass
# class FeaturesBase:
#     keypoints: np.ndarray
#     descriptors: np.ndarray = None
#     scores: np.ndarray = None
#     tile_idx: np.ndarray = None

#     def to_dict(self, as_tensor: bool = False, device=None) -> dict:
#         dic = {
#             "keypoints": self.keypoints,
#             "descriptors": self.descriptors,
#             "scores": self.scores,
#             "tile_idx": self.tile_idx,
#         }
#         if as_tensor:
#             dic = {k: torch.tensor(v, device=device) for k, v in dic.items()}
#         return dic

#     @classmethod
#     def from_dict(cls, dic: dict) -> "FeaturesBase":
#         defualt_dic = {
#             "keypoints": None,
#             "descriptors": None,
#             "scores": None,
#             "tile_idx": None,
#         }
#         dic = {**defualt_dic, **dic}
#         for k, v in dic.items():
#             if isinstance(v, np.ndarray):
#                 v = v.astype("float32")
#         return cls(
#             keypoints=dic["keypoints"],
#             descriptors=dic["descriptors"],
#             scores=dic["scores"],
#             tile_idx=dic["tile_idx"],
#         )


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


DEBUG = True

DEFAULT_CONFIG = {
    "general": {
        "quality": Quality.HIGH,
        "tile_selection": TileSelection.NONE,
        "tiling_grid": [1, 1],
        "tiling_overlap": 0,
        "save_dir": "results",
        "force_cpu": False,
        "do_viz": False,
        "fast_viz": True,
        "do_viz_tiles": False,
    }
}


# TODO: separate the ExtractorBase class from the SuperPointExtractor class
class ExtractorBase:
    def __init__(self, **custom_config: dict):
        # Set default config
        self._config = DEFAULT_CONFIG

        # If a custom config is passed, update the default config
        if not isinstance(custom_config, dict):
            raise TypeError("opt must be a dictionary")
        self._update_config(custom_config)

        # Load extractor (TODO: DO IT IN THE SUBCLASS!)
        from deep_image_matching.hloc.extractors.superpoint import SuperPoint

        cfg = {"name": "superpoint", "nms_radius": 3, "max_keypoints": 4096}
        self._extractor = SuperPoint(cfg).eval().to(self._device)

    def extract(self, img: Union[Image, Path]) -> np.ndarray:
        # Load image

        im_path = img if isinstance(img, Path) else img.absolute_path
        image = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)

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

        # Save features to disk in h5 format (TODO: MOVE it to another method)
        # def save_features_to_h5(self)
        as_half = False  # TODO: add this to the config
        save_dir = Path(self._config["general"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        feature_path = save_dir / "features.h5"
        im_name = im_path.name

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

        return feature_path

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> dict:
        # raise NotImplementedError("Subclasses should implement _extract method!")

        # Convert image from numpy array to tensor
        image_ = self._frame2tensor(image, self._device)

        # Extract features
        feats = self._extractor({"image": image_})

        # Remove elements from list/tuple
        feats = {
            k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()
        }
        # Convert tensors to numpy arrays
        feats = {k: v.cpu().numpy() for k, v in feats.items()}

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu"):
        if len(image.shape) == 2:
            image = image[None][None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)[None]
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)

    def _update_config(self, config: dict):
        """Check the matching config dictionary for missing keys or invalid values."""

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
        logger.info(
            f"Matching options: Quality: {self._quality.name} - Tiling: {self._tiling.name}"
        )

        # Define saving directory
        save_dir = self._config["general"]["save_dir"]
        if save_dir is not None:
            self._save_dir = Path(save_dir)
            self._save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._save_dir = None
        logger.info(f"Saving directory: {self._save_dir}")

        # Get device
        self._device = (
            "cuda"
            if torch.cuda.is_available() and not self._config["general"]["force_cpu"]
            else "cpu"
        )
        logger.info(f"Running inference on device {self._device}")

    def _extract_by_tile(self, image: np.ndarray, select_unique: bool = True):
        # Compute tiles limits
        grid = self._config["general"]["tiling_grid"]
        overlap = self._config["general"]["tiling_overlap"]
        tiler = Tiler(grid=grid, overlap=overlap)
        t_lims, t_origin = tiler.compute_limits_by_grid(image)

        # Initialize empty arrays
        kpts_full = np.array([], dtype=np.float32).reshape(0, 2)
        descriptors_full = np.array([], dtype=np.float32).reshape(256, 0)
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

    def _filter_kpts_by_mask(self, features: FeaturesDict, inlMask: np.ndarray) -> None:
        """
        Filter matches based on the specified mask.

        Args:
            features (FeaturesBase): The features to filter.
            inlMask (np.ndarray): The mask to filter matches.
        """
        features.keypoints = features.keypoints[inlMask, :]
        if features.descriptors is not None:
            features.descriptors = features.descriptors[:, inlMask]
        if features.scores is not None:
            features.scores = features.scores[inlMask]


class SuperPointExtractor(ExtractorBase):
    def __init__(self, config: dict):
        super().__init__(config)

    def extract(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
