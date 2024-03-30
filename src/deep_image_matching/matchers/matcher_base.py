import inspect
import logging
from abc import ABCMeta, abstractmethod
from itertools import product
from pathlib import Path
from typing import Optional, Tuple, TypedDict

import cv2
import h5py
import numpy as np
import torch

from ..config import Config
from ..constants import Quality, TileSelection, Timer, get_size_by_quality
from ..extractors.extractor_base import ExtractorBase
from ..io.h5 import get_features, get_matches
from ..thirdparty.hloc.extractors.superpoint import SuperPoint
from ..thirdparty.LightGlue.lightglue import LightGlue
from ..utils.geometric_verification import geometric_verification
from ..utils.image import resize_image
from ..utils.tiling import Tiler
from ..visualization import viz_matches_cv2, viz_matches_mpl

logger = logging.getLogger("dim")


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray]
    lafs: Optional[np.ndarray]
    tile_idx: Optional[np.ndarray]


def matcher_loader(root, model):
    """
    Load a matcher class from a specified module.

    Args:
        root (module): The root module containing the specified matcher module.
        model (str): The name of the matcher module to load.

    Returns:
        type: The matcher class.
    """
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    # classes = [c for c in classes if issubclass(c[1], MatcherBase)]
    classes = [c for c in classes if issubclass(c[1], MatcherBase) or issubclass(c[1], DetectorFreeMatcherBase)]
    assert len(classes) == 1, classes
    return classes[0][1]


class MatcherBase(metaclass=ABCMeta):
    """
    Base class for matchers. It defines the basic interface for matchers
    and basic functionalities that are shared among all matchers,
    in particular the `match` method. It must be subclassed to implement a new matcher.

    Attributes:
        general_conf (dict): Default configuration for general settings.
        _default_conf (dict): Default configuration for matcher-specific settings.
        required_inputs (list): List of required input parameters.
        min_inliers_per_pair (int): Minimum number of matches required.
        min_matches_per_tile (int): Minimum number of matches required per tile.
        max_feat_no_tiling (int): Maximum number of features without tiling.
        tile_preselection_size (int): Maximum resize dimension for preselection.
    """

    _default_general_conf = {
        "quality": Quality.LOW,
        "tile_selection": TileSelection.NONE,
        "tile_size": [2048, 1365],
        "tile_overlap": 0,
        "force_cpu": False,
        "do_viz": False,
        "min_inliers_per_pair": 15,
        "min_inlier_ratio_per_pair": 0.2,
        "min_matches_per_tile": 5,
        "tile_preselection_size": 1000,
    }
    _default_conf = {}
    required_inputs = []
    max_feat_no_tiling = 20000

    def __init__(self, custom_config: Config) -> None:
        """
        Initialize the MatcherBase with a custom config. This is the method to be called by subclasses

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
            "matcher": {
                **self._default_conf,
                **custom_config.matcher,
            },
        }
        # Get main processing parameters and save them as class members
        # NOTE: this is used for backward compatibility, it should be removed
        self._quality = self.config["general"]["quality"]
        self._tiling = self.config["general"]["tile_selection"]
        self.min_inliers_per_pair = self.config["general"]["min_inliers_per_pair"]
        self.min_inlier_ratio_per_pair = self.config["general"]["min_inlier_ratio_per_pair"]
        self.min_matches_per_tile = self.config["general"]["min_matches_per_tile"]
        self.tile_preselection_size = self.config["general"]["tile_preselection_size"]

        # Get main processing parameters and save them as class members
        self._tiling = self.config["general"]["tile_selection"]
        logger.debug(f"Matching options: Tiling: {self._tiling.name}")
        logger.debug(f"Saving directory: {self.config['general']['output_dir']}")
        # Get device
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config["general"]["force_cpu"] else "cpu"
        )
        logger.debug(f"Running inference on device {self._device}")

        # Load extractor and matcher for the preselction
        if self.config["general"]["tile_selection"] != TileSelection.NONE:
            sp_cfg = {
                "nms_radius": 5,  # 3
                "max_keypoints": 4000,  # 2048
                "keypoint_threshold": 0.005,  # 0.0005
            }
            lg_cfg = {
                "features": "superpoint",
                "n_layers": 9,
                "depth_confidence": 0.9,
                "width_confidence": 0.95,
                "filter_threshold": 0.3,
                "flash": True,
            }
            self._preselction_extractor = SuperPoint(sp_cfg).eval().to(self._device)
            self._preselction_matcher = LightGlue(**lg_cfg).eval().to(self._device)
        else:
            self._preselction_extractor = None
            self._preselction_matcher = None

    @abstractmethod
    def _match_pairs(
        self,
        feats0: dict,
        feats1: dict,
    ) -> np.ndarray:
        """
        Perform matching between two sets of features. This method must be implemented by subclasses. It takes in two dictionaries containing the features of the two images and returns the matches between keypoints and descriptors in those images.

        Args:
            feats0 (dict): Features of the first image.
            feats1 (dict): Features of the second image.

        Raises:
            NotImplementedError: Subclasses must implement _match_pairs() method.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.
        """
        raise NotImplementedError("Subclasses must implement _match_pairs() method.")

    def match(
        self,
        feature_path: Path,
        matches_path: Path,
        img0: Path,
        img1: Path,
        try_full_image: bool = False,
    ) -> np.ndarray:
        """
        Match features between two images.

        Args:
            feature_path (Path): Path to the feature file.
            matches_path (Path): Path to save the matches.
            img0 (Path): Path to the first image.
            img1 (Path): Path to the second image.
            try_full_image (bool, optional): Flag to attempt matching on full images. Defaults to False.

        Raises:
            RuntimeError: If there are too many features to match on full images.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.
        """

        timer_match = Timer(log_level="debug")

        # Check that feature_path exists
        if not Path(feature_path).exists():
            raise FileNotFoundError(f"Feature file {feature_path} does not exist.")
        else:
            self._feature_path = Path(feature_path)

        # Get features from h5 file
        img0_name = img0.name
        img1_name = img1.name
        features0 = get_features(self._feature_path, img0.name)
        features1 = get_features(self._feature_path, img1.name)
        timer_match.update("load h5 features")

        # Perform matching (on tiles or full images)
        fallback_flag = False

        if self._tiling == TileSelection.NONE:
            too_many_features = (
                len(features0["keypoints"]) > self.max_feat_no_tiling
                or len(features1["keypoints"]) > self.max_feat_no_tiling
            )
            if too_many_features:
                raise RuntimeError(
                    "Too many features to run the matching on full images. Try running the matching with tile selection or use a lower max_keypoints value."
                )
            logger.debug(f"Tile selection was {self._tiling.name}. Matching full images...")
            matches = self._match_pairs(features0, features1)
            timer_match.update("match full images")
        else:
            # If try_full_image is set, try to match full images first
            if try_full_image:
                try:
                    logger.debug(
                        f"Tile selection was {self._tiling.name} but try_full_image is set. Matching full images..."
                    )
                    matches = self._match_pairs(features0, features1)
                    timer_match.update("match full images")
                except Exception as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(f"Matching full images failed: {e}.")
                        fallback_flag = True
                    else:
                        raise e
            else:
                logger.debug(f"Matching by tile with {self._tiling.name} selection...")
                matches = self._match_by_tile(
                    img0,
                    img1,
                    features0,
                    features1,
                    method=self._tiling,
                )

        # If the fallback flag was set for any reason, try to match by tiles
        if fallback_flag:
            logger.debug(f"Fallback: matching by tile with {self._tiling.name} selection...")
            matches = self._match_by_tile(
                img0,
                img1,
                features0,
                features1,
                method=self._tiling,
            )
            timer_match.update("tile matching")

        # Save to h5 file
        raw_matches_path = matches_path.parent / "raw_matches.h5"
        with h5py.File(str(raw_matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            group.create_dataset(img1_name, data=matches)

        timer_match.update("save to h5")
        timer_match.print(f"{__class__.__name__} match")

        # Do Geometric verification
        # Rescale threshold according the image qualit
        if len(matches) < 8:
            logger.debug(f"Too few matches found ({len(matches)}). Skipping image pair {img0.name}-{img1.name}")
            return None

        scales = {
            Quality.HIGHEST: 1.0,
            Quality.HIGH: 1.0,
            Quality.MEDIUM: 1.5,
            Quality.LOW: 2.0,
            Quality.LOWEST: 3.0,
        }
        gv_threshold = self.config["general"]["gv_threshold"] * scales[self.config["general"]["quality"]]

        # Apply geometric verification
        _, inlMask = geometric_verification(
            kpts0=features0["keypoints"][matches[:, 0]],
            kpts1=features1["keypoints"][matches[:, 1]],
            method=self.config["general"]["geom_verification"],
            threshold=gv_threshold,
            confidence=self.config["general"]["gv_confidence"],
        )
        num_inliers = np.sum(inlMask)
        inliers_ratio = num_inliers / len(matches)
        matches = matches[inlMask]

        if num_inliers < self.min_inliers_per_pair:
            logger.debug(f"Too few inliers matches found ({num_inliers}). Skipping image pair {img0.name}-{img1.name}")
            timer_match.print(f"{__class__.__name__} match")
            return None
        elif inliers_ratio < self.min_inlier_ratio_per_pair:
            logger.debug(
                f"Too small inlier ratio ({inliers_ratio*100:.2f}%). Skipping image pair {img0.name}-{img1.name}"
            )
            timer_match.print(f"{__class__.__name__} match")
            return None
        timer_match.update("Geom. verification")

        # Save to h5 file
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            group.create_dataset(img1_name, data=matches)

        timer_match.update("save to h5")
        timer_match.print(f"{__class__.__name__} match")

        logger.debug(f"Matching {img0_name}-{img1_name} done!")

        # # For debugging
        if self.config["general"]["verbose"]:
            viz_dir = self.config["general"]["output_dir"] / "debug" / "matches"
            viz_dir.mkdir(parents=True, exist_ok=True)
            self.viz_matches(
                feature_path,
                matches_path,
                img0,
                img1,
                save_path=viz_dir / f"{img0_name}_{img1_name}.jpg",
                img_format="jpg",
                jpg_quality=70,
            )

        return matches

    def _match_by_tile(
        self,
        img0: Path,
        img1: Path,
        features0: FeaturesDict,
        features1: FeaturesDict,
        method: TileSelection = TileSelection.PRESELECTION,
        select_unique: bool = True,
    ) -> np.ndarray:
        """
        Match features between two images using a tiling approach.

        Args:
            img0 (Path): Path to the first image.
            img1 (Path): Path to the second image.
            features0 (FeaturesDict): Features of the first image.
            features1 (FeaturesDict): Features of the second image.
            method (TileSelection, optional): Tile selection method. Defaults to TileSelection.PRESELECTION.
            select_unique (bool, optional): Flag to select unique matches. Defaults to True.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.
        """

        timer = Timer(log_level="debug", cumulate_by_key=True)

        # Initialize empty matches array
        matches_full = np.array([], dtype=np.int64).reshape(0, 2)

        # Select tile pairs to match
        tile_pairs = tile_selection(
            img0,
            img1,
            method=method,
            quality=self.config["general"]["quality"],
            preselction_extractor=self._preselction_extractor,
            preselction_matcher=self._preselction_matcher,
            tile_size=self.config["general"]["tile_size"],
            tile_overlap=self.config["general"]["tile_overlap"],
            tile_preselection_size=self.tile_preselection_size,
            min_matches_per_tile=self.min_matches_per_tile,
            device=self._device,
        )
        timer.update("tile selection")

        # If no tile pairs are selected, return an empty array
        if len(tile_pairs) == 0:
            logger.debug("No tile pairs selected.")
            return matches_full

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logger.debug(f" - Matching tile pair ({tidx0}, {tidx1})")

            # Get features in tile and their ids in original array
            feats0_tile, idx0 = get_features_by_tile(features0, tidx0)
            feats1_tile, idx1 = get_features_by_tile(features1, tidx1)

            # Match features
            correspondences = self._match_pairs(feats0_tile, feats1_tile)
            logger.debug(f"     Found {len(correspondences)} matches")
            timer.update("match tile")

            # Restore original ids of the matched keypoints
            matches_orig = np.zeros_like(correspondences)
            matches_orig[:, 0] = idx0[correspondences[:, 0]]
            matches_orig[:, 1] = idx1[correspondences[:, 1]]
            matches_full = np.vstack((matches_full, matches_orig))

        # Select unique matches
        if select_unique is True:
            matches_full, idx, counts = np.unique(matches_full, axis=0, return_index=True, return_counts=True)
            if any(counts > 1):
                logger.warning(f"Found {sum(counts>1)} duplicate matches in tile pair ({tidx0}, {tidx1})")

        # Viz for debugging
        # if self.config["general"]["verbose"]:
        #     tile_match_dir = (
        #         Path(self.config["general"]["output_dir"])
        #         / "debug"
        #         / "matches_by_tile"
        #     )
        #     tile_match_dir.mkdir(parents=True, exist_ok=True)
        #     image0 = cv2.imread(str(img0))
        #     image1 = cv2.imread(str(img1))
        #     viz_matches_cv2(
        #         image0,
        #         image1,
        #         features0["keypoints"][matches_full[:, 0]],
        #         features1["keypoints"][matches_full[:, 1]],
        #         save_path=tile_match_dir / f"{img0.stem}-{img1.stem}.jpg",
        #         line_thickness=-1,
        #         autoresize=True,
        #         jpg_quality=60,
        #     )

        logger.debug("Matching by tile completed.")
        timer.print(f"{__class__.__name__} match_by_tile")

        return matches_full

    def viz_matches(
        self,
        feature_path: Path,
        matchings_path: Path,
        img0: Path,
        img1: Path,
        save_path: str = None,
        fast_viz: bool = True,
        interactive_viz: bool = False,
        **kwargs,
    ) -> None:
        # Check input parameters
        if not interactive_viz:
            assert save_path is not None, "output_dir must be specified if interactive_viz is False"
        if fast_viz:
            if interactive_viz:
                logger.warning("interactive_viz is ignored if fast_viz is True")
            assert save_path is not None, "output_dir must be specified if fast_viz is True"

        # Get config parameters
        interactive_viz = kwargs.get("interactive_viz", False)
        autoresize = kwargs.get("autoresize", True)
        max_long_edge = kwargs.get("max_long_edge", 1200)
        jpg_quality = kwargs.get("jpg_quality", 80)
        hide_matching_track = kwargs.get("hide_matching_track", False)

        img0 = Path(img0)
        img1 = Path(img1)
        img0_name = img0.name
        img1_name = img1.name

        # Load images
        image0 = load_image_np(img0, as_float=False, grayscale=True)
        image1 = load_image_np(img1, as_float=False, grayscale=True)

        # Load features and matches
        features0 = get_features(feature_path, img0_name)
        features1 = get_features(feature_path, img1_name)
        matches = get_matches(matchings_path, img0_name, img1_name)
        kpts0 = features0["keypoints"][matches[:, 0]]
        kpts1 = features1["keypoints"][matches[:, 1]]

        # Make visualization with OpenCV or Matplotlib
        if fast_viz:
            if hide_matching_track:
                line_thickness = -1
            else:
                line_thickness = 1

            viz_matches_cv2(
                image0,
                image1,
                kpts0,
                kpts1,
                str(save_path),
                line_thickness=line_thickness,
                autoresize=autoresize,
                max_long_edge=max_long_edge,
                jpg_quality=jpg_quality,
            )
        else:
            hide_fig = not interactive_viz
            if interactive_viz:
                viz_matches_mpl(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    hide_fig=hide_fig,
                    config=kwargs,
                )
            else:
                viz_matches_mpl(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    save_path,
                    hide_fig=hide_fig,
                    point_size=5,
                    config=kwargs,
                )


class DetectorFreeMatcherBase(metaclass=ABCMeta):
    """
    Base class for matchers. It defines the basic interface for matchers
    and basic functionalities that are shared among all matchers,
    in particular the `match` method. It must be subclassed to implement a new matcher.

    Attributes:
        default_general_conf (dict): Default configuration for general settings.
        _default_conf (dict): Default configuration for matcher-specific settings.
        required_inputs (list): List of required input parameters.
        min_inliers_per_pair (int): Minimum number of matches required.
        min_matches_per_tile (int): Minimum number of matches required per tile.
        max_feat_no_tiling (int): Maximum number of features without tiling.
        tile_preselection_size (int): Maximum resize dimension for preselection.
    """

    _default_general_conf = {
        "quality": Quality.LOW,
        "tile_selection": TileSelection.NONE,
        "tile_size": [1024, 1024],
        "tile_overlap": 0,
        "force_cpu": False,
        "do_viz": False,
        "min_inliers_per_pair": 15,
        "min_inlier_ratio_per_pair": 0.2,
        "min_matches_per_tile": 5,
    }
    _default_conf = {}
    required_inputs = []

    def __init__(self, custom_config: Config) -> None:
        """
        Initialize the MatcherBase with a custom config. This is the method to be called by subclasses

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
            "matcher": {
                **self._default_conf,
                **custom_config.matcher,
            },
        }
        # Get main processing parameters and save them as class members
        # NOTE: this is used for backward compatibility, it should be removed
        self._quality = self.config["general"]["quality"]
        self._tiling = self.config["general"]["tile_selection"]
        self.min_inliers_per_pair = self.config["general"]["min_inliers_per_pair"]
        self.min_matches_per_tile = self.config["general"]["min_matches_per_tile"]
        self.tile_preselection_size = self.config["general"]["tile_preselection_size"]

        logger.debug(f"Matching options: Tiling: {self._tiling.name}")
        logger.debug(f"Saving directory: {self.config['general']['output_dir']}")

        # Get device
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config["general"]["force_cpu"] else "cpu"
        )
        logger.debug(f"Running inference on device {self._device}")

        # Load extractor and matcher for the preselction
        if self.config["general"]["tile_selection"] == TileSelection.PRESELECTION:
            sp_cfg = {
                "nms_radius": 5,
                "max_keypoints": 4000,
                "keypoint_threshold": 0.005,
            }
            lg_cfg = {
                "features": "superpoint",
                "n_layers": 9,
                "depth_confidence": 0.9,
                "width_confidence": 0.95,
                "filter_threshold": 0.5,
                "flash": True,
            }
            self._preselction_extractor = SuperPoint(sp_cfg).eval().to(self._device)
            self._preselction_matcher = LightGlue(**lg_cfg).eval().to(self._device)
        else:
            self._preselction_extractor = None
            self._preselction_matcher = None

    def match(
        self,
        feature_path: Path,
        matches_path: Path,
        img0: Path,
        img1: Path,
        try_full_image: bool = False,
    ) -> np.ndarray:
        """
        Match features between two images.

        Args:
            feature_path (Path): Path to the feature file.
            matches_path (Path): Path to save the matches.
            img0 (Path): Path to the first image.
            img1 (Path): Path to the second image.
            try_full_image (bool, optional): Flag to attempt matching on full images. Defaults to False.

        Raises:
            RuntimeError: If there are too many features to match on full images.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.
        """

        timer_match = Timer(log_level="debug")

        # Check that feature_path exists
        if not Path(feature_path).exists():
            raise FileNotFoundError(f"Feature file {feature_path} does not exist.")
        else:
            self._feature_path = Path(feature_path)

        img0 = Path(img0)
        img1 = Path(img1)
        img0_name = img0.name
        img1_name = img1.name

        # Perform matching
        if self._tiling == TileSelection.NONE:
            matches = self._match_pairs(self._feature_path, img0, img1)
            timer_match.update("[match] Match full images")
        else:
            matches = self._match_by_tile(
                feature_path,
                img0,
                img1,
                method=self._tiling,
                select_unique=True,
            )
            timer_match.update("[match] Match by tile")

        # Do Geometric verification
        features0 = get_features(feature_path, img0_name)
        features1 = get_features(feature_path, img1_name)

        # Rescale threshold according the image original image size
        img_shape = cv2.imread(str(img0)).shape
        scale_fct = np.floor(max(img_shape) / self.max_tile_size / 2)
        gv_threshold = self.config["general"]["gv_threshold"] * scale_fct

        # Apply geometric verification
        _, inlMask = geometric_verification(
            kpts0=features0["keypoints"][matches[:, 0]],
            kpts1=features1["keypoints"][matches[:, 1]],
            method=self.config["general"]["geom_verification"],
            threshold=gv_threshold,
            confidence=self.config["general"]["gv_confidence"],
        )
        matches = matches[inlMask]
        timer_match.update("Geom. verification")

        # Save to h5 file
        n_matches = len(matches)
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            if n_matches >= self.min_inliers_per_pair:
                group.create_dataset(img1_name, data=matches)
            else:
                logger.debug(f"Too few matches found. Skipping image pair {img0.name}-{img1.name}")
                return None
        timer_match.update("save to h5")
        timer_match.print(f"{__class__.__name__} match")

        # For debugging
        # viz_dir = self.config["general"]["output_dir"] / "viz"
        # viz_dir.mkdir(parents=True, exist_ok=True)
        # self.viz_matches(
        #     feature_path,
        #     matches_path,
        #     img0,
        #     img1,
        #     save_path=viz_dir / f"{img0_name}_{img1_name}.png",
        # )

        logger.debug(f"Matching {img0_name}-{img1_name} done!")

        return matches

    @abstractmethod
    def _match_pairs(
        self,
        feature_path: Path,
        img0_path: Path,
        img1_path: Path,
    ):
        """
        Perform matching between two images using a detector-free matcher. It takes the path to two images, and returns the matches between keypoints and descriptors in those images. It also saves the updated features to the specified h5 file. This method must be implemented by subclasses.

        Args:
            feature_path (Path): Path to the h5 feature file where to save the updated features.
            img0_path (Path): Path to the first image.
            img1_path (Path): Path to the second image.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.

        Raises:
            NotImplementedError: Subclasses must implement _match_pairs() method.
            torch.cuda.OutOfMemoryError: If an out-of-memory error occurs while matching images.
        """

        raise NotImplementedError("Subclasses must implement _match_pairs() method.")

    @abstractmethod
    def _match_by_tile(
        self,
        feature_path: Path,
        img0: Path,
        img1: Path,
        method: TileSelection = TileSelection.PRESELECTION,
        select_unique: bool = True,
    ) -> np.ndarray:
        """
        Match features between two images using a tiling approach. This method must be implemented by subclasses.

        Args:
            img0 (Path): Path to the first image.
            img1 (Path): Path to the second image.
            features0 (FeaturesDict): Features of the first image.
            features1 (FeaturesDict): Features of the second image.
            method (TileSelection, optional): Tile selection method. Defaults to TileSelection.PRESELECTION.
            select_unique (bool, optional): Flag to select unique matches. Defaults to True.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.
        """
        raise NotImplementedError("Subclasses must implement _match_by_tile() method.")

    def _update_features_h5(
        self, feature_path, im0_name, im1_name, new_keypoints0, new_keypoints1, matches0
    ) -> np.ndarray:
        for i, im_name, new_keypoints in zip([0, 1], [im0_name, im1_name], [new_keypoints0, new_keypoints1]):
            features = get_features(feature_path, im_name)
            existing_keypoints = features["keypoints"]

            if len(existing_keypoints.shape) == 1:
                features["keypoints"] = new_keypoints
            else:
                n_exisiting_keypoints = existing_keypoints.shape[0]
                features["keypoints"] = np.vstack((existing_keypoints, new_keypoints))
                matches0[:, i] = matches0[:, i] + n_exisiting_keypoints

            with h5py.File(feature_path, "r+", libver="latest") as fd:
                del fd[im_name]
                grp = fd.create_group(im_name)
                for k, v in features.items():
                    if k == "im_path" or k == "feature_path":
                        grp.create_dataset(k, data=str(v))
                    if isinstance(v, np.ndarray):
                        grp.create_dataset(k, data=v)

        return matches0

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

    def _resize_keypoints(self, quality: Quality, keypoints: np.ndarray) -> np.ndarray:
        """
        Resize features based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            features0 (np.ndarray): The array of keypoints.

        Returns:
            np.ndarray: Resized keypoints.

        """
        return resize_keypoints(quality, keypoints)

    def _load_image_np(self, img_path: Path) -> np.ndarray:
        """
        Load image as numpy array.

        Args:
            img_path (Path): Path to the image.

        Returns:
            np.ndarray: The image as numpy array.
        """
        return load_image_np(img_path, self.as_float, self.grayscale)

    def viz_matches(
        self,
        feature_path: Path,
        matchings_path: Path,
        img0: Path,
        img1: Path,
        save_path: str = None,
        fast_viz: bool = True,
        interactive_viz: bool = False,
        **config,
    ) -> None:
        # Check input parameters
        if not interactive_viz:
            assert save_path is not None, "output_dir must be specified if interactive_viz is False"
        if fast_viz:
            if interactive_viz:
                logger.warning("interactive_viz is ignored if fast_viz is True")
            assert save_path is not None, "output_dir must be specified if fast_viz is True"

        img0 = Path(img0)
        img1 = Path(img1)
        img0_name = img0.name
        img1_name = img1.name

        # Load images
        image0 = load_image_np(img0, self.as_float, self.grayscale)
        image1 = load_image_np(img1, self.as_float, self.grayscale)

        # Load features and matches
        features0 = get_features(feature_path, img0_name)
        features1 = get_features(feature_path, img1_name)
        matches = get_matches(matchings_path, img0_name, img1_name)
        kpts0 = features0["keypoints"][matches[:, 0]]
        kpts1 = features1["keypoints"][matches[:, 1]]

        # Make visualization with OpenCV or Matplotlib
        if fast_viz:
            # Get config for OpenCV visualization
            autoresize = config.get("autoresize", True)
            max_long_edge = config.get("max_long_edge", 1200)
            jpg_quality = config.get("jpg_quality", 80)
            hide_matching_track = config.get("hide_matching_track", False)
            if hide_matching_track:
                line_thickness = -1
            else:
                line_thickness = 0.2

            viz_matches_cv2(
                image0,
                image1,
                kpts0,
                kpts1,
                str(save_path),
                line_thickness=line_thickness,
                autoresize=autoresize,
                max_long_edge=max_long_edge,
                jpg_quality=jpg_quality,
            )
        else:
            interactive_viz = config.get("interactive_viz", False)
            hide_fig = not interactive_viz
            if interactive_viz:
                viz_matches_mpl(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    hide_fig=hide_fig,
                    config=config,
                )
            else:
                viz_matches_mpl(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    save_path,
                    hide_fig=hide_fig,
                    point_size=5,
                    config=config,
                )


# Various util functions


def tile_selection(
    img0: Path,
    img1: Path,
    method: TileSelection,
    quality: Quality,
    preselction_extractor: ExtractorBase,
    preselction_matcher: MatcherBase,
    tile_size: Tuple[int, int],
    tile_overlap: int,
    tile_preselection_size: int = 1024,
    min_matches_per_tile: int = 5,
    do_geometric_verification: bool = True,
    device: str = "cpu",
):
    """
    Selects tile pairs for matching based on the specified method.

    Args:
        img0 (Path): Path to the first image.
        img1 (Path): Path to the second image.
        method (TileSelection, optional): Tile selection method. Defaults to TileSelection.PRESELECTION.

    Returns:
        List[Tuple[int, int]]: The selected tile pairs.
    """

    # Compute tiles limits and origin
    tiler = Tiler(tiling_mode="size")
    i0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    i1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Resize images to the specified quality to reproduce the same tiling as in feature extraction
    if quality != Quality.HIGH:
        i0_new_size = get_size_by_quality(quality, i0.shape[:2])
        i1_new_size = get_size_by_quality(quality, i1.shape[:2])
        i0 = resize_image(i0, (i0_new_size[1], i0_new_size[0]))
        i1 = resize_image(i1, (i1_new_size[1], i1_new_size[0]))
    else:
        i0_new_size = i0.shape[:2]
        i1_new_size = i1.shape[:2]

    # Compute tiles
    tiles0, t_orig0, t_padding0 = tiler.compute_tiles_by_size(input=i0, window_size=tile_size, overlap=tile_overlap)
    tiles1, t_orig1, t_padding1 = tiler.compute_tiles_by_size(input=i1, window_size=tile_size, overlap=tile_overlap)

    # Select tile selection method

    if method == TileSelection.EXHAUSTIVE:
        # Match all the tiles with all the tiles
        logger.debug("Matching tiles exaustively")
        tile_pairs = sorted(product(tiles0.keys(), tiles1.keys()))
    elif method == TileSelection.GRID:
        # Match tiles by regular grid
        logger.debug("Matching tiles by regular grid")
        tile_pairs = sorted(zip(tiles0.keys(), tiles1.keys()))
    elif method == TileSelection.PRESELECTION:
        # Match tiles by preselection running matching on downsampled images
        logger.debug("Matching tiles by downsampling preselection")

        # match downsampled images with roma
        from ..thirdparty.RoMa.roma import roma_outdoor

        n_matches = 5000
        matcher = roma_outdoor(device, coarse_res=448)
        H_A, W_A = i0_new_size
        H_B, W_B = i1_new_size
        warp, certainty = matcher.match(str(img0), str(img1), device=device)
        matches, certainty = matcher.sample(warp, certainty, num=n_matches)
        kp0, kp1 = matcher.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        kp0, kp1 = kp0.cpu().numpy(), kp1.cpu().numpy()

        # # Downsampled images
        # size0 = i0.shape[:2][::-1]
        # size1 = i1.shape[:2][::-1]
        # scale0 = tile_preselection_size / max(size0)
        # scale1 = tile_preselection_size / max(size1)
        # size0_new = tuple(int(round(x * scale0)) for x in size0)
        # size1_new = tuple(int(round(x * scale1)) for x in size1)
        # i0 = cv2.resize(i0, size0_new, interpolation=cv2.INTER_AREA)
        # i1 = cv2.resize(i1, size1_new, interpolation=cv2.INTER_AREA)

        # # Run SuperPoint on downsampled images
        # with torch.inference_mode():
        #     feats0 = preselction_extractor({"image": frame2tensor(i0, device)})
        #     feats1 = preselction_extractor({"image": frame2tensor(i1, device)})

        #     # Match features with LightGlue
        #     feats0 = sp2lg(feats0)
        #     feats1 = sp2lg(feats1)
        #     res = preselction_matcher({"image0": feats0, "image1": feats1})
        #     res = rbd2np(res)

        # # Get keypoints in original image
        # kp0 = feats0["keypoints"].cpu().numpy()[0]
        # kp0 = kp0[res["matches"][:, 0], :]
        # kp1 = feats1["keypoints"].cpu().numpy()[0]
        # kp1 = kp1[res["matches"][:, 1], :]

        # # Scale up keypoints
        # kp0 = kp0 / scale0
        # kp1 = kp1 / scale1

        # geometric verification
        if do_geometric_verification:
            _, inlMask = geometric_verification(
                kpts0=kp0,
                kpts1=kp1,
                threshold=5,
                confidence=0.9999,
                quiet=True,
            )
            kp0 = kp0[inlMask]
            kp1 = kp1[inlMask]

        # Select tile pairs where there are enough matches
        tile_pairs = set()
        all_pairs = sorted(product(tiles0.keys(), tiles1.keys()))
        for tidx0, tidx1 in all_pairs:
            ret0 = points_in_rect(kp0, get_tile_bounding_box(t_orig0[tidx0], tile_size))
            ret1 = points_in_rect(kp1, get_tile_bounding_box(t_orig1[tidx1], tile_size))
            n_matches = sum(ret0 & ret1)
            if n_matches > min_matches_per_tile:
                tile_pairs.add((tidx0, tidx1))
        tile_pairs = sorted(tile_pairs)

        # For Debugging...
        # if False:
        from matplotlib import pyplot as plt

        out_dir = Path("sandbox/preselection")
        out_dir.mkdir(parents=True, exist_ok=True)
        image0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
        image0 = resize_image(image0, (i0_new_size[1], i0_new_size[0]))
        image1 = resize_image(image1, (i1_new_size[1], i1_new_size[0]))
        c = "r"
        s = 5
        fig, axes = plt.subplots(1, 2)
        for ax, img, kp in zip(axes, [image0, image1], [kp0, kp1]):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR))
            ax.scatter(kp[:, 0], kp[:, 1], s=s, c=c)
            ax.axis("off")
            ax.set_aspect("equal")
        for lim0, lim1 in zip(t_orig0.values(), t_orig0.values()):
            axes[0].axvline(lim0[0])
            axes[0].axhline(lim0[1])
            axes[1].axvline(lim1[0])
            axes[1].axhline(lim1[1])
        axes[1].get_yaxis().set_visible(False)
        fig.tight_layout()
        # plt.show()
        fig.savefig(out_dir / f"{img0.name}-{img1.name}.jpg")
        plt.close()

    return tile_pairs


def load_image_np(img_path: Path, as_float: bool = True, grayscale: bool = False):
    image = cv2.imread(str(img_path))
    if as_float:
        image = image.astype(np.float32)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def resize_keypoints(quality: Quality, keypoints: np.ndarray) -> np.ndarray:
    """
    Resize keypoints based on the specified quality.

    Args:
        quality (Quality): The quality level for resizing.
        eypoints (np.ndarray): The array (nx2) of keypoints.

    Returns:
        np.ndarray: Resized keypoints.

    """
    if quality == Quality.HIGH:
        return keypoints
    if quality == Quality.HIGHEST:
        keypoints /= 2
    elif quality == Quality.MEDIUM:
        keypoints *= 2
    elif quality == Quality.LOW:
        keypoints *= 4
    elif quality == Quality.LOWEST:
        keypoints *= 8

    return keypoints


def get_features_by_tile(features: FeaturesDict, tile_idx: int):
    if "tile_idx" not in features:
        raise KeyError("tile_idx not found in features")
    pts_in_tile = features["tile_idx"] == tile_idx
    idx = np.where(pts_in_tile)[0]
    feat_tile = {
        "keypoints": features["keypoints"][pts_in_tile],
        "descriptors": features["descriptors"][:, pts_in_tile],
        "scores": features["scores"][pts_in_tile],
        "image_size": features["image_size"],
    }
    return (feat_tile, idx)


def frame2tensor(image: np.ndarray, device: str = "cpu"):
    if len(image.shape) == 2:
        image = image[None][None]
    elif len(image.shape) == 3:
        image = image.transpose(2, 0, 1)[None]
    return torch.tensor(image / 255.0, dtype=torch.float).to(device)


def get_tile_bounding_box(bottom_left, tile_size):
    return [
        bottom_left[0],
        bottom_left[1],
        bottom_left[0] + tile_size[0],
        bottom_left[1] + tile_size[1],
    ]


def points_in_rect(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
    logic = np.all(points > rect[:2], axis=1) & np.all(points < rect[2:], axis=1)
    return logic


def sp2lg(feats: dict) -> dict:
    feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
    if feats["descriptors"].shape[-1] != 256:
        feats["descriptors"] = feats["descriptors"].T
    feats = {k: v[None] for k, v in feats.items()}
    return feats


def rbd2np(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {k: v[0].cpu().numpy() if isinstance(v, (torch.Tensor, np.ndarray, list)) else v for k, v in data.items()}
