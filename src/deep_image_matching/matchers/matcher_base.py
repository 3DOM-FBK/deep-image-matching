import inspect
from abc import ABCMeta, abstractmethod
from itertools import product
from pathlib import Path
from typing import Optional, TypedDict

import cv2
import h5py
import numpy as np
import torch

from .. import Timer, logger
from ..hloc.extractors.superpoint import SuperPoint
from ..io.h5 import get_features
from ..thirdparty.LightGlue.lightglue import LightGlue
from ..utils.consts import Quality, TileSelection
from ..utils.geometric_verification import geometric_verification
from ..utils.tiling import Tiler
from ..visualization import viz_matches_cv2, viz_matches_mpl


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
    classes = [c for c in classes if issubclass(c[1], MatcherBase)]
    assert len(classes) == 1, classes
    return classes[0][1]


# NOTE: The MatcherBase class should contain all the common methods and attributes for all the matchers and must be used as a base class.
# The specific matchers MUST contain at least the `_match_pairs` method, which takes in two images as Numpy arrays, and returns the matches between keypoints and descriptors in those images. It doesn not care if the images are tiles or full-res images, as the tiling is handled by the MatcherBase class that calls the `_match_pairs` method for each tile pair or for the full images depending on the tile selection method.


class MatcherBase(metaclass=ABCMeta):
    """
    Base class for matchers. It defines the basic interface for matchers
    and basic functionalities that are shared among all matchers,
    in particular the `match` method. It must be subclassed to implement a new matcher.

    Attributes:
        general_conf (dict): Default configuration for general settings.
        default_conf (dict): Default configuration for matcher-specific settings.
        required_inputs (list): List of required input parameters.
        min_matches (int): Minimum number of matches required.
        min_matches_per_tile (int): Minimum number of matches required per tile.
        max_feat_no_tiling (int): Maximum number of features without tiling.
        preselection_size_max (int): Maximum resize dimension for preselection.
    """

    general_conf = {
        "output_dir": None,
        "quality": Quality.HIGH,
        "tile_selection": TileSelection.NONE,
        "tiling_grid": [1, 1],
        "tiling_overlap": 0,
        "force_cpu": False,
        "do_viz": False,
    }
    default_conf = {}
    required_inputs = []
    min_matches = 20
    min_matches_per_tile = 5
    max_feat_no_tiling = 100000
    preselection_size_max = 1024

    def __init__(self, custom_config) -> None:
        """
        Initializes the MatcherBase object.

        Args:
            custom_config (dict): Options for the matcher.

        Raises:
            TypeError: If `custom_config` is not a dictionary.
        """

        # If a custom config is passed, update the default config
        if not isinstance(custom_config, dict):
            raise TypeError("opt must be a dictionary")

        # Update default config
        self._config = {
            "general": {
                **self.general_conf,
                **custom_config.get("general", {}),
            },
            "matcher": {
                **self.default_conf,
                **custom_config.get("matcher", {}),
            },
        }

        if "min_matches" in custom_config["general"]:
            self.min_matches = custom_config["general"]["min_matches"]
        if "min_matches_per_tile" in custom_config["general"]:
            self.min_matches_per_tile = custom_config["general"]["min_matches_per_tile"]
        if "preselection_size_max" in custom_config["general"]:
            self.preselection_size_max = custom_config["general"][
                "preselection_size_max"
            ]

        # Get main processing parameters and save them as class members
        self._tiling = self._config["general"]["tile_selection"]
        logger.debug(f"Matching options: Tiling: {self._tiling.name}")

        # Define saving directory
        output_dir = self._config["general"]["output_dir"]
        if output_dir is not None:
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._output_dir = None
        logger.debug(f"Saving directory: {self._output_dir}")

        # Get device
        self._device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not self._config["general"]["force_cpu"]
            else "cpu"
        )
        logger.debug(f"Running inference on device {self._device}")

        # All features detected on image 0 (FeaturesBase object with N keypoints)
        self._features0 = None

        # All features detected on image 1 (FeaturesBase object with M keypoints)
        self._features1 = None

        # Index of the matches in both the images. The first column contains the index of the mathced keypoints in features0.keypoints, the second column contains the index of the matched keypoints in features1.keypoints (Kx2 array)
        self._matches01 = None

        # Load extractor and matcher for the preselction
        if self._config["general"]["tile_selection"] == TileSelection.PRESELECTION:
            self._preselction_extractor = (
                SuperPoint({"max_keypoints": 2048}).eval().to(self._device)
            )
            self._preselction_matcher = (
                LightGlue(
                    features="superpoint",
                    n_layers=7,
                    depth_confidence=0.9,
                    width_confidence=0.95,
                    flash=False,
                )
                .eval()
                .to(self._device)
            )

    @property
    def features0(self):
        return self._features0

    @property
    def features1(self):
        return self._features1

    @property
    def matches0(self):
        return self._matches0

    @abstractmethod
    def _match_pairs(
        self,
        feats0: dict,
        feats1: dict,
    ) -> np.ndarray:
        """
        Perform matching between two sets of features.

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
        self._features0 = get_features(self._feature_path, img0.name)
        self._features1 = get_features(self._feature_path, img1.name)
        timer_match.update("load h5 features")

        # Perform matching (on tiles or full images)
        # If the features are not too many, try first to match all features together on full image, if it fails, try to match by tiles
        fallback_flag = False
        high_feats_flag = (
            len(self._features0["keypoints"]) > self.max_feat_no_tiling
            or len(self._features1["keypoints"]) > self.max_feat_no_tiling
        )

        if self._tiling == TileSelection.NONE:
            if high_feats_flag:
                raise RuntimeError(
                    "Too many features to run the matching on full images. Try running the matching with tile selection or use a lower max_keypoints value."
                )
            else:
                try_full_image = True

        # Try to match full images first
        if try_full_image and not high_feats_flag:
            try:
                logger.debug(
                    f"Tile selection was {self._tiling.name}. Matching full images..."
                )
                self._matches = self._match_pairs(self._features0, self._features1)
                timer_match.update("[match] try to match full images")
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(
                        f"Matching full images failed: {e}. \nTrying to match by tiles..."
                    )
                    fallback_flag = True
                else:
                    raise e

        # If try_full_image is disabled or matching full images failed, match by tiles
        if not try_full_image or fallback_flag:
            logger.debug(f"Matching by tile with {self._tiling.name} selection...")
            self._matches = self._match_by_tile(
                img0,
                img1,
                self._features0,
                self._features1,
                method=self._tiling,
            )
            timer_match.update("tile matching")

        # Save to h5 file
        n_matches = len(self._matches)
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            if n_matches >= self.min_matches:
                group.create_dataset(
                    img1_name, data=self._matches
                )  # or use require_dataset
            else:
                logger.debug(
                    f"Too few matches found. Skipping image pair {img0.name}-{img1.name}"
                )
                return None
        timer_match.update("save to h5")
        timer_match.print(f"{__class__.__name__} match")

        logger.debug(f"Matching {img0_name}-{img1_name} done!")

        # Visualize matches (temporarily disabled)
        # if self._config["general"]["do_viz"] is True:
        #     self.viz_matches(
        #         image0,
        #         image1,
        #         self._mkpts0,
        #         self._mkpts1,
        #         str(self._output_dir / "matches.jpg"),
        #         fast_viz=self._config["general"]["fast_viz"],
        #         hide_matching_track=self._config["general"]["hide_matching_track"],
        #     )

        return self._matches

    def _match_by_tile(
        self,
        img0: Path,
        img1: Path,
        features0: FeaturesDict,
        features1: FeaturesDict,
        method: TileSelection = TileSelection.PRESELECTION,
        select_unique: bool = True,
    ):
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
        tile_pairs = self._tile_selection(img0, img1, method)
        timer.update("tile selection")

        # If no tile pairs are selected, return an empty array
        if len(tile_pairs) == 0:
            logger.debug("No tile pairs selected.")
            return matches_full

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logger.debug(f"  - Matching tile pair ({tidx0}, {tidx1})")

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

            # Get features in tile and their ids in original array
            feats0_tile, idx0 = get_features_by_tile(features0, tidx0)
            feats1_tile, idx1 = get_features_by_tile(features1, tidx1)

            # Match features
            correspondences = self._match_pairs(feats0_tile, feats1_tile)
            timer.update("match tile")

            # Restore original ids of the matched keypoints
            matches_orig = np.zeros_like(correspondences)
            matches_orig[:, 0] = idx0[correspondences[:, 0]]
            matches_orig[:, 1] = idx1[correspondences[:, 1]]
            matches_full = np.vstack((matches_full, matches_orig))

        # Select unique matches
        if select_unique is True:
            matches_full, idx, counts = np.unique(
                matches_full, axis=0, return_index=True, return_counts=True
            )
            if any(counts > 1):
                logger.warning(
                    f"Found {sum(counts>1)} duplicate matches in tile pair ({tidx0}, {tidx1})"
                )

        # # Visualize matches on tile
        # if do_viz_tiles is True:
        #     out_img_path = str(self._output_dir / f"matches_tile_{tidx0}-{tidx1}.jpg")
        #     self.viz_matches(
        #         tile0,
        #         tile1,
        #         mkpts0,
        #         mkpts1,
        #         out_img_path,
        #         fast_viz=True,
        #         hide_matching_track=True,
        #         autoresize=True,
        #         max_long_edge=1200,
        #     )

        logger.debug("Matching by tile completed.")
        timer.print(f"{__class__.__name__} match_by_tile")

        return matches_full

    def _tile_selection(
        self,
        img0: Path,
        img1: Path,
        method: TileSelection = TileSelection.PRESELECTION,
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

        def frame2tensor(image: np.ndarray, device: str = "cpu"):
            if len(image.shape) == 2:
                image = image[None][None]
            elif len(image.shape) == 3:
                image = image.transpose(2, 0, 1)[None]
            return torch.tensor(image / 255.0, dtype=torch.float).to(device)

        def points_in_rect(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
            logic = np.all(points > rect[:2], axis=1) & np.all(
                points < rect[2:], axis=1
            )
            return logic

        def sp2lg(feats: dict) -> dict:
            feats = {
                k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()
            }
            if feats["descriptors"].shape[-1] != 256:
                feats["descriptors"] = feats["descriptors"].T
            feats = {k: v[None] for k, v in feats.items()}
            return feats

        def rbd2np(data: dict) -> dict:
            """Remove batch dimension from elements in data"""
            return {
                k: v[0].cpu().numpy()
                if isinstance(v, (torch.Tensor, np.ndarray, list))
                else v
                for k, v in data.items()
            }

        # Compute tiles limits and origin
        grid = self._config["general"]["tiling_grid"]
        overlap = self._config["general"]["tiling_overlap"]
        tiler = Tiler(grid=grid, overlap=overlap)
        i0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        i1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        t0_lims, _ = tiler.compute_limits_by_grid(i0)
        t1_lims, _ = tiler.compute_limits_by_grid(i1)

        # Select tile selection method
        if method == TileSelection.EXHAUSTIVE:
            # Match all the tiles with all the tiles
            logger.debug("Matching tiles exaustively")
            tile_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.GRID:
            # Match tiles by regular grid
            logger.debug("Matching tiles by regular grid")
            tile_pairs = sorted(zip(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.PRESELECTION:
            # Match tiles by preselection running matching on downsampled images
            logger.debug("Matching tiles by downsampling preselection")

            # Downsampled images
            max_len = self.preselection_size_max
            size0 = i0.shape[:2][::-1]
            size1 = i1.shape[:2][::-1]
            scale0 = max_len / max(size0)
            scale1 = max_len / max(size1)
            size0_new = tuple(int(round(x * scale0)) for x in size0)
            size1_new = tuple(int(round(x * scale1)) for x in size1)
            i0 = cv2.resize(i0, size0_new, interpolation=cv2.INTER_AREA)
            i1 = cv2.resize(i1, size1_new, interpolation=cv2.INTER_AREA)

            # Run SuperPoint on downsampled images
            with torch.no_grad():
                feats0 = self._preselction_extractor(
                    {"image": frame2tensor(i0, self._device)}
                )
                feats1 = self._preselction_extractor(
                    {"image": frame2tensor(i1, self._device)}
                )

                # Match features with LightGlue
                feats0 = sp2lg(feats0)
                feats1 = sp2lg(feats1)
                res = self._preselction_matcher({"image0": feats0, "image1": feats1})
                res = rbd2np(res)

            # TODO: implement a better way to prune matching pair
            # If not enough matches, return None and quit the matching
            if len(res["matches"]) < self.min_matches:
                return []

            # Get keypoints in original image
            kp0 = feats0["keypoints"].cpu().numpy()[0]
            kp0 = kp0[res["matches"][:, 0], :]
            kp1 = feats1["keypoints"].cpu().numpy()[0]
            kp1 = kp1[res["matches"][:, 1], :]

            if self._config["general"]["do_viz"] is True:
                self.viz_matches(
                    i0,
                    i1,
                    kp0,
                    kp1,
                    str(self._output_dir / "tile_preselection.jpg"),
                    fast_viz=True,
                    hide_matching_track=True,
                    autoresize=True,
                    max_long_edge=1200,
                )

            # Scale up keypoints
            kp0 = kp0 / scale0
            kp1 = kp1 / scale1

            # geometric verification
            _, inlMask = geometric_verification(kpts0=kp0, kpts1=kp1, threshold=2)
            kp0 = kp0[inlMask]
            kp1 = kp1[inlMask]

            # Select tile pairs where there are enough matches
            tile_pairs = []
            all_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
            for tidx0, tidx1 in all_pairs:
                lim0 = t0_lims[tidx0]
                lim1 = t1_lims[tidx1]
                ret0 = points_in_rect(kp0, lim0)
                ret1 = points_in_rect(kp1, lim1)
                ret = ret0 & ret1
                if sum(ret) > self.min_matches_per_tile:
                    tile_pairs.append((tidx0, tidx1))

            # For Debugging...
            # if False:
            #     from matplotlib import pyplot as plt

            #     out_dir = Path("sandbox/preselection")
            #     out_dir.mkdir(parents=True, exist_ok=True)
            #     image0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE)
            #     image1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
            #     c = "r"
            #     s = 5
            #     fig, axes = plt.subplots(1, 2)
            #     for ax, img, kp in zip(axes, [image0, image1], [kp0, kp1]):
            #         ax.imshow(cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR))
            #         ax.scatter(kp[:, 0], kp[:, 1], s=s, c=c)
            #         ax.axis("off")
            #     for lim0, lim1 in zip(t0_lims.values(), t1_lims.values()):
            #         axes[0].axvline(lim0[0])
            #         axes[0].axhline(lim0[1])
            #         axes[1].axvline(lim1[0])
            #         axes[1].axhline(lim1[1])
            #     # axes[1].get_yaxis().set_visible(False)
            #     fig.tight_layout()
            #     plt.show()
            #     fig.savefig(out_dir / f"{img0.name}-{img1.name}.jpg")
            #     plt.close()

        return tile_pairs

    def viz_matches(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        save_path: str = None,
        fast_viz: bool = True,
        interactive_viz: bool = False,
        **config,
    ) -> None:
        if not interactive_viz:
            assert (
                save_path is not None
            ), "output_dir must be specified if interactive_viz is False"

        # Check input parameters
        if fast_viz:
            if interactive_viz:
                logger.warning("interactive_viz is ignored if fast_viz is True")
            assert (
                save_path is not None
            ), "output_dir must be specified if fast_viz is True"

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
