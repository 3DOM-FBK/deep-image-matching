import logging
from copy import deepcopy
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import cv2
import h5py
import numpy as np
import torch

from .consts import Quality, TileSelection
from .io.h5 import get_features
from .tiling import Tiler
from .visualization import viz_matches_cv2, viz_matches_mpl

logger = logging.getLogger(__name__)

MIN_MATCHES = 20


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray]
    tile_idx: Optional[np.ndarray]


DEFAULT_CONFIG = {
    "general": {
        "tile_selection": TileSelection.NONE,
        "max_keypoints": 4096,
        "force_cpu": False,
        "save_dir": "results",
        "do_viz": False,
        "fast_viz": True,
        # "interactive_viz": False,
        "hide_matching_track": True,
        "do_viz_tiles": False,
        "tiling_grid": [1, 1],
        "tiling_overlap": 0,
        "min_matches_per_tile": 5,
    }
}


class MatcherBase:
    def __init__(self, **custom_config) -> None:
        """
        Base class for matchers. It defines the basic interface for matchers and basic functionalities that are shared among all matchers, in particular the `match` method. It must be subclassed to implement a new matcher.

        Args:
            opt (dict): Options for the matcher.

        Raises:
            TypeError: If `opt` is not a dictionary.
        """

        # Set default config
        self._config = DEFAULT_CONFIG

        # If a custom config is passed, update the default config
        if not isinstance(custom_config, dict):
            raise TypeError("config must be a dictionary")
        self._update_config(custom_config)

        # All features detected on image 0 (FeaturesBase object with N keypoints)
        self._features0 = None

        # All features detected on image 1 (FeaturesBase object with M keypoints)
        self._features1 = None

        # Index of the matches in both the images. The first column contains the index of the mathced keypoints in features0.keypoints, the second column contains the index of the matched keypoints in features1.keypoints (Kx2 array)
        self._matches01 = None

    @property
    def features0(self):
        return self._features0

    @property
    def features1(self):
        return self._features1

    @property
    def matches0(self):
        return self._matches0

    def _match_pairs(
        self,
        feats0: dict,
        feats1: dict,
    ) -> np.ndarray:
        """
        _match_pairs _summary_

        Args:
            feats0 (dict): _description_
            feats1 (dict): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """
        raise NotImplementedError(
            "Subclasses must implement _match_full_images() method."
        )

    # @timeit
    def match(
        self,
        feature_path: Path,
        img0: Path,
        img1: Path,
        **custom_config,
    ) -> np.ndarray:
        """ """
        # If a custom config is passed, update the default config
        self._update_config(custom_config)

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

        # Testing PRESELECTION
        # self._matches = self._match_by_tile(
        #     img0,
        #     img1,
        #     method=TileSelection.PRESELECTION,
        # )

        # Perform matching (on tiles or full images)
        # If the features are not too many, try first to match all features together on full image, if it fails, try to match by tiles
        max_feats = 200000
        high_feats_flag = (
            len(self._features0["keypoints"]) > max_feats
            or len(self._features1["keypoints"]) > max_feats
        )
        if self._tiling == TileSelection.NONE:
            logger.debug(
                f"Tile selection was {self._tiling.name}. Matching full images..."
            )
            if high_feats_flag:
                raise RuntimeError(
                    "Too many features to match full images. Try running the matching with tile selection or use a lower max_keypoints value."
                )
            self._matches = self._match_pairs(self._features0, self._features1)
        else:
            if not high_feats_flag:
                try:
                    logger.debug(
                        f"Tile selection was {self._tiling.name}, but features are less then {max_feats}. Trying to match full images..."
                    )
                    self._matches = self._match_pairs(self._features0, self._features1)
                except Exception as e:
                    logger.warning(
                        f"Matching full images failed: {e}. Trying to match by tiles..."
                    )
                    self._matches = self._match_by_tile(
                        img0,
                        img1,
                        method=self._tiling,
                    )
            else:
                logger.debug(
                    f"Tile selection was {self._tiling.name} and features are more than {max_feats}. Matching by tile with {self._tiling.name} selection..."
                )
                self._matches = self._match_by_tile(
                    img0,
                    img1,
                    method=self._tiling,
                )

        # Save to h5 file
        n_matches = len(self._matches)
        matches_path = self._save_dir / "matches.h5"
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            if n_matches >= MIN_MATCHES:
                group.create_dataset(img1_name, data=self._matches)

        logger.debug(f"Matching {img0_name}-{img1_name} done!")

        # Visualize matches (temporarily disabled)
        # if self._config["general"]["do_viz"] is True:
        #     self.viz_matches(
        #         image0,
        #         image1,
        #         self._mkpts0,
        #         self._mkpts1,
        #         str(self._save_dir / "matches.jpg"),
        #         fast_viz=self._config["general"]["fast_viz"],
        #         hide_matching_track=self._config["general"]["hide_matching_track"],
        #     )

        return self._matches

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
            "save_dir",
            "force_cpu",
            "hide_matching_track",
            "tile_selection",
            "tiling_grid",
            "tiling_overlap",
            "do_viz",
            "fast_viz",
            "do_viz_tiles",
            "min_matches_per_tile",
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
        self._tiling = self._config["general"]["tile_selection"]
        logger.debug(f"Matching options: Tiling: {self._tiling.name}")

        # Define saving directory
        save_dir = self._config["general"]["save_dir"]
        if save_dir is not None:
            self._save_dir = Path(save_dir)
            self._save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._save_dir = None
        logger.debug(f"Saving directory: {self._save_dir}")

        # Get device
        self._device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not self._config["general"]["force_cpu"]
            else "cpu"
        )
        logger.debug(f"Running inference on device {self._device}")

    def _match_by_tile(
        self,
        img0: Path,
        img1: Path,
        method: TileSelection = TileSelection.PRESELECTION,
    ):
        # Get features from h5 file
        features0 = get_features(self._feature_path, img0.name)
        features1 = get_features(self._feature_path, img1.name)

        # Compute tiles limits and origin
        grid = self._config["general"]["tiling_grid"]
        overlap = self._config["general"]["tiling_overlap"]
        do_viz_tiles = self._config["general"]["do_viz_tiles"]
        tiler = Tiler(grid=grid, overlap=overlap)
        image0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        t0_lims, t0_origin = tiler.compute_limits_by_grid(image0)
        t1_lims, t1_origin = tiler.compute_limits_by_grid(image1)

        # Select tile pairs to match
        tile_pairs = self._tile_selection(image0, image1, t0_lims, t1_lims, method)

        # Match each tile pair
        matches_full = np.array([], dtype=np.int64).reshape(0, 2)
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
                }
                return (feat_tile, idx)

            # Get features in tile and their ids in original array
            feats0_tile, idx0 = get_features_by_tile(features0, tidx0)
            feats1_tile, idx1 = get_features_by_tile(features1, tidx1)

            # Match features
            correspondences = self._match_pairs(feats0_tile, feats1_tile)

            # Restore original ids of the matched keypoints
            matches_orig = np.zeros_like(correspondences)
            matches_orig[:, 0] = idx0[correspondences[:, 0]]
            matches_orig[:, 1] = idx1[correspondences[:, 1]]
            matches_full = np.vstack((matches_full, matches_orig))

            # # Visualize matches on tile
            # if do_viz_tiles is True:
            #     out_img_path = str(self._save_dir / f"matches_tile_{tidx0}-{tidx1}.jpg")
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

        return matches_full

    def _tile_selection(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        t0_lims: dict[int, np.ndarray],
        t1_lims: dict[int, np.ndarray],
        method: TileSelection = TileSelection.PRESELECTION,
    ) -> List[Tuple[int, int]]:
        """
        Selects tile pairs for matching based on the specified method.

        Args:
            image0 (np.ndarray): The first image.
            image1 (np.ndarray): The second image.
            t0_lims (dict[int, np.ndarray]): The limits of tiles in image0.
            t1_lims (dict[int, np.ndarray]): The limits of tiles in image1.
            method (TileSelection, optional): The tile selection method. Defaults to TileSelection.PRESELECTION.

        Returns:
            List[Tuple[int, int]]: The selected tile pairs.

        """

        def points_in_rect(points: np.ndarray, rect: np.ndarray) -> np.ndarray:
            logic = np.all(points > rect[:2], axis=1) & np.all(
                points < rect[2:], axis=1
            )
            return logic

        # print(config)
        # local_feat_extractor = config["config"].get("local_feat_extractor")
        # print(local_feat_extractor)

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
            logger.debug("Matching tiles by preselection tile selection")
            if image0.shape[0] > 8000:
                n_down = 4
            if image0.shape[0] > 4000:
                n_down = 3
            elif image0.shape[0] > 2000:
                n_down = 2
            else:
                n_down = 1

            # Downsampled images
            i0 = deepcopy(image0)
            i1 = deepcopy(image1)
            for _ in range(n_down):
                i0 = cv2.pyrDown(i0)
                i1 = cv2.pyrDown(i1)

            # Import superpoint and lightglue
            SP = import_module("deep_image_matching.hloc.extractors.superpoint")
            LG = import_module("deep_image_matching.thirdparty.LightGlue.lightglue")

            def sp2lg(feats: dict) -> dict:
                feats = {
                    k: v[0] if isinstance(v, (list, tuple)) else v
                    for k, v in feats.items()
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

            # Run SuperPoint on downsampled images
            with torch.no_grad():
                SP_cfg = {"max_keypoints": 1024}
                SP_extractor = SP.SuperPoint(SP_cfg).eval().to(self._device)
                feats0 = SP_extractor({"image": self._frame2tensor(i0, self._device)})
                feats1 = SP_extractor({"image": self._frame2tensor(i1, self._device)})

                # Match features with LightGlue
                LG_matcher = LG.LightGlue("superpoint").eval().to(self._device)
                feats0 = sp2lg(feats0)
                feats1 = sp2lg(feats1)
                res = LG_matcher({"image0": feats0, "image1": feats1})
                res = rbd2np(res)

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
                    str(self._save_dir / "tile_preselection.jpg"),
                    fast_viz=True,
                    hide_matching_track=True,
                    autoresize=True,
                    max_long_edge=1200,
                )

            for _ in range(n_down):
                kp0 *= 2
                kp1 *= 2

            # Select tile pairs where there are enough matches
            tile_pairs = []
            all_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
            for tidx0, tidx1 in all_pairs:
                lim0 = t0_lims[tidx0]
                lim1 = t1_lims[tidx1]
                ret0 = points_in_rect(kp0, lim0)
                ret1 = points_in_rect(kp1, lim1)
                ret = ret0 & ret1
                if sum(ret) > self._config["general"]["min_matches_per_tile"]:
                    tile_pairs.append((tidx0, tidx1))

            # For Debugging...
            # if False:
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
            #     fig.savefig("preselection.jpg")
            #     plt.close()

        return tile_pairs

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu"):
        if len(image.shape) == 2:
            image = image[None][None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)[None]
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)

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
            ), "save_dir must be specified if interactive_viz is False"

        # Check input parameters
        if fast_viz:
            if interactive_viz:
                logger.warning("interactive_viz is ignored if fast_viz is True")
            assert (
                save_path is not None
            ), "save_dir must be specified if fast_viz is True"

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
