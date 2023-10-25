import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union, TypedDict, Optional
import h5py

import cv2
import numpy as np
import torch


from .io.h5 import get_features
from .consts import GeometricVerification, Quality, TileSelection
from .tiling import Tiler
from .visualization import viz_matches_cv2, viz_matches_mpl

logger = logging.getLogger(__name__)


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray]
    tile_idx: Optional[np.ndarray]


DEFAULT_CONFIG = {
    "general": {
        "quality": Quality.HIGH,
        "tile_selection": TileSelection.NONE,
        "geometric_verification": GeometricVerification.NONE,
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
            feature_path = Path(feature_path)

        # Check if images exist
        if not Path(img0).exists():
            raise FileNotFoundError(f"Image {img0} does not exist.")
        if not Path(img1).exists():
            raise FileNotFoundError(f"Image {img1} does not exist.")
        img0_name = img0.name
        img1_name = img1.name

        # Get features from h5 file
        feature_path = Path(self._config["general"]["save_dir"]) / "features.h5"
        features0 = get_features(feature_path, img0.name)
        features1 = get_features(feature_path, img1.name)

        # Perform matching (on tiles or full images)
        if self._tiling == TileSelection.NONE:
            logger.info("Matching full images...")
            matches = self._match_pairs(features0, features1)

        else:
            logger.info("Matching by tiles...")
            matches = self._match_by_tile(
                img0,
                img1,
            )

        # Save to h5 file
        min_matches = 10
        n_matches = len(matches)

        matches_path = self._save_dir / "matches.h5"
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            if n_matches >= min_matches:
                group.create_dataset(img1_name, data=matches)

        logger.info("Matching done!")

        # Perform geometric verification (temporarily disabled)
        # logger.info("Performing geometric verification...")
        # if self._gv is not GeometricVerification.NONE:
        #     F, inlMask = geometric_verification(
        #         self._mkpts0,
        #         self._mkpts1,
        #         method=self._gv,
        #         confidence=self._config["general"]["gv_confidence"],
        #         threshold=self._config["general"]["gv_threshold"],
        #     )
        #     self._F = F
        #     self._filter_matches_by_mask(inlMask)
        #     logger.info("Geometric verification done.")
        #     self.timer.update("geometric_verification")

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

        # Save matches as txt file (temporarily disabled)
        # if self._save_dir is not None:
        #     self.save_mkpts_as_txt(self._save_dir)

        return matches

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
            # "quality",
            "tile_selection",
            "force_cpu",
            "save_dir",
            "do_viz",
            "fast_viz",
            "hide_matching_track",
            "do_viz_tiles",
            "tiling_grid",
            "tiling_overlap",
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
        if not isinstance(
            new_config["general"]["geometric_verification"], GeometricVerification
        ):
            raise TypeError(
                "geometric_verification must be a GeometricVerification enum"
            )

        # Update the current config with the custom config
        self._config = new_config

        # Get main processing parameters and save them as class members
        self._quality = self._config["general"]["quality"]
        self._tiling = self._config["general"]["tile_selection"]
        self._gv = self._config["general"]["geometric_verification"]
        logger.info(
            f"Matching options: Quality: {self._quality.name} - Tiling: {self._tiling.name} - Geometric Verification: {self._gv.name}"
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
        self._device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not self._config["general"]["force_cpu"]
            else "cpu"
        )
        logger.info(f"Running inference on device {self._device}")

    def _match_by_tile(
        self,
        img0: Path,
        img1: Path,
    ):
        # Get features from h5 file
        feature_path = Path(self._config["general"]["save_dir"]) / "features.h5"
        features0 = get_features(feature_path, img0.name)
        features1 = get_features(feature_path, img1.name)

        # Compute tiles limits and origin
        tile_selection = self._tiling
        grid = self._config["general"]["tiling_grid"]
        overlap = self._config["general"]["tiling_overlap"]
        do_viz_tiles = self._config["general"]["do_viz_tiles"]
        tiler = Tiler(grid=grid, overlap=overlap)
        image0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        t0_lims, t0_origin = tiler.compute_limits_by_grid(image0)
        t1_lims, t1_origin = tiler.compute_limits_by_grid(image1)

        # Select tile pairs to match
        tile_pairs = self._tile_selection(
            image0, image1, t0_lims, t1_lims, tile_selection
        )

        # Initialize empty array for storing matched keypoints, descriptors and scores
        matches_full = np.array([], dtype=np.int64).reshape(0, 2)

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
                }
                return (feat_tile, idx)

            # Get features in tile and their ids in original array
            feats0_tile, idx0 = get_features_by_tile(features0, tidx0)
            feats1_tile, idx1 = get_features_by_tile(features1, tidx1)

            # Match features
            matches = self._match_pairs(feats0_tile, feats1_tile)

            # Restore original ids of the matched keypoints
            matches_orig = np.zeros_like(matches)
            matches_orig[:, 0] = idx0[matches[:, 0]]
            matches_orig[:, 1] = idx1[matches[:, 1]]
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

        logger.info("Matching by tile done.")

        return matches_full

    def _frame2tensor(self, image: np.ndarray) -> torch.Tensor:
        """Normalize the image tensor and add batch dimension."""

        device = torch.device(self._device if torch.cuda.is_available() else "cpu")
        return torch.tensor(image / 255.0, dtype=torch.float)[None, None].to(device)

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
            logger.info("Matching tiles exaustively")
            tile_pairs = sorted(product(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.GRID:
            # Match tiles by regular grid
            logger.info("Matching tiles by regular grid")
            tile_pairs = sorted(zip(t0_lims.keys(), t1_lims.keys()))
        elif method == TileSelection.PRESELECTION:
            # Match tiles by preselection running matching on downsampled images
            logger.info("Matching tiles by preselection tile selection")
            if image0.shape[0] > 8000:
                n_down = 4
            if image0.shape[0] > 4000:
                n_down = 3
            elif image0.shape[0] > 2000:
                n_down = 2
            else:
                n_down = 1

            # Run inference on downsampled images
            i0 = deepcopy(image0)
            i1 = deepcopy(image1)
            for _ in range(n_down):
                i0 = cv2.pyrDown(i0)
                i1 = cv2.pyrDown(i1)
            # conf_for_downsamp = {
            #     "local_feat_extractor": local_feat_extractor,
            #     "max_keypoints": 4096,
            # }
            f0, f1, mtc, _, _ = self._match_pairs(i0, i1)
            vld = mtc > -1
            kp0 = f0.keypoints[vld]
            kp1 = f1.keypoints[mtc[vld]]
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
            self.timer.update("preselection")

            # Debug...
            # c = "r"
            # s = 5
            # fig, axes = plt.subplots(1, 2)
            # for ax, img, kp in zip(axes, [image0, image1], [kp0, kp1]):
            #     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR))
            #     ax.scatter(kp[:, 0], kp[:, 1], s=s, c=c)
            #     ax.axis("off")
            # for lim0, lim1 in zip(t0_lims.values(), t1_lims.values()):
            #     axes[0].axvline(lim0[0])
            #     axes[0].axhline(lim0[1])
            #     axes[1].axvline(lim1[0])
            #     axes[1].axhline(lim1[1])
            # # axes[1].get_yaxis().set_visible(False)
            # fig.tight_layout()
            # plt.show()
            # fig.savefig("preselection.jpg")
            # plt.close()

        return tile_pairs

    def _filter_matches_by_mask(self, inlMask: np.ndarray) -> None:
        """
        Filter matches based on the specified mask.

        Args:
            inlMask (np.ndarray): The mask to filter matches.
        """
        self._mkpts0 = self._mkpts0[inlMask, :]
        self._mkpts1 = self._mkpts1[inlMask, :]
        if self._descriptors0 is not None and self._descriptors1 is not None:
            self._descriptors0 = self._descriptors0[:, inlMask]
            self._descriptors1 = self._descriptors1[:, inlMask]
        if self._scores0 is not None and self._scores1 is not None:
            self._scores0 = self._scores0[inlMask]
            self._scores1 = self._scores1[inlMask]
        if self._mconf is not None:
            self._mconf = self._mconf[inlMask]

    # def reset(self):
    #     """Reset the matcher by cleaning the features and matches"""
    #     pass

    # def _store_matched_features(
    #     self,
    #     features0: FeaturesBase,
    #     features1: FeaturesBase,
    #     matches0: np.ndarray,
    #     matches01: np.ndarray = None,
    #     mconf: np.ndarray = None,
    #     force_overwrite: bool = True,
    # ) -> bool:
    #     """Stores keypoints, descriptors and scores of the matches in the object's members."""

    #     assert isinstance(
    #         features0, FeaturesBase
    #     ), "features0 must be a FeaturesBase object"
    #     assert isinstance(
    #         features1, FeaturesBase
    #     ), "features1 must be a FeaturesBase object"
    #     assert hasattr(features0, "keypoints"), "No keypoints found in features0"
    #     assert hasattr(features1, "keypoints"), "No keypoints found in features1"

    #     if self._mkpts0 is not None and self._mkpts1 is not None:
    #         if force_overwrite is False:
    #             logger.warning(
    #                 "Matches already stored. Not overwriting them. Use force_overwrite=True to force overwrite them."
    #             )
    #             return False

    #     # Store features as class members
    #     self._features0 = features0
    #     self._features1 = features1

    #     # Store matching arrays as class members
    #     self._matches0 = matches0
    #     self._matches01 = matches01

    #     # Store match confidence (store None if not available)
    #     self._mconf = mconf

    #     # Stored matched keypoints
    #     valid = matches0 > -1
    #     idx1 = matches0[valid]
    #     self._mkpts0 = features0.keypoints[valid]
    #     self._mkpts1 = features1.keypoints[idx1]
    #     if features0.descriptors is not None:
    #         self._descriptors0 = features0.descriptors[:, valid]
    #         self._descriptors1 = features1.descriptors[:, idx1]
    #     if features0.scores is not None:
    #         self._scores0 = features0.scores[valid]
    #         self._scores1 = features1.scores[idx1]

    #     return True

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

    def save_mkpts_as_txt(
        self,
        savedir: Union[str, Path],
        delimiter: str = ",",
        header: str = "x,y",
    ) -> None:
        """Save keypoints in a .txt file"""
        path = Path(savedir)
        path.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            path / "keypoints_0.txt",
            self.mkpts0,
            delimiter=delimiter,
            newline="\n",
            header=header,
            fmt="%.2f",  # Format to two decimal places
        )
        np.savetxt(
            path / "keypoints_1.txt",
            self.mkpts1,
            delimiter=delimiter,
            newline="\n",
            header=header,
            fmt="%.2f",  # Format to two decimal places
        )
