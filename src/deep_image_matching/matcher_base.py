import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .consts import GeometricVerification, Quality, TileSelection
from .geometric_verification import geometric_verification
from .tiling import Tiler
from .utils import Timer, timeit
from .visualization import viz_matches_cv2, viz_matches_mpl

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "quality": Quality.HIGH,
    "tile_selection": TileSelection.NONE,
    "geometric_verification": GeometricVerification.PYDEGENSAC,
    "max_keypoints": 4096,
    "force_cpu": False,
    "save_dir": "results",
    "do_viz": False,
    "fast_viz": True,
    # "interactive_viz": False,
    "hide_matching_track": True,
    "gv_threshold": 1,
    "gv_confidence": 0.9999,
    "do_viz_tiles": False,
    "tiling_grid": [1, 1],
    "tiling_overlap": 0,
    "tiling_origin": [0, 0],
    "min_matches_per_tile": 5,
}


@dataclass
class FeaturesBase:
    keypoints: np.ndarray
    descriptors: np.ndarray = None
    scores: np.ndarray = None


def check_matching_config(config: dict) -> None:
    """Check the matching config dictionary for missing keys or invalid values."""
    required_keys = [
        "quality",
        "tile_selection",
        "geometric_verification",
        "max_keypoints",
        "force_cpu",
        "save_dir",
        "do_viz",
        "fast_viz",
        # "interactive_viz",
        "hide_matching_track",
        "gv_threshold",
        "gv_confidence",
        "do_viz_tiles",
        "tiling_grid",
        "tiling_overlap",
        "tiling_origin",
        "min_matches_per_tile",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {', '.join(missing_keys)} Matcher option dictionary"
        )
    if not isinstance(config["quality"], Quality):
        raise TypeError("quality must be a Quality enum")
    if not isinstance(config["tile_selection"], TileSelection):
        raise TypeError("tile_selection must be a TileSelection enum")
    if not isinstance(config["geometric_verification"], GeometricVerification):
        raise TypeError("geometric_verification must be a GeometricVerification enum")
    if not isinstance(config["max_keypoints"], int):
        raise TypeError("max_keypoints must be an integer")


class ImageMatcherBase:
    def __init__(self, **config) -> None:
        """
        Base class for matchers. It defines the basic interface for matchers and basic functionalities that are shared among all matchers, in particular the `match` method. It must be subclassed to implement a new matcher.

        Args:
            opt (dict): Options for the matcher.

        Raises:
            TypeError: If `opt` is not a dictionary.
        """

        # Get custom config
        if not isinstance(config, dict):
            raise TypeError("opt must be a dictionary")

        # Get defaualt config and update it with custom config
        config = {**DEFAULT_CONFIG, **config}
        self._config = config

        # Get main processing parameters
        self._quality = config["quality"]
        self._tiling = config["tile_selection"]
        self._gv = config["geometric_verification"]

        # Check input config
        check_matching_config(config)
        logger.info(f"Matching options: {self._quality} - {self._tiling}")

        # Define saving directory
        save_dir = config["save_dir"]
        if save_dir is not None:
            self._save_dir = Path(save_dir)
            self._save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._save_dir = None
        logger.info(f"Saving directory: {self._save_dir}")

        # Get device
        self._device = (
            "cuda" if torch.cuda.is_available() and not config["force_cpu"] else "cpu"
        )
        logger.info(f"Running inference on device {self._device}")

        # initialize additional variable members for storing matched
        # keypoints descriptors and scores
        self._mkpts0 = None  # matched keypoints on image 0
        self._mkpts1 = None  # matched keypoints on image 1
        self._descriptors0 = None  # descriptors of mkpts on image 0
        self._descriptors1 = None  # descriptors of mkpts on image 1
        self._scores0 = None  # scores of mkpts on image 0
        self._scores1 = None  # scores of mkpts on image 1
        self._mconf = None  # match confidence (i.e., scores0 of the valid matches)

    @property
    def device(self):
        return self._device

    @property
    def mkpts0(self):
        return self._mkpts0

    @property
    def mkpts1(self):
        return self._mkpts1

    @property
    def descriptors0(self):
        return self._descriptors0

    @property
    def descriptors1(self):
        return self._descriptors1

    @property
    def scores0(self):
        return self._scores0

    @property
    def scores1(self):
        return self._scores1

    @property
    def mconf(self):
        return self._mconf

    def reset(self):
        """Reset the matcher by cleaning the features and matches"""
        self._mkpts0 = None
        self._mkpts1 = None
        self._descriptors0 = None
        self._descriptors1 = None
        self._scores0 = None
        self._scores1 = None
        self._mconf = None

    def _match_pairs(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        **config,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no
        matter if they are tiles or full-res images).

        This method takes in two images as Numpy arrays, and returns
        the matches between keypoints and descriptors in those images.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]: a
            tuple containing the features of the first image, the features
            of the second image, the matches between them and the match
            confidence.
        """
        raise NotImplementedError(
            "Subclasses must implement _match_full_images() method."
        )

    @timeit
    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        **config,
    ) -> bool:
        """
        Matches images and performs geometric verification.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            **config: Additional keyword arguments for customization.

        Returns:
            A boolean indicating the success of the matching process.

        """
        self.timer = Timer()

        # Check input images
        assert isinstance(image0, np.ndarray), "image0 must be a NumPy array"
        assert isinstance(image1, np.ndarray), "image1 must be a NumPy array"

        # If a custom config is passed, update the default config
        config = {**self._config, **config}

        # Resize images if needed
        image0_, image1_ = self._resize_images(self._quality, image0, image1)

        # Perform matching (on tiles or full images)
        if self._tiling == TileSelection.NONE:
            logger.info("Matching full images...")
            features0, features1, matches0, mconf = self._match_pairs(
                image0_, image1_, **config
            )

        else:
            logger.info("Matching by tiles...")
            features0, features1, matches0, mconf = self._match_by_tile(
                image0_, image1_, **config
            )

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        self._features0, self._features1 = self._resize_features(
            self._quality, features0, features1
        )

        # Added by Luca
        return self._features0, self._features1, matches0, mconf

        # Store features as class members
        try:
            self._store_matched_features(self._features0, self._features1, matches0)
            self._mconf = mconf
        except Exception as e:
            raise NotImplementedError(
                f"""{e}. 
                Error when storing matches. Implement your own _store_matched_features() method if the output of your _match_pairs function is different from FeaturesBase."""
            )
        self.timer.update("matching")
        logger.info("Matching done!")

        # Perform geometric verification
        logger.info("Performing geometric verification...")
        if self._gv is not GeometricVerification.NONE:
            F, inlMask = geometric_verification(
                self._mkpts0,
                self._mkpts1,
                method=self._gv,
                confidence=config["gv_confidence"],
                threshold=config["gv_threshold"],
            )
            self._F = F
            self._filter_matches_by_mask(inlMask)
            logger.info("Geometric verification done.")
            self.timer.update("geometric_verification")

        if config["do_viz"] is True:
            self.viz_matches(
                image0,
                image1,
                self._mkpts0,
                self._mkpts1,
                str(self._save_dir / "matches.jpg"),
                fast_viz=config["fast_viz"],
                hide_matching_track=config["hide_matching_track"],
            )

        if self._save_dir is not None:
            self.save_mkpts_as_txt(self._save_dir)

        self.timer.print("Matching")

        return True

    def _match_by_tile(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        **config,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """
        Matches tiles in two images and returns the features, matches, and confidence.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            **config: Additional keyword arguments for customization.

        Returns:
            A tuple containing:
            - features0: FeaturesBase object representing keypoints, descriptors, and scores of image0.
            - features1: FeaturesBase object representing keypoints, descriptors, and scores of image1.
            - matches0: NumPy array with indices of matched keypoints in image0.
            - mconf: NumPy array with confidence scores for the matches.

        Raises:
            AssertionError: If image0 or image1 is not a NumPy array.

        """

        # Get config parameters
        tile_selection = self._tiling
        grid = config["tiling_grid"]
        overlap = config["tiling_overlap"]
        origin = config["tiling_origin"]
        do_viz_tiles = config["do_viz_tiles"]

        # Compute tiles limits and origin
        self._tiler = Tiler(grid=grid, overlap=overlap, origin=origin)
        t0_lims, t0_origin = self._tiler.compute_limits_by_grid(image0)
        t1_lims, t1_origin = self._tiler.compute_limits_by_grid(image1)

        # Select tile pairs to match
        tile_pairs = self._tile_selection(
            image0, image1, t0_lims, t1_lims, tile_selection, config=config
        )

        # Initialize empty array for storing matched keypoints, descriptors and scores
        mkpts0_full = np.array([], dtype=np.float32).reshape(0, 2)
        mkpts1_full = np.array([], dtype=np.float32).reshape(0, 2)
        descriptors0_full = np.array([], dtype=np.float32).reshape(256, 0)
        descriptors1_full = np.array([], dtype=np.float32).reshape(256, 0)
        scores0_full = np.array([], dtype=np.float32)
        scores1_full = np.array([], dtype=np.float32)
        conf_full = np.array([], dtype=np.float32)

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logger.info(f" - Matching tile pair ({tidx0}, {tidx1})")

            lim0 = t0_lims[tidx0]
            lim1 = t1_lims[tidx1]
            tile0 = self._tiler.extract_patch(image0, lim0)
            tile1 = self._tiler.extract_patch(image1, lim1)
            features0, features1, matches0, conf = self._match_pairs(
                tile0, tile1, **config
            )

            kpts0, kpts1 = features0.keypoints, features1.keypoints
            descriptors0, descriptors1 = (
                features0.descriptors,
                features1.descriptors,
            )
            scores0, scores1 = features0.scores, features1.scores

            valid = matches0 > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches0[valid]]
            descriptors0 = descriptors0[:, valid]
            descriptors1 = descriptors1[:, matches0[valid]]
            scores0 = scores0[valid]
            scores1 = scores1[matches0[valid]]

            # Append matches to full array
            mkpts0_full = np.vstack(
                (mkpts0_full, mkpts0 + np.array(lim0[0:2]).astype("float32"))
            )
            mkpts1_full = np.vstack(
                (mkpts1_full, mkpts1 + np.array(lim1[0:2]).astype("float32"))
            )
            descriptors0_full = np.hstack((descriptors0_full, descriptors0))
            descriptors1_full = np.hstack((descriptors1_full, descriptors1))
            scores0_full = np.concatenate((scores0_full, scores0))
            scores1_full = np.concatenate((scores1_full, scores1))
            conf_full = np.concatenate((conf_full, conf))

            # Visualize matches on tile
            if do_viz_tiles is True:
                out_img_path = str(self._save_dir / f"matches_tile_{tidx0}-{tidx1}.jpg")
                self.viz_matches(
                    tile0,
                    tile1,
                    mkpts0,
                    mkpts1,
                    out_img_path,
                    fast_viz=True,
                    hide_matching_track=True,
                    autoresize=True,
                    max_long_edge=1200,
                )
        logger.info("Restoring full image coordinates of matches...")

        # Restore original image coordinates (not cropped)
        mkpts0_full = mkpts0_full + np.array(t0_origin).astype("float32")
        mkpts1_full = mkpts1_full + np.array(t1_origin).astype("float32")

        # Select uniue features on image 0
        mkpts0_full, unique_idx = np.unique(mkpts0_full, axis=0, return_index=True)
        descriptors0_full = descriptors0_full[:, unique_idx]
        scores0_full = scores0_full[unique_idx]
        mkpts1_full = mkpts1_full[unique_idx]
        descriptors1_full = descriptors1_full[:, unique_idx]
        scores1_full = scores1_full[unique_idx]

        # Create features
        features0 = FeaturesBase(
            keypoints=mkpts0_full, descriptors=descriptors0_full, scores=scores0_full
        )
        features1 = FeaturesBase(
            keypoints=mkpts1_full, descriptors=descriptors1_full, scores=scores1_full
        )

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        # Create a match confidence array
        valid = matches0 > -1
        mconf = features0.scores[valid]

        logger.info("Matching by tile completed.")

        return features0, features1, matches0, mconf

    def _frame2tensor(self, image: np.ndarray) -> torch.Tensor:
        """Normalize the image tensor and add batch dimension."""
        # if image.ndim == 3:
        #     image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        # elif image.ndim == 2:
        #     image = image[None]  # add channel axis
        # else:
        #     raise ValueError(f'Not an image: {image.shape}')
        # return torch.tensor(image / 255., dtype=torch.float).to(device)

        device = torch.device(self._device if torch.cuda.is_available() else "cpu")
        return torch.tensor(image / 255.0, dtype=torch.float)[None, None].to(device)

    def _tile_selection(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        t0_lims: dict[int, np.ndarray],
        t1_lims: dict[int, np.ndarray],
        method: TileSelection = TileSelection.PRESELECTION,
        **config,
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

        # Get MIN_MATCHES_PER_TILE
        min_matches_per_tile = config["min_matches_per_tile"]

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
            f0, f1, mtc, _ = self._match_pairs(i0, i1, **config)
            vld = mtc > -1
            kp0 = f0.keypoints[vld]
            kp1 = f1.keypoints[mtc[vld]]
            if self._config["do_viz"] is True:
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
                if sum(ret) > min_matches_per_tile:
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

    def _resize_images(
        self, quality: Quality, image0: np.ndarray, image1: np.ndarray
    ) -> Tuple[np.ndarray]:
        """
        Resize images based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            image0 (np.ndarray): The first image.
            image1 (np.ndarray): The second image.

        Returns:
            Tuple[np.ndarray]: Resized images.

        """
        if quality == Quality.HIGHEST:
            image0_ = cv2.pyrUp(image0)
            image1_ = cv2.pyrUp(image1)
        elif quality == Quality.HIGH:
            image0_ = image0
            image1_ = image1
        elif quality == Quality.MEDIUM:
            image0_ = cv2.pyrDown(image0)
            image1_ = cv2.pyrDown(image1)
        elif quality == Quality.LOW:
            image0_ = cv2.pyrDown(cv2.pyrDown(image0))
            image1_ = cv2.pyrDown(cv2.pyrDown(image1))
        return image0_, image1_

    def _resize_features(
        self, quality: Quality, features0: FeaturesBase, features1: FeaturesBase
    ) -> Tuple[FeaturesBase]:
        """
        Resize features based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            features0 (FeaturesBase): The features of the first image.
            features1 (FeaturesBase): The features of the second image.

        Returns:
            Tuple[FeaturesBase]: Resized features.

        """
        if quality == Quality.HIGHEST:
            features0.keypoints /= 2
            features1.keypoints /= 2
        elif quality == Quality.HIGH:
            pass
        elif quality == Quality.MEDIUM:
            features0.keypoints *= 2
            features1.keypoints *= 2
        elif quality == Quality.LOW:
            features0.keypoints *= 4
            features1.keypoints *= 4

        return features0, features1

    def _store_matched_features(
        self,
        features0: FeaturesBase,
        features1: FeaturesBase,
        matches0: np.ndarray,
        force_overwrite: bool = True,
    ) -> bool:
        """Stores keypoints, descriptors and scores of the matches in the object's members."""

        assert isinstance(
            features0, FeaturesBase
        ), "features0 must be a FeaturesBase object"
        assert isinstance(
            features1, FeaturesBase
        ), "features1 must be a FeaturesBase object"
        assert hasattr(features0, "keypoints"), "No keypoints found in features0"
        assert hasattr(features1, "keypoints"), "No keypoints found in features1"

        if self._mkpts0 is not None and self._mkpts1 is not None:
            if force_overwrite is False:
                logger.warning(
                    "Matches already stored. Not overwriting them. Use force_overwrite=True to force overwrite them."
                )
                return False
            else:
                logger.warning("Matches already stored. Overwrite them")

        valid = matches0 > -1
        # self._valid = valid
        idx1 = matches0[valid]
        self._mkpts0 = features0.keypoints[valid]
        self._mkpts1 = features1.keypoints[idx1]
        if features0.descriptors is not None:
            self._descriptors0 = features0.descriptors[:, valid]
            self._descriptors1 = features1.descriptors[:, idx1]
        if features0.scores is not None:
            self._scores0 = features0.scores[valid]
            self._scores1 = features1.scores[idx1]

        return True

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
