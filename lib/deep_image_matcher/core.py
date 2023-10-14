import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union
import importlib
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .consts import GeometricVerification, Quality, TileSelection
from .geometric_verification import geometric_verification
from .tiling import Tiler
from .timer import AverageTimer, timeit

logger = logging.getLogger(__name__)

# default parameters
MIN_MATCHES_PER_TILE = 5


@dataclass
class FeaturesBase:
    keypoints: np.ndarray
    descriptors: np.ndarray = None
    scores: np.ndarray = None


def check_dict_keys(dict: dict, keys: List[str]):
    missing_keys = [key for key in keys if key not in dict]
    if missing_keys:
        raise KeyError(
            f"Missing required keys: {', '.join(missing_keys)} Matcher option dictionary"
        )


def viz_matches_mpl(
    image0: np.ndarray,
    image1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    save_path: str = None,
    hide_fig: bool = True,
    **config,
) -> None:
    if hide_fig:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")  # Use the Agg backend for rendering

    # Get config
    colors = config.get("c", config.get("color", ["r", "r"]))
    if isinstance(colors, str):
        colors = [colors, colors]
    s = config.get("s", 2)
    figsize = config.get("figsize", (20, 8))

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], s=s, c=colors[0])
    ax[0].axis("equal")
    ax[1].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], s=s, c=colors[1])
    ax[1].axis("equal")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if hide_fig is False:
        plt.show()
    else:
        plt.close(fig)


def viz_matches_cv2(
    image0: np.ndarray,
    image1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    save_path: str = None,
    pts_col: Tuple[int] = (0, 0, 255),
    point_size: int = 1,
    line_col: Tuple[int] = (0, 255, 0),
    line_thickness: int = 1,
    margin: int = 10,
    autoresize: bool = True,
    max_long_edge: int = 2000,
    jpg_quality: int = 80,
) -> np.ndarray:
    """Plot matching points between two images using OpenCV.

    Args:
        image0: The first image.
        image1: The second image.
        pts0: List of 2D points in the first image.
        pts1: List of 2D points in the second image.
        pts_col: RGB color of the points.
        point_size: Size of the circles representing the points.
        line_col: RGB color of the matching lines.
        line_thickness: Thickness of the lines connecting the points.
        path: Path to save the output image.
        margin: Margin between the two images in the output.

    Returns:
        np.ndarrya: The output image.
    """
    if image0.ndim > 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    if image1.ndim > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    if autoresize:
        H0, W0 = image0.shape[:2]
        H1, W1 = image1.shape[:2]
        max_size = max(H0, W0, H1, W1)
        scale_factor = max_long_edge / max_size
        new_W0, new_H0 = int(W0 * scale_factor), int(H0 * scale_factor)
        new_W1, new_H1 = int(W1 * scale_factor), int(H1 * scale_factor)

        image0 = cv2.resize(image0, (new_W0, new_H0))
        image1 = cv2.resize(image1, (new_W1, new_H1))

        # Scale the keypoints accordingly
        pts0 = (pts0 * scale_factor).astype(int)
        pts1 = (pts1 * scale_factor).astype(int)

    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin :] = image1
    out = np.stack([out] * 3, -1)

    mkpts0, mkpts1 = np.round(pts0).astype(int), np.round(pts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        if line_thickness > -1:
            # draw lines between matching keypoints
            cv2.line(
                out,
                (x0, y0),
                (x1 + margin + W0, y1),
                color=line_col,
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_size, pts_col, -1, lineType=cv2.LINE_AA)
        cv2.circle(
            out,
            (x1 + margin + W0, y1),
            point_size,
            pts_col,
            -1,
            lineType=cv2.LINE_AA,
        )
    if save_path is not None:
        cv2.imwrite(str(save_path), out, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])

    return out


class ImageMatcherBase:
    def __init__(self, **config) -> None:
        """
        Base class for matchers. It defines the basic interface for matchers and basic functionalities that are shared among all matchers, in particular the `match` method. It must be subclassed to implement a new matcher.

        Args:
            opt (dict): Options for the matcher.

        Raises:
            TypeError: If `opt` is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("opt must be a dictionary")

        # Get main config parameters
        self._quality = config.get("quality", Quality.HIGH)
        self._tiling = config.get("tile_selection", TileSelection.NONE)
        self._gv = config.get(
            "geometric_verification", GeometricVerification.PYDEGENSAC
        )
        self._config = config

        # Get device
        self._device = (
            "cuda"
            if torch.cuda.is_available() and not config.get("force_cpu")
            else "cpu"
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
        """Reset the matcher by clearing the features and matches"""
        self._mkpts0 = None
        self._mkpts1 = None
        self._descriptors0 = None
        self._descriptors1 = None
        self._scores0 = None
        self._scores1 = None
        self._mconf = None

    @timeit
    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        # quality: Quality = Quality.HIGH,
        # tile_selection: TileSelection = TileSelection.NONE,
        **config,
    ) -> bool:
        """
        Matches images and performs geometric verification.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            quality: The quality level for resizing images (default: Quality.HIGH).
            tile_selection: The method for selecting tiles for matching (default: TileSelection.NONE).
            **config: Additional keyword arguments for customization.

        Returns:
            A boolean indicating the success of the matching process.

        """
        assert isinstance(image0, np.ndarray), "image0 must be a NumPy array"
        assert isinstance(image1, np.ndarray), "image1 must be a NumPy array"

        self.timer = AverageTimer()

        # Get config from class members or from user input
        config = {**self._config, **config}

        # Get qualtiy and tile selection parameters
        quality = config.get("quality", self._quality)
        tile_selection = config.get("tile_selection", self._tiling)

        # Get geometric verification config parameters
        gv_method = config.get("geometric_verification", self._gv)
        gv_threshold = config.get("threshold", 1)
        config.get("confidence", 0.9999)

        # Get visualization parameters
        do_viz = config.get("do_viz", False)
        fast_viz = config.get("fast_viz", True)
        config.get("interactive_viz", True)
        hide_matching_track = self._config.get("hide_matching_track", False)

        # Define saving directory
        save_dir = config.get("save_dir", None)
        if save_dir is not None:
            self._save_dir = Path(save_dir)
            self._save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._save_dir = None

        # Check input parameters
        assert isinstance(quality, Quality), "quality must be a Quality enum"
        assert isinstance(
            tile_selection, TileSelection
        ), "tile_selection must be a TileSelection enum"
        assert isinstance(
            gv_method, GeometricVerification
        ), "geometric_verification must be a GeometricVerification enum"

        # Resize images if needed
        image0_, image1_ = self._resize_images(quality, image0, image1)

        # Perform matching (on tiles or full images)
        if tile_selection == TileSelection.NONE:
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
            quality, features0, features1
        )

        # Store features as class members
        try:
            self._store_matched_features(self._features0, self._features1, matches0)
            self._mconf = mconf
        except Exception as e:
            logger.error(
                f"""Error storing matches: {e}. 
                Implement your own _store_matched_features() method if the
                output of your matcher is different from FeaturesBase."""
            )
        self.timer.update("matching")
        logger.info("Matching done!")

        # Perform geometric verification
        logger.info("Performing geometric verification...")
        if gv_method is not GeometricVerification.NONE:
            F, inlMask = geometric_verification(
                self._mkpts0,
                self._mkpts1,
                method=gv_method,
                confidence=gv_threshold,
                threshold=gv_threshold,
            )
            self._F = F
            self._filter_matches_by_mask(inlMask)
            logger.info("Geometric verification done.")
            self.timer.update("geometric_verification")

        if do_viz is True:
            self.viz_matches(
                image0,
                image1,
                self._mkpts0,
                self._mkpts1,
                str(self._save_dir / "matches.jpg"),
                fast_viz=fast_viz,
                hide_matching_track=hide_matching_track,
            )
            # else:
            #     if self._save_dir is not None:
            #         self.viz_matches_mpl(
            #             image0,
            #             image1,
            #             self._mkpts0,
            #             self._mkpts1,
            #             self._save_dir / "matches.jpg",
            #             hide_fig=True,
            #             point_size=5,
            #         )
            #     else:
            #         self.viz_matches_mpl(
            #             image0,
            #             image1,
            #             self._mkpts0,
            #             self._mkpts1,
            #             hide_fig=False,
            #         )

            # if fast_viz:
            #     assert (
            #         self._save_dir is not None
            #     ), "save_dir must be specified for fast_viz"
            #     if hide_matching_track:
            #         line_thickness = -1
            #     else:
            #         line_thickness = 1

            #     self.viz_matches_cv2(
            #         image0,
            #         image1,
            #         self._mkpts0,
            #         self._mkpts1,
            #         str(self._save_dir / "matches.jpg"),
            #         line_thickness=line_thickness,
            #         autoresize=True,
            #         max_long_edge=2000,
            #         jpg_quality=95,
            #     )
            # else:
            #     if self._save_dir is not None:
            #         self.viz_matches_mpl(
            #             image0,
            #             image1,
            #             self._mkpts0,
            #             self._mkpts1,
            #             self._save_dir / "matches.jpg",
            #             hide_fig=True,
            #             point_size=5,
            #         )
            #     else:
            #         self.viz_matches_mpl(
            #             image0,
            #             image1,
            #             self._mkpts0,
            #             self._mkpts1,
            #             hide_fig=False,
            #         )
        if self._save_dir is not None:
            self.save_mkpts_as_txt(self._save_dir)

        self.timer.print("Matching")

        return True

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

        # Get config
        tile_selection = config.get("tile_selection", TileSelection.PRESELECTION)
        grid = config.get("grid", [1, 1])
        overlap = config.get("overlap", 0)
        origin = config.get("origin", [0, 0])
        do_viz_tiles = config.get("do_viz_tiles", False)

        # # Convert images to grayscale if needed
        # if grayscale is True:
        #     if len(image0.shape) > 2:
        #         image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        #     if len(image1.shape) > 2:
        #         image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

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
            save_dir = config.get("save_dir", ".")
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if do_viz_tiles is True:
                out_img_path = str(save_dir / f"matches_tile_{tidx0}-{tidx1}.jpg")
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
                # try:
                #     hide_matching_track = self._config.get("hide_matching_track", False)
                #     if hide_matching_track:
                #         line_thickness = -1
                #     else:
                #         line_thickness = 1
                #     self.viz_matches_cv2(
                #         tile0,
                #         tile1,
                #         mkpts0,
                #         mkpts1,
                #         str(out_img_path),
                #         line_thickness=line_thickness,
                #         autoresize=True,
                #         max_long_edge=1200,
                #     )
                # except Exception:
                #     self.viz_matches_mpl(
                #         tile0,
                #         tile1,
                #         mkpts0,
                #         mkpts1,
                #         out_img_path,
                #         hide_fig=True,
                #     )

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

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
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

        # Get MIN_MATCHES_PER_TILE
        min_matches_per_tile = config.get("min_matches_per_tile", MIN_MATCHES_PER_TILE)

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
            f0, f1, mtc, _ = self._match_pairs(i0, i1, max_keypoints=4096)
            vld = mtc > -1
            kp0 = f0.keypoints[vld]
            kp1 = f1.keypoints[mtc[vld]]
            if self._config.get("do_viz", False) is True:
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

                # try:
                #     self.viz_matches_cv2(
                #         i0,
                #         i1,
                #         kp0,
                #         kp1,
                #         str(self._save_dir / "tile_preselection.jpg"),
                #         line_thickness=-1,
                #         autoresize=True,
                #         max_long_edge=1200,
                #     )
                # except Exception:
                #     self.viz_matches_mpl(
                #         i0,
                #         i1,
                #         kp0,
                #         kp1,
                #         self._save_dir / "tile_preselection.jpg",
                #         hide_fig=True,
                #     )

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
            # Get config for Matplotlib visualization
            # colors = config.get("c", config.get("color", ["r", "r"]))
            # if isinstance(colors, str):
            #     colors = [colors, colors]
            # config.get("s", 2)
            # config.get("figsize", (20, 8))

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

    # def viz_matches_mpl(
    #     self,
    #     image0: np.ndarray,
    #     image1: np.ndarray,
    #     kpts0: np.ndarray,
    #     kpts1: np.ndarray,
    #     save_path: str = None,
    #     hide_fig: bool = True,
    #     **config,
    # ) -> None:
    #     if hide_fig:
    #         matplotlib = importlib.import_module("matplotlib")
    #         matplotlib.use("Agg")  # Use the Agg backend for rendering

    #     # Get config
    #     colors = config.get("c", config.get("color", ["r", "r"]))
    #     if isinstance(colors, str):
    #         colors = [colors, colors]
    #     s = config.get("s", 2)
    #     figsize = config.get("figsize", (20, 8))

    #     fig, ax = plt.subplots(1, 2, figsize=figsize)
    #     ax[0].imshow(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
    #     ax[0].scatter(kpts0[:, 0], kpts0[:, 1], s=s, c=colors[0])
    #     ax[0].axis("equal")
    #     ax[1].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    #     ax[1].scatter(kpts1[:, 0], kpts1[:, 1], s=s, c=colors[1])
    #     ax[1].axis("equal")
    #     fig.tight_layout()
    #     if save_path is not None:
    #         fig.savefig(save_path)
    #     if hide_fig is False:
    #         plt.show()
    #     else:
    #         plt.close(fig)

    # def viz_matches_cv2(
    #     self,
    #     image0: np.ndarray,
    #     image1: np.ndarray,
    #     pts0: np.ndarray,
    #     pts1: np.ndarray,
    #     save_path: str = None,
    #     pts_col: Tuple[int] = (0, 0, 255),
    #     point_size: int = 1,
    #     line_col: Tuple[int] = (0, 255, 0),
    #     line_thickness: int = 1,
    #     margin: int = 10,
    #     autoresize: bool = True,
    #     max_long_edge: int = 1200,
    #     jpg_quality: int = 80,
    # ) -> np.ndarray:
    #     """Plot matching points between two images using OpenCV.

    #     Args:
    #         image0: The first image.
    #         image1: The second image.
    #         pts0: List of 2D points in the first image.
    #         pts1: List of 2D points in the second image.
    #         pts_col: RGB color of the points.
    #         point_size: Size of the circles representing the points.
    #         line_col: RGB color of the matching lines.
    #         line_thickness: Thickness of the lines connecting the points.
    #         path: Path to save the output image.
    #         margin: Margin between the two images in the output.

    #     Returns:
    #         np.ndarrya: The output image.
    #     """
    #     if image0.ndim > 2:
    #         image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    #     if image1.ndim > 2:
    #         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    #     if autoresize:
    #         H0, W0 = image0.shape[:2]
    #         H1, W1 = image1.shape[:2]
    #         max_size = max(H0, W0, H1, W1)
    #         scale_factor = max_long_edge / max_size
    #         new_W0, new_H0 = int(W0 * scale_factor), int(H0 * scale_factor)
    #         new_W1, new_H1 = int(W1 * scale_factor), int(H1 * scale_factor)

    #         image0 = cv2.resize(image0, (new_W0, new_H0))
    #         image1 = cv2.resize(image1, (new_W1, new_H1))

    #         # Scale the keypoints accordingly
    #         pts0 = (pts0 * scale_factor).astype(int)
    #         pts1 = (pts1 * scale_factor).astype(int)

    #     H0, W0 = image0.shape
    #     H1, W1 = image1.shape
    #     H, W = max(H0, H1), W0 + W1 + margin

    #     out = 255 * np.ones((H, W), np.uint8)
    #     out[:H0, :W0] = image0
    #     out[:H1, W0 + margin :] = image1
    #     out = np.stack([out] * 3, -1)

    #     mkpts0, mkpts1 = np.round(pts0).astype(int), np.round(pts1).astype(int)
    #     for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
    #         if line_thickness > -1:
    #             # draw lines between matching keypoints
    #             cv2.line(
    #                 out,
    #                 (x0, y0),
    #                 (x1 + margin + W0, y1),
    #                 color=line_col,
    #                 thickness=line_thickness,
    #                 lineType=cv2.LINE_AA,
    #             )
    #         # display line end-points as circles
    #         cv2.circle(out, (x0, y0), point_size, pts_col, -1, lineType=cv2.LINE_AA)
    #         cv2.circle(
    #             out,
    #             (x1 + margin + W0, y1),
    #             point_size,
    #             pts_col,
    #             -1,
    #             lineType=cv2.LINE_AA,
    #         )
    #     if save_path is not None:
    #         cv2.imwrite(
    #             str(save_path), out, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    #         )

    #     return out

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
