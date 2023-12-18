from pathlib import Path
from typing import Tuple

import cv2
import h5py
import kornia as K
import numpy as np
import torch
from kornia import feature as KF

from .. import Timer, logger
from ..io.h5 import get_features
from ..utils.consts import Quality, TileSelection
from ..utils.tiling import Tiler
from ..visualization import viz_matches_cv2, viz_matches_mpl
from .matcher_base import FeaturesDict, MatcherBase


class LOFTRMatcher(MatcherBase):
    default_conf = {"pretrained": "outdoor"}
    max_feat_no_tiling = 100000
    grayscale = False
    as_float = True

    def __init__(self, config={}) -> None:
        """Initializes a LOFTRMatcher with Kornia object with the given options dictionary."""

        super().__init__(config)

        model = config["matcher"]["pretrained"]
        self.matcher = KF.LoFTR(pretrained=model).to(self._device).eval()

        self._quality = config["general"]["quality"]
        self._tiling = config["general"]["tile_selection"]

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
        features0 = get_features(self._feature_path, img0_name)
        features1 = get_features(self._feature_path, img1_name)
        timer_match.update("load h5 features")

        # Perform matching
        if self._tiling == TileSelection.NONE:
            matches = self._match_pairs(features0, features1)
            timer_match.update("[match] try to match full images")
        else:
            pass

        # Save to h5 file
        n_matches = len(matches)
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            if n_matches >= self.min_matches:
                group.create_dataset(img1_name, data=matches)
            else:
                logger.debug(
                    f"Too few matches found. Skipping image pair {img0.name}-{img1.name}"
                )
                return None
        timer_match.update("save to h5")
        timer_match.print(f"{__class__.__name__} match")

        image0 = self._load_image_np(img0)
        image1 = self._load_image_np(img1)
        features0 = get_features(self._feature_path, img0_name)
        features1 = get_features(self._feature_path, img1_name)
        self.viz_matches(
            image0=image0,
            image1=image1,
            kpts0=features0["keypoints"][matches[:, 0]],
            kpts1=features1["keypoints"][matches[:, 1]],
            save_path="sandbox/test.png",
        )

        logger.debug(f"Matching {img0_name}-{img1_name} done!")

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ):
        """

        """

        feature_path = feats0["feature_path"]

        im_path0 = feats0["im_path"]
        im_path1 = feats1["im_path"]

        # Load images
        image0 = self._load_image_np(im_path0)
        image1 = self._load_image_np(im_path1)

        # # Resize images if needed
        image0_ = self._resize_image(self._quality, image0)
        image1_ = self._resize_image(self._quality, image1)

        # Covert images to tensor
        timg0_ = self._frame2tensor(image0_, self._device)
        timg1_ = self._frame2tensor(image1_, self._device)

        # Run inference
        with torch.inference_mode():
            input_dict = {"image0": timg0_, "image1": timg1_}
            correspondences = self.matcher(input_dict)

        # Get matches and build features
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

        # Create features dictionaries
        features0 = FeaturesDict(keypoints=mkpts0)
        features1 = FeaturesDict(keypoints=mkpts1)

        # # Retrieve original image coordinates if matching was performed on up/down-sampled images
        features0 = self._resize_features(self._quality, features0)
        features1 = self._resize_features(self._quality, features1)

        # Get match confidence
        mconf = correspondences["confidence"].cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))
        matches0 = self._update_features_h5(
            feature_path,
            Path(im_path0).name,
            Path(im_path1).name,
            features0["keypoints"],
            features1["keypoints"],
            matches,
        )

        return matches0

    def _match_by_tile(
        self,
        img0: Path,
        img1: Path,
        features0: FeaturesDict,
        features1: FeaturesDict,
        method: TileSelection = TileSelection.PRESELECTION,
        select_unique: bool = True,
    ):
        # Get config
        grid = self._config["general"]["tiling_grid"]
        overlap = self._config["general"]["tiling_overlap"]

        # Load image
        image0 = cv2.imread(str(img0))
        image1 = cv2.imread(str(img1))

        # Compute tiles limits and origin
        self._tiler = Tiler(grid=grid, overlap=overlap)
        t0_lims, t0_origin = self._tiler.compute_limits_by_grid(image0)
        t1_lims, t1_origin = self._tiler.compute_limits_by_grid(image1)

        # Select tile pairs to match
        tile_pairs = self._tile_selection(img0, img1, method)

        # Initialize empty array for storing matched keypoints, descriptors and scores
        mkpts0_full = np.array([], dtype=np.float32).reshape(0, 2)
        mkpts1_full = np.array([], dtype=np.float32).reshape(0, 2)
        conf_full = np.array([], dtype=np.float32)

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logger.debug(f" - Matching tile pair ({tidx0}, {tidx1})")

            lim0 = t0_lims[tidx0]
            lim1 = t1_lims[tidx1]
            tile0 = self._tiler.extract_patch(image0, lim0)
            tile1 = self._tiler.extract_patch(image1, lim1)

            # Covert patch to tensor
            timg0_ = self._frame2tensor(tile0, self._device)
            timg1_ = self._frame2tensor(tile1, self._device)

            # Run inference
            try:
                with torch.inference_mode():
                    input_dict = {"image0": timg0_, "image1": timg1_}
                    correspondences = self.matcher(input_dict)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(
                    f"Out of memory error while matching tile pair ({tidx0}, {tidx1}). Try using a lower quality level or a smaller tile size (increase the tile numbers)."
                )
                raise e

            # Get matches and build features
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()

            # Get match confidence
            conf = correspondences["confidence"].cpu().numpy()

            # Append to full arrays
            mkpts0_full = np.vstack(
                (mkpts0_full, mkpts0 + np.array(lim0[0:2]).astype("float32"))
            )
            mkpts1_full = np.vstack(
                (mkpts1_full, mkpts1 + np.array(lim1[0:2]).astype("float32"))
            )
            conf_full = np.concatenate((conf_full, conf))

        # Restore original image coordinates (not cropped)
        mkpts0_full = mkpts0_full + np.array(t0_origin).astype("float32")
        mkpts1_full = mkpts1_full + np.array(t1_origin).astype("float32")

        # Select uniue features on image 0, on rounded coordinates
        if select_unique is True:
            decimals = 1
            _, unique_idx = np.unique(
                np.round(mkpts0_full, decimals), axis=0, return_index=True
            )
            mkpts0_full = mkpts0_full[unique_idx]
            mkpts1_full = mkpts1_full[unique_idx]
            conf_full = conf_full[unique_idx]

        # Create features
        features0 = FeaturesDict(keypoints=mkpts0_full)
        features1 = FeaturesDict(keypoints=mkpts1_full)

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        logger.info("Matching by tile completed.")

        return features0, features1, matches0, conf_full

    def _load_image_np(self, img_path):
        image = cv2.imread(str(img_path))
        if self.as_float:
            image = image.astype(np.float32)
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        image = K.image_to_tensor(np.array(image), False).float() / 255.0
        image = K.color.bgr_to_rgb(image.to(device))
        if image.shape[1] > 2:
            image = K.color.rgb_to_grayscale(image)
        return image

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

    def _update_features_h5(
        self, feature_path, im0_name, im1_name, new_keypoints0, new_keypoints1, matches0
    ) -> np.ndarray:
        for i, im_name, new_keypoints in zip(
            [0, 1], [im0_name, im1_name], [new_keypoints0, new_keypoints1]
        ):
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
