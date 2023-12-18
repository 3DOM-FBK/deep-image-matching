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
from .matcher_base import FeaturesDict, MatcherBase


class LOFTRMatcher(MatcherBase):
    def __init__(self, config={}) -> None:
        """Initializes a LOFTRMatcher with Kornia object with the given options dictionary."""

        super().__init__(config)

        model = config["matcher"]["pretrained"]
        self.matcher = KF.LoFTR(pretrained=model).to(self._device).eval()

        self.grayscale = True
        self.as_float = True
        self._quality = config["general"]["quality"]

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

        tile_selection = self._config["general"]["tile_selection"]

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

        if 


    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ):
        """Matches keypoints and descriptors in two given images
        (no matter if they are tiles or full-res images) using
        the LoFTR algorithm.

        Args:

        Returns:

        """

        feature_path = feats0["feature_path"]

        im_path0 = feats0["im_path"]
        im_path1 = feats1["im_path"]

        # Load image
        image0 = cv2.imread(im_path0)
        image1 = cv2.imread(im_path1)

        # Covert images to tensor
        timg0_ = self._frame2tensor(image0, self._device)
        timg1_ = self._frame2tensor(image1, self._device)

        # Run inference
        with torch.inference_mode():
            input_dict = {"image0": timg0_, "image1": timg1_}
            correspondences = self.matcher(input_dict)

        # Get matches and build features
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        features0 = FeaturesDict(keypoints=mkpts0)
        features1 = FeaturesDict(keypoints=mkpts1)

        # Get match confidence
        mconf = correspondences["confidence"].cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))
        self._update_features(
            feature_path,
            Path(im_path0).name,
            Path(im_path1).name,
            mkpts0,
            mkpts1,
            matches,
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

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        image = K.image_to_tensor(np.array(image), False).float() / 255.0
        image = K.color.bgr_to_rgb(image.to(device))
        if image.shape[1] > 2:
            image = K.color.rgb_to_grayscale(image)
        return image

    def _update_features(
        self, feature_path, im0_name, im1_name, new_keypoints0, new_keypoints1, matches0
    ) -> None:
        for i, im_name, new_keypoints in zip(
            [0, 1], [im0_name, im1_name], [new_keypoints0, new_keypoints1]
        ):
            features = get_features(feature_path, im_name)
            existing_keypoints = features["keypoints"]

            if len(existing_keypoints.shape) == 1:
                features["keypoints"] = new_keypoints
                with h5py.File(feature_path, "r+", libver="latest") as fd:
                    del fd[im_name]
                    grp = fd.create_group(im_name)
                    for k, v in features.items():
                        if k == "im_path" or k == "feature_path":
                            grp.create_dataset(k, data=str(v))
                        if isinstance(v, np.ndarray):
                            grp.create_dataset(k, data=v)

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
