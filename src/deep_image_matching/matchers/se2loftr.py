from pathlib import Path
from typing import Tuple

import cv2
import h5py
import kornia as K
import numpy as np
import torch

# from ..thirdparty.se2loftr.src.utils.misc import lower_config
from yacs.config import CfgNode as CN

from .. import Quality, TileSelection, logger
from ..io.h5 import get_features
from ..thirdparty.se2loftr.configs.loftr.outdoor import loftr_ds_e2_dense_8rot
from ..thirdparty.se2loftr.src.loftr import LoFTR
from ..utils.tiling import Tiler
from .matcher_base import FeaturesDict, MatcherBase


class SE2LOFTRMatcher(MatcherBase):
    def __init__(self, config={}) -> None:
        """Initializes a LOFTRMatcher with Kornia object with the given options dictionary."""

        super().__init__(config)

        # _default_cfg = deepcopy(default_cfg)
        # _default_cfg['coarse']['temp_bug_fix'] = False  # set to False when using the old ckpt

        _used_cfg = self._lower_config(loftr_ds_e2_dense_8rot.cfg)["loftr"]
        _used_cfg["coarse"][
            "temp_bug_fix"
        ] = True  # set to False when using the old ckpt

        # se2loftr_path = Path(__file__).parent.parent / "thirdparty" / "se2loftr" / "loftr_weights" / "outdoor_ds.ckpt" # Path to loftr weights
        se2loftr_path = (
            Path(__file__).parent.parent
            / "thirdparty"
            / "weights"
            / "se2loftr_weights"
            / "8rot.ckpt"
        )  # Path to se2loftr weights
        self.matcher = LoFTR(config=_used_cfg)
        self.matcher.load_state_dict(torch.load(str(se2loftr_path))["state_dict"])
        self.matcher = self.matcher.eval().to(device=self._device)

        self.grayscale = True
        self.as_float = True
        self._quality = config["general"]["quality"]

    def _lower_config(self, yacs_cfg):
        if not isinstance(yacs_cfg, CN):
            return yacs_cfg
        return {k.lower(): self._lower_config(v) for k, v in yacs_cfg.items()}

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

        if self.as_float:
            image0 = image0.astype(np.float32)
            image1 = image1.astype(np.float32)

        if self.grayscale:
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            height0, width0 = image0.shape[:2]
            height1, width1 = image1.shape[:2]

        ## Resize images if needed
        # image0 = self._resize_image(self._quality, image0)
        # image1 = self._resize_image(self._quality, image1)

        # Inference with LoFTR and get prediction
        with torch.no_grad(), torch.inference_mode():
            img0_raw = cv2.resize(image0, (640, 480))
            img1_raw = cv2.resize(image1, (640, 480))
            img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.0
            img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.0
            batch = {"image0": img0, "image1": img1}
            self.matcher(batch)
            mkpts0 = batch["mkpts0_f"].cpu().numpy()
            mkpts1 = batch["mkpts1_f"].cpu().numpy()
            mconf = batch["mconf"].cpu().numpy()

            # Upscale features to original size
            mkpts0[:, 0] = mkpts0[:, 0] * width0 / 640
            mkpts0[:, 1] = mkpts0[:, 1] * height0 / 480
            mkpts1[:, 0] = mkpts1[:, 0] * width1 / 640
            mkpts1[:, 1] = mkpts1[:, 1] * height1 / 480

        features0 = FeaturesDict(keypoints=mkpts0)
        features1 = FeaturesDict(keypoints=mkpts1)

        # Get match confidence
        mconf = mconf

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
        image0: np.ndarray,
        image1: np.ndarray,
        tile_selection: TileSelection = TileSelection.PRESELECTION,
        **config,
    ):
        """
        Matches tiles in two images and returns the features, matches, and confidence.

        Args:
            image0: The first input image as a NumPy array.
            image1: The second input image as a NumPy array.
            tile_selection: The method for selecting tile pairs to match (default: TileSelection.PRESELECTION).
            **config: Additional keyword arguments for customization.

        Returns:
            A tuple containing:
            - features0: FeaturesBase object representing keypoints, descriptors, and scores of image0.
            - features1: FeaturesBase object representing keypoints, descriptors, and scores of image1.
            - matches0: NumPy array with indices of matched keypoints in image0.
            - mconf: NumPy array with confidence scores for the matches.

        """
        # Get config
        grid = config.get("grid", [1, 1])
        overlap = config.get("overlap", 0)
        origin = config.get("origin", [0, 0])
        do_viz_tiles = config.get("do_viz_tiles", False)

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
        conf_full = np.array([], dtype=np.float32)

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logger.info(f" - Matching tile pair ({tidx0}, {tidx1})")

            lim0 = t0_lims[tidx0]
            lim1 = t1_lims[tidx1]
            tile0 = self._tiler.extract_patch(image0, lim0)
            tile1 = self._tiler.extract_patch(image1, lim1)

            # Covert patch to tensor
            timg0_ = self._frame2tensor(tile0, self._device)
            timg1_ = self._frame2tensor(tile1, self._device)

            # Run inference
            with torch.inference_mode():
                input_dict = {"image0": timg0_, "image1": timg1_}
                correspondences = self.matcher(input_dict)

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

            # Plot matches on tile
            output_dir = config.get("output_dir", ".")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if do_viz_tiles is True:
                self.viz_matches_mpl(
                    tile0,
                    tile1,
                    mkpts0,
                    mkpts1,
                    output_dir / f"matches_tile_{tidx0}-{tidx1}.png",
                )

        logger.info("Restoring full image coordinates of matches...")

        # Restore original image coordinates (not cropped)
        mkpts0_full = mkpts0_full + np.array(t0_origin).astype("float32")
        mkpts1_full = mkpts1_full + np.array(t1_origin).astype("float32")

        # Select uniue features on image 0, on rounded coordinates
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
