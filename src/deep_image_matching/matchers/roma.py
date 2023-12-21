from pathlib import Path
from typing import Tuple

import cv2
import h5py
import kornia as K
import numpy as np
import torch
from PIL import Image

from .. import Timer, logger
from ..io.h5 import get_features
from ..thirdparty.RoMa.roma import roma_outdoor
from ..utils.consts import Quality, TileSelection
from .matcher_base import DetectorFreeMatcherBase, FeaturesDict


class RomaMatcher(DetectorFreeMatcherBase):
    """
    RomaMatcher class for feature matching using RoMa.

    Attributes:
        default_conf (dict): Default configuration options.
        max_feat_no_tiling (int): Maximum number of features when tiling is not used.
        grayscale (bool): Flag indicating whether images are processed in grayscale.
        as_float (bool): Flag indicating whether to use float for image pixel values.

    Methods:
        __init__: Constructor.
        match(self, feature_path: Path, matches_path: Path, img0: Path, img1: Path, try_full_image: bool = False) -> np.ndarray: Match features between two images.
    """

    grayscale = True
    as_float = True
    max_tile_size = 448

    def __init__(self, config={}) -> None:
        super().__init__(config)

        self.matcher = roma_outdoor(device=self._device)

        self._quality = config["general"]["quality"]

        # If tiling is used, extract tiles with proper size for RoMa matching and save them to disk
        if self._tiling != TileSelection.NONE:
            pass

    def match(
        self,
        feature_path: Path,
        matches_path: Path,
        img0: Path,
        img1: Path,
        try_full_image: bool = False,
    ):
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
            matches = self._match_by_tile(
                img0,
                img1,
                features0,
                features1,
                method=self._tiling,
                select_unique=True,
            )
            timer_match.update("[match] try to match by tile")

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

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ):
        """
        Perform matching between feature pairs.

        Args:
            feats0 (FeaturesDict): Features dictionary for the first image.
            feats1 (FeaturesDict): Features dictionary for the second image.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.
        """

        feature_path = feats0["feature_path"]

        img0_path = feats0["im_path"]
        img0_name = Path(img0_path).name
        img1_path = feats1["im_path"]
        img1_name = Path(img1_path).name

        # Run inference
        with torch.inference_mode():
            # /home/luca/Desktop/gitprojects/github_3dom/deep-image-matching/src/deep_image_matching/thirdparty/RoMa/roma/models/matcher.py
            # in class RegressionMatcher(nn.Module) def __init__ hardcoded self.upsample_res = (int(864/4), int(864/4))
            W_A, H_A = Image.open(img0_path).size
            W_B, H_B = Image.open(img1_path).size

            warp, certainty = self.matcher.match(
                str(img0_path), str(img1_path), device=self._device, batched=False
            )
            matches, certainty = self.matcher.sample(warp, certainty)
            kptsA, kptsB = self.matcher.to_pixel_coordinates(
                matches, H_A, W_A, H_B, W_B
            )
            kptsA, kptsB = kptsA.cpu().numpy(), kptsB.cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(kptsA.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))
        self._update_features_h5(
            feature_path,
            img0_name,
            img1_name,
            kptsA,
            kptsB,
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

        raise NotImplementedError(
            "Matching by tile is not implemented for Roma.yet. Please match full images (downscale images using a lower 'Quality' if you run out of memory)."
        )

        # # Get config
        # grid = config.get("grid", [1, 1])
        # overlap = config.get("overlap", 0)
        # origin = config.get("origin", [0, 0])
        # do_viz_tiles = config.get("do_viz_tiles", False)

        # # Compute tiles limits and origin
        # self._tiler = Tiler(grid=grid, overlap=overlap, origin=origin)
        # t0_lims, t0_origin = self._tiler.compute_limits_by_grid(image0)
        # t1_lims, t1_origin = self._tiler.compute_limits_by_grid(image1)

        # # Select tile pairs to match
        # tile_pairs = self._tile_selection(
        #     image0, image1, t0_lims, t1_lims, tile_selection, config=config
        # )

        # # Initialize empty array for storing matched keypoints, descriptors and scores
        # mkpts0_full = np.array([], dtype=np.float32).reshape(0, 2)
        # mkpts1_full = np.array([], dtype=np.float32).reshape(0, 2)
        # conf_full = np.array([], dtype=np.float32)

        # # Match each tile pair
        # for tidx0, tidx1 in tile_pairs:
        #     logger.info(f" - Matching tile pair ({tidx0}, {tidx1})")

        #     lim0 = t0_lims[tidx0]
        #     lim1 = t1_lims[tidx1]
        #     tile0 = self._tiler.extract_patch(image0, lim0)
        #     tile1 = self._tiler.extract_patch(image1, lim1)

        #     # Covert patch to tensor
        #     timg0_ = self._frame2tensor(tile0, self._device)
        #     timg1_ = self._frame2tensor(tile1, self._device)

        #     # Run inference
        #     with torch.inference_mode():
        #         input_dict = {"image0": timg0_, "image1": timg1_}
        #         correspondences = self.matcher(input_dict)

        #     # Get matches and build features
        #     mkpts0 = correspondences["keypoints0"].cpu().numpy()
        #     mkpts1 = correspondences["keypoints1"].cpu().numpy()

        #     # Get match confidence
        #     conf = correspondences["confidence"].cpu().numpy()

        #     # Append to full arrays
        #     mkpts0_full = np.vstack(
        #         (mkpts0_full, mkpts0 + np.array(lim0[0:2]).astype("float32"))
        #     )
        #     mkpts1_full = np.vstack(
        #         (mkpts1_full, mkpts1 + np.array(lim1[0:2]).astype("float32"))
        #     )
        #     conf_full = np.concatenate((conf_full, conf))

        #     # Plot matches on tile
        #     output_dir = config.get("output_dir", ".")
        #     output_dir = Path(output_dir)
        #     output_dir.mkdir(parents=True, exist_ok=True)
        #     if do_viz_tiles is True:
        #         self.viz_matches_mpl(
        #             tile0,
        #             tile1,
        #             mkpts0,
        #             mkpts1,
        #             output_dir / f"matches_tile_{tidx0}-{tidx1}.png",
        #         )

        # logger.info("Restoring full image coordinates of matches...")

        # # Restore original image coordinates (not cropped)
        # mkpts0_full = mkpts0_full + np.array(t0_origin).astype("float32")
        # mkpts1_full = mkpts1_full + np.array(t1_origin).astype("float32")

        # # Select uniue features on image 0, on rounded coordinates
        # decimals = 1
        # _, unique_idx = np.unique(
        #     np.round(mkpts0_full, decimals), axis=0, return_index=True
        # )
        # mkpts0_full = mkpts0_full[unique_idx]
        # mkpts1_full = mkpts1_full[unique_idx]
        # conf_full = conf_full[unique_idx]

        # # Create features
        # features0 = FeaturesDict(keypoints=mkpts0_full)
        # features1 = FeaturesDict(keypoints=mkpts1_full)

        # # Create a 1-to-1 matching array
        # matches0 = np.arange(mkpts0_full.shape[0])

        # logger.info("Matching by tile completed.")

        # return features0, features1, matches0, conf_full

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
        elif quality == Quality.LOWEST:
            image_ = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(image)))
        return image_

    def _resize_features(self, quality: Quality, keypoints: np.ndarray) -> np.ndarray:
        """
        Resize features based on the specified quality.

        Args:
            quality (Quality): The quality level for resizing.
            features0 (np.ndarray): The array of keypoints.

        Returns:
            np.ndarray: Resized keypoints.

        """
        if quality == Quality.HIGHEST:
            keypoints /= 2
        elif quality == Quality.HIGH:
            pass
        elif quality == Quality.MEDIUM:
            keypoints *= 2
        elif quality == Quality.LOW:
            keypoints *= 4
        elif quality == Quality.LOWEST:
            keypoints *= 8

        return keypoints

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        image = K.image_to_tensor(np.array(image), False).float() / 255.0
        image = K.color.bgr_to_rgb(image.to(device))
        if image.shape[1] > 2:
            image = K.color.rgb_to_grayscale(image)
        return image

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
