import logging
from pathlib import Path

import kornia as K
import numpy as np
import torch
from kornia import feature as KF

from ..constants import TileSelection, Timer
from ..utils.tiling import Tiler
from .matcher_base import DetectorFreeMatcherBase, tile_selection

logger = logging.getLogger("dim")


class LOFTRMatcher(DetectorFreeMatcherBase):
    """
    LOFTRMatcher class for feature matching using the LOFTR method.

    Attributes:
        _default_conf (dict): Default configuration options.
        max_feat_no_tiling (int): Maximum number of features when tiling is not used.
        grayscale (bool): Flag indicating whether images are processed in grayscale.
        as_float (bool): Flag indicating whether to use float for image pixel values.

    Methods:
        __init__(self, config={}): Initializes a LOFTRMatcher with Kornia object with the given options dictionary.
        match(self, feature_path: Path, matches_path: Path, img0: Path, img1: Path, try_full_image: bool = False) -> np.ndarray: Match features between two images.
        _match_pairs(self, feats0: FeaturesDict, feats1: FeaturesDict) -> np.ndarray: Perform matching between feature pairs.
        _match_by_tile(self, img0: Path, img1: Path, features0: FeaturesDict, features1: FeaturesDict, method: TileSelection = TileSelection.PRESELECTION, select_unique: bool = True) -> np.ndarray: Match features between two images using a tiling approach.
        _load_image_np(self, img_path): Load image as numpy array.
        _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor: Convert image to tensor.
        _resize_image(self, quality: Quality, image: np.ndarray) -> Tuple[np.ndarray]: Resize images based on the specified quality.
        _resize_keypoints(self, quality: Quality, keypoints: np.ndarray) -> np.ndarray: Resize features based on the specified quality.
        _update_features_h5(self, feature_path, im0_name, im1_name, new_keypoints0, new_keypoints1, matches0) -> np.ndarray: Update features in h5 file.
        viz_matches(self, feature_path: Path, matchings_path: Path, img0: Path, img1: Path, save_path: str = None, fast_viz: bool = True, interactive_viz: bool = False, **config) -> None: Visualize matches.
    """

    _default_conf = {"pretrained": "outdoor"}
    grayscale = False
    as_float = True
    min_matches = 100
    min_matches_per_tile = 3
    max_tile_size = 1200

    def __init__(self, config={}) -> None:
        """
        Initializes a LOFTRMatcher with Kornia object with the given options dictionary.

        Args:
            config (dict, optional): Configuration options. Defaults to an empty dictionary.
        """

        super().__init__(config)

        model = self.config["matcher"]["pretrained"]
        self.matcher = KF.LoFTR(pretrained=model).to(self._device).eval()

        tile_size = self.config["general"]["tile_size"]
        if max(tile_size) > self.max_tile_size:
            logger.warning(
                f"The tile size is too large large ({tile_size}) for running LOFTR. Using a maximum tile size of {self.max_tile_size} px (this may take some time...)."
            )
            ratio = max(tile_size) / self.max_tile_size
            self.config["general"]["tile_size"] = (
                int(tile_size[0] / ratio),
                int(tile_size[1] / ratio),
            )
        elif max(tile_size) > 1000:
            logger.warning(
                "The tile size is large, this may cause out-of-memory error during matching. You should consider using a smaller tile size or a lower image resolution."
            )

    @torch.no_grad()
    def _match_pairs(
        self,
        feature_path: Path,
        img0_path: Path,
        img1_path: Path,
    ):
        """
        Perform matching between two images using a detector-free matcher. It takes in two images as Numpy arrays, and returns the matches between keypoints and descriptors in those images. It also saves the updated features to the specified h5 file.

        Args:
            feature_path (Path): Path to the h5 feature file where to save the updated features.
            img0_path (Path): Path to the first image.
            img1_path (Path): Path to the second image.

        Returns:
            np.ndarray: Array containing the indices of matched keypoints.

        Raises:
            torch.cuda.OutOfMemoryError: If an out-of-memory error occurs while matching images.
        """

        img0_name = img0_path.name
        img1_name = img1_path.name

        # Load images
        image0 = self._load_image_np(img0_path)
        image1 = self._load_image_np(img1_path)

        # Resize images if needed
        image0_ = self._resize_image(self._quality, image0)
        image1_ = self._resize_image(self._quality, image1)

        # Covert images to tensor
        timg0_ = self._frame2tensor(image0_, self._device)
        timg1_ = self._frame2tensor(image1_, self._device)

        # Run inference
        try:
            with torch.inference_mode():
                input_dict = {"image0": timg0_, "image1": timg1_}
                correspondences = self.matcher(input_dict)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"Out of memory error while matching images {img0_name} and {img1_name}. Try using a lower quality level or use a tiling approach."
            )
            raise e

        # Get matches and build features
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        mkpts0 = self._resize_keypoints(self._quality, mkpts0)
        mkpts1 = self._resize_keypoints(self._quality, mkpts1)

        # Get match confidence
        mconf = correspondences["confidence"].cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))

        # Update features in h5 file
        matches = self._update_features_h5(
            feature_path,
            img0_name,
            img1_name,
            mkpts0,
            mkpts1,
            matches,
        )

        return matches

    def _match_by_tile(
        self,
        feature_path: Path,
        img0: Path,
        img1: Path,
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

        tile_size = self.config["general"]["tile_size"]
        overlap = self.config["general"]["tile_overlap"]

        # Select tile pairs to match
        tile_pairs = tile_selection(
            img0,
            img1,
            method=method,
            quality=self._quality,
            preselction_extractor=self._preselction_extractor,
            preselction_matcher=self._preselction_matcher,
            tile_size=tile_size,
            tile_overlap=overlap,
            tile_preselection_size=self.tile_preselection_size,
            min_matches_per_tile=self.min_matches_per_tile,
            device=self._device,
        )
        timer.update("tile selection")

        # If no tile pairs are selected, return an empty array
        if len(tile_pairs) == 0:
            logger.debug("No tile pairs selected.")
            matches = np.array([], dtype=np.int32).reshape(0, 2)
            return matches

        # Load images
        img0_name = img0.name
        img1_name = img1.name
        image0 = self._load_image_np(img0)
        image1 = self._load_image_np(img1)

        # Resize images if needed
        image0 = self._resize_image(self._quality, image0)
        image1 = self._resize_image(self._quality, image1)

        # Extract tiles
        tiler = Tiler(tiling_mode="size")
        tiles0, t_origins0, t_padding0 = tiler.compute_tiles_by_size(
            input=image0, window_size=tile_size, overlap=overlap
        )
        tiles1, t_origins1, t_padding1 = tiler.compute_tiles_by_size(
            input=image1, window_size=tile_size, overlap=overlap
        )

        # Initialize empty arrays
        mkpts0_full = np.array([], dtype=np.float32).reshape(0, 2)
        mkpts1_full = np.array([], dtype=np.float32).reshape(0, 2)

        # Match each tile pair
        for tidx0, tidx1 in tile_pairs:
            logger.debug(f"  - Matching tile pair ({tidx0}, {tidx1})")

            # Get tiles and covert to tensor
            timg0_ = self._frame2tensor(tiles0[tidx0], self._device)
            timg1_ = self._frame2tensor(tiles1[tidx1], self._device)

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

            # Get matches and restore original image coordinates
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()

            mkpts0_full = np.vstack((mkpts0_full, mkpts0 + np.array(t_origins0[tidx0])))
            mkpts1_full = np.vstack((mkpts1_full, mkpts1 + np.array(t_origins1[tidx1])))
            timer.update("match tile")

        # Check if any matched keypoints has negative coordinates (due to padding of the original image) and remove them
        mask = np.all(mkpts0_full > 0, axis=1) & np.all(mkpts1_full > 0, axis=1)
        mkpts0_full = mkpts0_full[mask]
        mkpts1_full = mkpts1_full[mask]

        # Retrieve original image coordinates if matching was performed on up/down-sampled images
        mkpts0_full = self._resize_keypoints(self._quality, mkpts0_full)
        mkpts1_full = self._resize_keypoints(self._quality, mkpts1_full)

        # Select uniue features on image 0, on rounded coordinates
        if select_unique is True:
            decimals = 1
            _, unique_idx = np.unique(np.round(mkpts0_full, decimals), axis=0, return_index=True)
            mkpts0_full = mkpts0_full[unique_idx]
            mkpts1_full = mkpts1_full[unique_idx]

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))
        matches = self._update_features_h5(
            feature_path,
            img0_name,
            img1_name,
            mkpts0_full,
            mkpts1_full,
            matches,
        )

        return matches

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        image = K.image_to_tensor(np.array(image), False).float() / 255.0
        image = image.to(device)
        if image.shape[1] > 2:
            image = K.color.bgr_to_rgb(image)
            image = K.color.rgb_to_grayscale(image)
        return image
