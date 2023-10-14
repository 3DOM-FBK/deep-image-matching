import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.cm as cm
import numpy as np
import torch

from .consts import (
    GeometricVerification,
    Quality,
    TileSelection,
)
from .core import FeaturesBase, ImageMatcherBase, check_dict_keys
from .tiling import Tiler
from .thirdparty.LightGlue.lightglue import LightGlue, SuperPoint
from .thirdparty.SuperGlue.models.matching import Matching
from .thirdparty.SuperGlue.models.utils import make_matching_plot

logger = logging.getLogger(__name__)

# NOTE: the ImageMatcherBase class should be used as a base class for all the image matchers.
# The ImageMatcherBase class should contain all the common methods and attributes for all the matchers (tile suddivision, image downsampling/upsampling), geometric verification etc.
# The specific matchers MUST contain at least the `_match_pairs` method, which takes in two images as Numpy arrays, and returns the matches between keypoints and descriptors in those images. It doesn not care if the images are tiles or full-res images, as the tiling is handled by the ImageMatcherBase class that calls the `_match_pairs` method for each tile pair or for the full images depending on the tile selection method.

# TODO: divide the matching in two steps: one for the feature extractor and one for the matcher.
# TODO: allows the user to provide the features (keypoints, descriptors and scores) as input to match method when using SuperGlue/LightGlue (e.g., for tracking features in a new image of a sequence)
# TODO: move all the configuration parameters to the __init__ method of the ImageMatcherBase class. The match method should only take the images as input (and optionally the already extracted features).
# TODO: add integration with KORNIA library for using all the extractors and mathers.
# TODO: add visualization functions for the matches (take the functions from the visualization module of ICEpy4d). Currentely, the visualization methods may not work!


class LightGlueMatcher(ImageMatcherBase):
    def __init__(self, **config) -> None:
        """Initializes a LightGlueMatcher with Kornia"""

        self._localfeatures = config.get("features", "superpoint")
        super().__init__(**config)

    # Override _frame2tensor method to shift channel first as batch dimension
    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Normalize the image tensor and reorder the dimensions."""
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        elif image.ndim == 2:
            image = image[None]  # add channel axis
        else:
            raise ValueError(f"Not an image: {image.shape}")
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)

    def _rbd(self, data: dict) -> dict:
        """Remove batch dimension from elements in data"""
        return {
            k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()
        }

    def _match_pairs(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        **config,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray]: a tuple containing the features of the first image, the features of the second image, and the matches between them
        """

        max_keypoints = config.get("max_keypoints", 4096)
        resize = config.get("resize", None)

        image0_ = self._frame2tensor(image0, self._device)
        image1_ = self._frame2tensor(image1, self._device)

        device = torch.device(self._device if torch.cuda.is_available() else "cpu")

        # load the extractor
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        # load the matcher
        self.matcher = LightGlue(features=self._localfeatures).eval().to(device)

        with torch.inference_mode():
            # extract the features
            try:
                feats0 = self.extractor.extract(image0_, resize=resize)
                feats1 = self.extractor.extract(image1_, resize=resize)
            except:
                feats0 = self.extractor.extract(image0_)
                feats1 = self.extractor.extract(image1_)

            # match the features
            matches01 = self.matcher({"image0": feats0, "image1": feats1})

            # remove batch dimension
            feats0, feats1, matches01 = [
                self._rbd(x) for x in [feats0, feats1, matches01]
            ]

        feats0 = {k: v.cpu().numpy() for k, v in feats0.items()}
        feats1 = {k: v.cpu().numpy() for k, v in feats1.items()}
        matches01 = {
            k: v.cpu().numpy()
            for k, v in matches01.items()
            if isinstance(v, torch.Tensor)
        }

        # Create FeaturesBase objects and matching array
        features0 = FeaturesBase(
            keypoints=feats0["keypoints"],
            descriptors=feats0["descriptors"].T,
            scores=feats0["keypoint_scores"],
        )
        features1 = FeaturesBase(
            keypoints=feats1["keypoints"],
            descriptors=feats1["descriptors"].T,
            scores=feats1["keypoint_scores"],
        )
        matches0 = matches01["matches0"]
        mconf = matches01["scores"]

        # # For debugging
        # def print_shapes_in_dict(dic: dict):
        #     for k, v in dic.items():
        #         shape = v.shape if isinstance(v, np.ndarray) else None
        #         print(f"{k} shape: {shape}")

        # def print_features_shape(features: FeaturesBase):
        #     print(f"keypoints: {features.keypoints.shape}")
        #     print(f"descriptors: {features.descriptors.shape}")
        #     print(f"scores: {features.scores.shape}")

        return features0, features1, matches0, mconf


class SuperGlueMatcher(ImageMatcherBase):
    def __init__(self, opt: dict) -> None:
        """Initializes a SuperGlueMatcher object with the given options dictionary.

        The options dictionary should contain the following keys:

        - 'weights': defines the type of the weights used for SuperGlue inference. It can be either "indoor" or "outdoor". Default value is "outdoor".
        - 'keypoint_threshold': threshold for the SuperPoint keypoint detector
        - 'max_keypoints': maximum number of keypoints to extract with the SuperPoint detector. Default value is 0.001.
        - 'match_threshold': threshold for the SuperGlue feature matcher
        - 'force_cpu': whether to force using the CPU for inference

        Args:
            opt (dict): a dictionary of options for configuring the SuperGlueMatcher object

        Raises:
            KeyError: if one or more required options are missing from the options dictionary
            FileNotFoundError: if the specified SuperGlue model weights file cannot be found

        """
        opt = self._build_superglue_config(opt)
        super().__init__(opt)

        # initialize the Matching object with given configuration
        self.matcher = Matching(self._opt).eval().to(self._device)

    def _build_superglue_config(self, opt: dict) -> dict:
        # SuperPoint and SuperGlue default parameters
        NMS_RADIUS = 3
        SINKHORN_ITERATIONS = 20

        def_opt = {
            "weights": "outdoor",
            "keypoint_threshold": 0.001,
            "max_keypoints": -1,
            "match_threshold": 0.3,
            "force_cpu": False,
            "nms_radius": NMS_RADIUS,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
        }
        opt = {**def_opt, **opt}
        required_keys = [
            "weights",
            "keypoint_threshold",
            "max_keypoints",
            "match_threshold",
            "force_cpu",
        ]
        check_dict_keys(opt, required_keys)

        return {
            "superpoint": {
                "nms_radius": opt["nms_radius"],
                "keypoint_threshold": opt["keypoint_threshold"],
                "max_keypoints": opt["max_keypoints"],
            },
            "superglue": {
                "weights": opt["weights"],
                "sinkhorn_iterations": opt["sinkhorn_iterations"],
                "match_threshold": opt["match_threshold"],
            },
            "force_cpu": opt["force_cpu"],
        }

    def _match_pairs(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        **config,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images (no matter if they are tiles or full-res images) using the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns the matches between keypoints
        and descriptors in those images using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]: a tuple containing the features of the first image, the features of the second image, the matches between them and the match confidence.
        """

        if len(image0.shape) > 2:
            image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        if len(image1.shape) > 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

        tensor0 = self._frame2tensor(image0, self._device)
        tensor1 = self._frame2tensor(image1, self._device)

        with torch.inference_mode():
            pred_tensor = self.matcher({"image0": tensor0, "image1": tensor1})
        pred = {k: v[0].cpu().numpy() for k, v in pred_tensor.items()}

        # Create FeaturesBase objects and matching array
        features0 = FeaturesBase(
            keypoints=pred["keypoints0"],
            descriptors=pred["descriptors0"],
            scores=pred["scores0"],
        )
        features1 = FeaturesBase(
            keypoints=pred["keypoints1"],
            descriptors=pred["descriptors1"],
            scores=pred["scores1"],
        )
        matches0 = pred["matches0"]

        # Create a match confidence array
        valid = matches0 > -1
        mconf = features0.scores[valid]

        return features0, features1, matches0, mconf

    def viz_matches(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        path: str,
        fast_viz: bool = False,
        show_keypoints: bool = False,
        opencv_display: bool = False,
    ) -> None:
        """
        Visualize the matching result between two images.

        Args:
        - path (str): The output path to save the visualization.
        - fast_viz (bool): Whether to use preselection visualization method.
        - show_keypoints (bool): Whether to show the detected keypoints.
        - opencv_display (bool): Whether to use OpenCV for displaying the image.

        Returns:
        - None

        TODO: replace make_matching_plot with native a function implemented in icepy4D
        """

        assert self._mkpts0 is not None, "Matches not available."
        # image0 = np.uint8(tensor0.cpu().numpy() * 255),

        color = cm.jet(self._mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(
                len(self._mkpts0),
                len(self._mkpts1),
            ),
            "Matches: {}".format(len(self._mkpts0)),
        ]

        # Display extra parameter info.
        k_thresh = self._opt["superpoint"]["keypoint_threshold"]
        m_thresh = self._opt["superglue"]["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
        ]

        make_matching_plot(
            image0,
            image1,
            self._mkpts0,
            self._mkpts1,
            self._mkpts0,
            self._mkpts1,
            color,
            text,
            path=path,
            show_keypoints=show_keypoints,
            fast_viz=fast_viz,
            opencv_display=opencv_display,
            opencv_title="Matches",
            small_text=small_text,
        )


class LOFTRMatcher(ImageMatcherBase):
    def __init__(self, opt: dict = {}) -> None:
        """Initializes a LOFTRMatcher with Kornia object with the given options dictionary."""

        opt = self._build_config(opt)
        super().__init__(opt)

        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device).eval()

    def _build_config(self, opt: dict) -> dict:
        def_opt = {
            "pretrained": "outdoor",
            "force_cpu": False,
        }
        opt = {**def_opt, **opt}
        required_keys = [
            "pretrained",
            "force_cpu",
        ]
        check_dict_keys(opt, required_keys)

        return opt

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        image = K.image_to_tensor(np.array(image), False).float() / 255.0
        image = K.color.bgr_to_rgb(image.to(device))
        if image.shape[1] > 2:
            image = K.color.rgb_to_grayscale(image)
        return image

    def _match_pairs(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        **config,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
        """Matches keypoints and descriptors in two given images
        (no matter if they are tiles or full-res images) using
        the SuperGlue algorithm.

        This method takes in two images as Numpy arrays, and returns
        the matches between keypoints and descriptors in those images
        using the SuperGlue algorithm.

        Args:
            image0 (np.ndarray): the first image to match, as Numpy array
            image1 (np.ndarray): the second image to match, as Numpy array

        Returns:
            Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]: a
            tuple containing the features of the first image, the features
            of the second image, the matches between them and the match
            onfidence.
        """

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
        features0 = FeaturesBase(keypoints=mkpts0)
        features1 = FeaturesBase(keypoints=mkpts1)

        # Get match confidence
        mconf = correspondences["confidence"].cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0.shape[0])

        return features0, features1, matches0, mconf

    def _match_by_tile(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        tile_selection: TileSelection = TileSelection.PRESELECTION,
        **config,
    ) -> Tuple[FeaturesBase, FeaturesBase, np.ndarray, np.ndarray]:
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
            save_dir = config.get("save_dir", ".")
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if do_viz_tiles is True:
                self.viz_matches_mpl(
                    tile0,
                    tile1,
                    mkpts0,
                    mkpts1,
                    save_dir / f"matches_tile_{tidx0}-{tidx1}.png",
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
        features0 = FeaturesBase(keypoints=mkpts0_full)
        features1 = FeaturesBase(keypoints=mkpts1_full)

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        logger.info("Matching by tile completed.")

        return features0, features1, matches0, conf_full


if __name__ == "__main__":
    from .logger import setup_logger

    setup_logger()

    # assset_path = Path("assets")

    # im_path0 = assset_path / "img/cam1/IMG_2637.jpg"
    # im_path1 = assset_path / "img/cam2/IMG_1112.jpg"

    img_idx = 20
    outdir = "sandbox/matching_results"

    folders = [Path("data/img/p1"), Path("data/img/p2")]
    imlists = [sorted(f.glob("*.jpg")) for f in folders]
    im_path0 = imlists[0][img_idx]
    im_path1 = imlists[1][img_idx]
    img0 = cv2.imread(str(im_path0))
    img1 = cv2.imread(str(im_path1))
    outdir = Path(outdir)
    if outdir.exists():
        os.system(f"rm -rf {outdir}")

    # Test LightGlue
    matcher = LightGlueMatcher()
    matcher.match(
        img0,
        img1,
        quality=Quality.HIGH,
        tile_selection=TileSelection.PRESELECTION,
        grid=[2, 3],
        overlap=100,
        origin=[0, 0],
        min_matches_per_tile=3,
        max_keypoints=10240,
        do_viz_matches=True,
        do_viz_tiles=True,
        fast_viz=False,
        save_dir=outdir / "LIGHTGLUE",
        geometric_verification=GeometricVerification.PYDEGENSAC,
        threshold=2,
        confidence=0.9999,
    )
    mm = matcher.mkpts0

    # SuperGlue
    suerglue_cfg = {
        "weights": "outdoor",
        "keypoint_threshold": 0.001,
        "max_keypoints": 4096,
        "match_threshold": 0.1,
        "force_cpu": False,
    }
    matcher = SuperGlueMatcher(suerglue_cfg)
    tile_selection = TileSelection.PRESELECTION
    matcher.match(
        img0,
        img1,
        quality=Quality.HIGH,
        tile_selection=tile_selection,
        grid=[2, 3],
        overlap=200,
        origin=[0, 0],
        do_viz_matches=True,
        do_viz_tiles=True,
        save_dir=outdir / "superglue_PRESELECTION",
        geometric_verification=GeometricVerification.PYDEGENSAC,
        threshold=2,
        confidence=0.9999,
    )

    # Test LOFTR
    grid = [5, 4]
    overlap = 50
    origin = [0, 0]
    matcher = LOFTRMatcher()
    matcher.match(
        img0,
        img1,
        quality=Quality.HIGH,
        tile_selection=TileSelection.PRESELECTION,
        grid=grid,
        overlap=overlap,
        origin=origin,
        do_viz_matches=True,
        do_viz_tiles=True,
        save_dir=outdir / "LOFTR",
        geometric_verification=GeometricVerification.PYDEGENSAC,
        threshold=2,
        confidence=0.9999,
    )

    # tile_selection = TileSelection.GRID
    # matcher.match(
    #     img0,
    #     img1,
    #     tile_selection=tile_selection,
    #     grid=grid,
    #     overlap=overlap,
    #     origin=origin,
    #     do_viz_matches=True,
    #     do_viz_tiles=True,
    #     save_dir=outdir / str(tile_selection).split(".")[1],
    #     geometric_verification=GeometricVerification.PYDEGENSAC,
    #     threshold=1,
    #     confidence=0.9999,
    # )

    # tile_selection = TileSelection.EXHAUSTIVE
    # matcher.match(
    #     img0,
    #     img1,
    #     tile_selection=tile_selection,
    #     grid=grid,
    #     overlap=overlap,
    #     origin=origin,
    #     do_viz_matches=True,
    #     do_viz_tiles=True,
    #     save_dir=outdir / str(tile_selection).split(".")[1],
    #     geometric_verification=GeometricVerification.PYDEGENSAC,
    #     threshold=1,
    #     confidence=0.9999,
    # )

    print("Matching succeded.")
