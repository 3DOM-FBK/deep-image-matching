import logging
from importlib import import_module
from pathlib import Path
from typing import Optional, TypedDict

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia import feature

from .consts import TileSelection
from .matcher_base import MatcherBase
from .tiling import Tiler

logger = logging.getLogger(__name__)

# NOTE: the MatcherBase class should be used as a base class for all the image matchers.
# The MatcherBase class should contain all the common methods and attributes for all the matchers (tile suddivision, image downsampling/upsampling), geometric verification etc.
# The specific matchers MUST contain at least the `_match_pairs` method, which takes in two images as Numpy arrays, and returns the matches between keypoints and descriptors in those images. It doesn not care if the images are tiles or full-res images, as the tiling is handled by the MatcherBase class that calls the `_match_pairs` method for each tile pair or for the full images depending on the tile selection method.


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray]
    tile_idx: Optional[np.ndarray]


class DetectAndDescribe(MatcherBase):
    def __init__(self, **config) -> None:
        super().__init__(**config["general"])

    def kornia_matcher(
        self, desc_0, desc_1, approach="smnn", ratio_threshold=0.99
    ):  #'nn' or 'snn' or 'mnn' or 'smnn'
        torch_desc_0 = torch.from_numpy(desc_0)
        torch_desc_1 = torch.from_numpy(desc_1)
        matcher = feature.DescriptorMatcher(match_mode=approach, th=ratio_threshold)
        match_distances, matches_matrix = matcher.forward(torch_desc_0, torch_desc_1)
        return matches_matrix

    def _match_pairs(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        **config,
    ):
        # max_keypoints = config.get("max_keypoints", 4096)
        local_feat_extractor = config.get("local_feat_extractor")
        keypoints, descriptors, lafs = local_feat_extractor.run(
            image0, image1, config["w_size"]
        )
        kpys0 = keypoints[0]
        kpys1 = keypoints[1]
        desc0 = descriptors[0]
        desc1 = descriptors[1]

        matches_matrix = self.kornia_matcher(
            desc0, desc1, config["kornia_matcher"], config["ratio_threshold"]
        )

        matches_matrix = matches_matrix.numpy()

        matches0 = []
        matches_matrix_col0 = matches_matrix[:, 0]
        matches_matrix_col1 = matches_matrix[:, 1]
        for i in range(kpys0.shape[0]):
            if i in matches_matrix_col0:
                j = np.where(matches_matrix_col0 == i)[0]
                matches0.append(matches_matrix_col1[j][0])
            else:
                matches0.append(-1)
        matches0 = np.array(matches0)
        mconf = None

        # Create FeaturesDict objects and matching array
        features0 = FeaturesDict(
            keypoints=kpys0,
            descriptors=desc0,
            scores=None,
        )
        features1 = FeaturesDict(
            keypoints=kpys1,
            descriptors=desc1,
            scores=None,
        )

        matches_dict = {
            "matches0": matches0,
            "matches01": {},
        }

        return features0, features1, matches_dict, mconf


class LightGlueMatcher(MatcherBase):
    def __init__(self, **config) -> None:
        """Initializes a LightGlueMatcher"""

        self._localfeatures = "superpoint"
        super().__init__(**config)

        # load the LightGlue module
        LG = import_module("deep_image_matching.thirdparty.LightGlue.lightglue")

        # load the matcher
        sg_cfg = self._config["SperPoint+LightGlue"]["LightGlue"]
        self._matcher = (
            LG.LightGlue(self._localfeatures, **sg_cfg).eval().to(self._device)
        )

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
        def featuresDict_2_lightglue(feats: FeaturesDict, device: torch.device) -> dict:
            # Remove elements from list/tuple
            feats = {
                k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()
            }
            # Move descriptors dimension to last
            if "descriptors" in feats.keys():
                if feats["descriptors"].shape[-1] != 256:
                    feats["descriptors"] = feats["descriptors"].T
            # Add batch dimension
            feats = {k: v[None] for k, v in feats.items()}
            # Convert to tensor
            feats = {
                k: torch.tensor(v, dtype=torch.float, device=device)
                for k, v in feats.items()
            }
            # Check device
            feats = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in feats.items()
            }

            return feats

        feats0 = featuresDict_2_lightglue(feats0, self._device)
        feats1 = featuresDict_2_lightglue(feats1, self._device)

        # match the features
        match_res = self._matcher({"image0": feats0, "image1": feats1})

        # remove batch dimension and convert to numpy
        mfeats0, mfeats1, matches01 = [
            self._rbd(x) for x in [feats0, feats1, match_res]
        ]
        match_res = {
            k: v.cpu().numpy()
            for k, v in matches01.items()
            if isinstance(v, torch.Tensor)
        }

        # get matching array (indices of matched keypoints in image0 and image1)
        matches01_idx = match_res["matches"]

        return matches01_idx

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


class SuperGlueMatcher(MatcherBase):
    def __init__(self, **config) -> None:
        """Initializes a SuperGlueMatcher object with the given options dictionary."""

        raise NotImplementedError(
            "SuperGlueMatcher is not correctely implemented yet. It needs to be updated to the new version of the Matcher. Please use LightGlue in the meanwhile!"
        )

        super().__init__(**config)

        SG = import_module("deep_image_matching.thirdparty.SuperGlue.models.matching")

        # initialize the Matching object with given configuration
        self.matcher = (
            SG.Matching(config["SperPoint+SuperGlue"]["superglue"])
            .eval()
            .to(self._device)
        )

    def _match_pairs(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
    ):
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
        features0 = FeaturesDict(
            keypoints=pred["keypoints0"],
            descriptors=pred["descriptors0"],
            scores=pred["scores0"],
        )
        features1 = FeaturesDict(
            keypoints=pred["keypoints1"],
            descriptors=pred["descriptors1"],
            scores=pred["scores1"],
        )

        # get matching array
        matches0 = pred["matches0"]

        # Create a match confidence array
        valid = matches0 > -1
        mconf = features0.scores[valid]

        # matches_dict = {
        #     "matches0": matches0,
        #     "matches01": None,
        # }

        return features0, features1, matches0, mconf

    # def viz_matches(
    #     self,
    #     image0: np.ndarray,
    #     image1: np.ndarray,
    #     path: str,
    #     fast_viz: bool = False,
    #     show_keypoints: bool = False,
    #     opencv_display: bool = False,
    # ) -> None:
    #     """
    #     Visualize the matching result between two images.

    #     Args:
    #     - path (str): The output path to save the visualization.
    #     - fast_viz (bool): Whether to use preselection visualization method.
    #     - show_keypoints (bool): Whether to show the detected keypoints.
    #     - opencv_display (bool): Whether to use OpenCV for displaying the image.

    #     Returns:
    #     - None

    #     TODO: replace make_matching_plot with native a function implemented in icepy4D
    #     """

    #     assert self._mkpts0 is not None, "Matches not available."
    #     # image0 = np.uint8(tensor0.cpu().numpy() * 255),

    #     color = cm.jet(self._mconf)
    #     text = [
    #         "SuperGlue",
    #         "Keypoints: {}:{}".format(
    #             len(self._mkpts0),
    #             len(self._mkpts1),
    #         ),
    #         "Matches: {}".format(len(self._mkpts0)),
    #     ]

    #     # Display extra parameter info.
    #     k_thresh = self._opt["superpoint"]["keypoint_threshold"]
    #     m_thresh = self._opt["superglue"]["match_threshold"]
    #     small_text = [
    #         "Keypoint Threshold: {:.4f}".format(k_thresh),
    #         "Match Threshold: {:.2f}".format(m_thresh),
    #     ]

    #     make_matching_plot(
    #         image0,
    #         image1,
    #         self._mkpts0,
    #         self._mkpts1,
    #         self._mkpts0,
    #         self._mkpts1,
    #         color,
    #         text,
    #         path=path,
    #         show_keypoints=show_keypoints,
    #         fast_viz=fast_viz,
    #         opencv_display=opencv_display,
    #         opencv_title="Matches",
    #         small_text=small_text,
    #     )


class LOFTRMatcher(MatcherBase):
    def __init__(self, **config) -> None:
        """Initializes a LOFTRMatcher with Kornia object with the given options dictionary."""

        raise NotImplementedError(
            "LOFTR is not correctely implemented yet. It needs to be updated to the new version of the Matcher. Please use LightGlue in the meanwhile!"
        )

        super().__init__(**config["general"])

        self.matcher = (
            KF.LoFTR(pretrained=config["loftr"]["pretrained"]).to(self.device).eval()
        )

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
    ):
        """Matches keypoints and descriptors in two given images
        (no matter if they are tiles or full-res images) using
        the LoFTR algorithm.

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
        features0 = FeaturesDict(keypoints=mkpts0)
        features1 = FeaturesDict(keypoints=mkpts1)

        # Get match confidence
        mconf = correspondences["confidence"].cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))

        matches_dict = {
            "matches0": matches0,
            "matches01": matches,
        }

        return features0, features1, matches_dict, mconf

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
        features0 = FeaturesDict(keypoints=mkpts0_full)
        features1 = FeaturesDict(keypoints=mkpts1_full)

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])

        logger.info("Matching by tile completed.")

        return features0, features1, matches0, conf_full
