import logging
from importlib import import_module

import torch

from .matcher_base import FeaturesDict, MatcherBase

logger = logging.getLogger(__name__)


class SuperGlueMatcher(MatcherBase):
    default_config = {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.3,
    }

    def __init__(self, **config) -> None:
        """Initializes a SuperGlueMatcher object with the given options dictionary."""

        # raise NotImplementedError(
        #     "SuperGlueMatcher is not correctely implemented yet. It needs to be updated to the new version of the Matcher. Please use LightGlue in the meanwhile!"
        # )

        super().__init__(**config)

        SG = import_module("deep_image_matching.thirdparty.SuperGlue.models.matching")

        # initialize the Matching object with given configuration
        self._matcher = (
            SG.Matching(config["SuperPoint+SuperGlue"]["superglue"])
            .eval()
            .to(self._device)
        )

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ):
        """
        _match_pairs _summary_

        Args:
            feats0 (FeaturesDict): _description_
            feats1 (FeaturesDict): _description_

        Returns:
            _type_: _description_
        """

        def featuresDict_2_sg(feats: FeaturesDict, device: torch.device) -> dict:
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

        feats0_ = featuresDict_2_sg(feats0, self._device)
        feats1_ = featuresDict_2_sg(feats1, self._device)
        match_res = self._matcher({"image0": feats0_, "image1": feats1_})

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
