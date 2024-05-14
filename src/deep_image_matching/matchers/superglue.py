import numpy as np
import torch

from ..thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue
from .matcher_base import FeaturesDict, MatcherBase


def features_2_sg(feats0: FeaturesDict, feats1: FeaturesDict, device: torch.device) -> dict:
    # Merge feats0 and feats1 in a single dict
    data = {}
    data = {**data, **{k + "0": v for k, v in feats0.items()}}
    data = {**data, **{k + "1": v for k, v in feats1.items()}}
    if "feature_path0" in data.keys():
        del data["feature_path0"]
    if "feature_path1" in data.keys():
        del data["feature_path1"]
    if "im_path0" in data.keys():
        del data["im_path0"]
    if "im_path1" in data.keys():
        del data["im_path1"]
    data["image0"] = np.empty(data["image_size0"])
    data["image1"] = np.empty(data["image_size1"])
    data["image0"] = data["image0"][None]
    data["image1"] = data["image1"][None]

    # Add batch dimension
    data = {k: v[None] for k, v in data.items()}

    # Convert to tensor
    data = {k: torch.tensor(v, dtype=torch.float, device=device) for k, v in data.items()}

    # Add channel dimension if missing
    for i in range(2):
        s = data[f"image_size{i}"].cpu().numpy().astype(int).squeeze()
        data[f"image_size{i}"] = torch.Size((1, 1, s[0], s[1]))

    return data


def correspondence_matrix_from_matches0(kpts_number: int, matches0: np.ndarray) -> np.ndarray:
    n_tie_points = np.arange(kpts_number).reshape((-1, 1))
    matrix = np.hstack((n_tie_points, matches0.reshape((-1, 1))))
    correspondences = matrix[~np.any(matrix == -1, axis=1)]

    return correspondences


class SuperGlueMatcher(MatcherBase):
    default_config = {
        "name": "superglue",
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.3,
    }
    required_inputs = []
    min_matches = 20
    max_feat_no_tiling = 50000

    def __init__(self, config) -> None:
        """Initializes a SuperGlueMatcher object with the given options dictionary."""

        super().__init__(config)

        # initialize the Matching object with given configuration
        cfg = {**self._default_conf, **self.config.get("matcher", {})}
        self._matcher = SuperGlue(cfg).eval().to(self._device)

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

        # Check that the image size is rprovided into the features
        data = features_2_sg(feats0, feats1, self._device)

        match_res = self._matcher(data)
        match_res = {k: v.cpu().numpy() for k, v in match_res.items() if isinstance(v, torch.Tensor)}

        # Make correspondence matrix from matches0
        matches0 = match_res["matches0"]
        kpts_number = feats0["keypoints"].shape[0]
        correspondences = correspondence_matrix_from_matches0(kpts_number, matches0)

        return correspondences
