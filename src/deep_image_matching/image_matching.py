import cv2
import numpy as np
from pathlib import Path
import logging

from .pairs_generator import PairsGenerator
from .image import ImageList
from .matchers import (
    SuperGlueMatcher,
    LOFTRMatcher,
    LightGlueMatcher,
    DetectAndDescribe,
)
from .local_features import LocalFeatureExtractor
from .geometric_verification import geometric_verification
from .consts import GeometricVerification


logger = logging.getLogger(__name__)


def ApplyGeometricVer(kpts0, kpts1, matches):
    mkpts0 = kpts0[matches[:, 0]]
    mkpts1 = kpts1[matches[:, 1]]
    F, inlMask = geometric_verification(
        mkpts0,
        mkpts1,
    )
    return kpts0, kpts1, matches[inlMask]


def ReorganizeMatches(kpts_number, matches: dict) -> np.ndarray:
    for key in matches:
        if key == "matches0":
            n_tie_points = np.arange(kpts_number).reshape((-1, 1))
            matrix = np.hstack((n_tie_points, matches[key].reshape((-1, 1))))
            correspondences = matrix[~np.any(matrix == -1, axis=1)]
            return correspondences
        elif key == "matches01":
            correspondences = matches[key]
            return correspondences


class ImageMatching:
    def __init__(
        self,
        imgs_dir: Path,
        matching_strategy: str,
        pair_file: Path,
        retrieval_option: str,
        overlap: int,
        local_features: str,
        custom_config: dict,
        max_feat_numb: int,
    ):
        self.matching_strategy = matching_strategy
        self.pair_file = pair_file
        self.retrieval_option = retrieval_option
        self.overlap = overlap
        self.local_features = local_features
        self.custom_config = custom_config
        self.max_feat_numb = max_feat_numb
        self.keypoints = {}
        self.correspondences = {}

        # Initialize ImageList class
        self.image_list = ImageList(imgs_dir)
        images = self.image_list.img_names

        if len(images) == 0:
            raise ValueError(
                "Image folder empty. Supported formats: '.jpg', '.JPG', '.png'"
            )
        elif len(images) == 1:
            raise ValueError("Image folder must contain at least two images")

        # Initialize matcher
        # first_img = cv2.imread(str(self.image_list[0].absolute_path))
        # w_size = first_img.shape[1]
        # self.custom_config["general"]["w_size"] = w_size
        cfg = self.custom_config["general"]
        if self.local_features == "lightglue":
            self._matcher = LightGlueMatcher(**cfg)
        elif self.local_features == "superglue":
            self._matcher = SuperGlueMatcher(**cfg)
        elif self.local_features == "loftr":
            self._matcher = LOFTRMatcher(**cfg)
        elif self.local_features == "detect_and_describe":
            self.custom_config["ALIKE"]["n_limit"] = self.max_feat_numb
            detector_and_descriptor = self.custom_config["general"][
                "detector_and_descriptor"
            ]
            local_feat_conf = self.custom_config[detector_and_descriptor]
            local_feat_extractor = LocalFeatureExtractor(
                detector_and_descriptor,
                local_feat_conf,
                self.max_feat_numb,
            )
            self._matcher = DetectAndDescribe(**cfg)
            cfg["general"]["local_feat_extractor"] = local_feat_extractor
        else:
            raise ValueError(
                "Invalid local feature extractor. Supported extractors: lightglue, superglue, loftr, detect_and_describe"
            )

    @property
    def img_format(self):
        return self.image_list.img_format

    @property
    def width(self):
        return self.image_list.width

    @property
    def height(self):
        return self.image_list.height

    def img_names(self):
        return self.image_list.img_names

    def generate_pairs(self):
        self.pairs = []
        if self.pair_file is not None and self.matching_strategy == "custom_pairs":
            with open(self.pair_file, "r") as txt_file:
                lines = txt_file.readlines()
                for line in lines:
                    im1, im2 = line.strip().split(" ", 1)
                    self.pairs.append((im1, im2))
        else:
            pairs_generator = PairsGenerator(
                self.image_list.img_paths,
                self.matching_strategy,
                self.retrieval_option,
                self.overlap,
            )
            self.pairs = pairs_generator.run()

        return self.pairs

    def match_pairs(self):
        for pair in self.pairs:
            logger.info(f"Matching image pair: {pair[0].name} - {pair[1].name}")
            im0 = pair[0]
            im1 = pair[1]
            image0 = cv2.imread(str(im0), cv2.COLOR_RGB2BGR)
            image1 = cv2.imread(str(im1), cv2.COLOR_RGB2BGR)

            if len(image0.shape) == 2:
                image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
            if len(image1.shape) == 2:
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)

            # By Luca
            # features0, features1, matches, mconf = self._matcher.match(
            #     image0,
            #     image1,
            #     geometric_verification=GeometricVerification.NONE,
            # )
            # ktps0 = features0.keypoints
            # ktps1 = features1.keypoints
            # self.keypoints[im0.name] = ktps0
            # self.keypoints[im1.name] = ktps1

            self._matcher.match(
                image0,
                image1,
                geometric_verification=GeometricVerification.NONE,
            )
            ktps0 = self._matcher._features0.keypoints
            ktps1 = self._matcher._features1.keypoints
            self.keypoints[im0.name] = ktps0
            self.keypoints[im1.name] = ktps1

            matches0 = self._matcher._matches0
            matches01 = self._matcher._matches01
            matches_dict = {
                "matches0": matches0,
                "matches01": matches01,
            }
            self.correspondences[(im0, im1)] = ReorganizeMatches(
                ktps0.shape[0], matches_dict
            )

            (
                self.keypoints[im0.name],
                self.keypoints[im1.name],
                self.correspondences[(im0, im1)],
            ) = ApplyGeometricVer(
                self.keypoints[im0.name],
                self.keypoints[im1.name],
                self.correspondences[(im0, im1)],
            )

        return self.keypoints, self.correspondences
