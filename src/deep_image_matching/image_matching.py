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
        # TODO: add default values for not necessary parameters
        self,
        imgs_dir: Path,
        matching_strategy: str,
        retrieval_option: str,
        local_features: str,
        custom_config: dict,
        max_feat_numb: int = 2048,
        pair_file: Path = None,
        overlap: int = 1,
    ):
        self.matching_strategy = matching_strategy
        self.retrieval_option = retrieval_option
        self.local_features = local_features
        self.custom_config = custom_config
        self.max_feat_numb = max_feat_numb
        self.pair_file = Path(self.pair_file) if pair_file is not None else None
        self.overlap = overlap
        self.keypoints = {}
        self.correspondences = {}

        if retrieval_option == "sequential":
            if overlap is None:
                raise ValueError(
                    "'overlap' option is required when 'strategy' is set to sequential"
                )
        elif retrieval_option == "custom_pairs":
            if self.pair_file is None:
                raise ValueError(
                    "'pair_file' option is required when 'strategy' is set to custom_pairs"
                )
            else:
                if not self.pair_file.exists():
                    raise ValueError(f"File {self.pair_file} does not exist")

        # Initialize ImageList class
        self.image_list = ImageList(imgs_dir)
        images = self.image_list.img_names

        if len(images) == 0:
            raise ValueError(
                "Image folder empty. Supported formats: '.jpg', '.JPG', '.png'"
            )
        elif len(images) == 1:
            raise ValueError("Image folder must contain at least two images")

        # Do not use geometric verification within the matcher, but do it after
        self.custom_config["general"][
            "geometric_verification"
        ] = GeometricVerification.NONE

        # Initialize matcher
        # first_img = cv2.imread(str(self.image_list[0].absolute_path))
        # w_size = first_img.shape[1]
        # self.custom_config["general"]["w_size"] = w_size

        if self.local_features == "lightglue":
            self._matcher = LightGlueMatcher(**self.custom_config)
        elif self.local_features == "superglue":
            self._matcher = SuperGlueMatcher(**self.custom_config)
        elif self.local_features == "loftr":
            self._matcher = LOFTRMatcher(**self.custom_config)
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
            self._matcher = DetectAndDescribe(**self.custom_config)
            self.custom_config["general"]["local_feat_extractor"] = local_feat_extractor
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

    @property
    def img_names(self):
        return self.image_list.img_names

    def generate_pairs(self):
        self.pairs = []
        if self.pair_file is not None and self.matching_strategy == "custom_pairs":
            assert self.pair_file.exists(), f"File {self.pair_file} does not exist"
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
        for idx, pair in enumerate(self.pairs):
            logger.info(f"Matching image pair: {pair[0].name} - {pair[1].name}")
            im0 = pair[0]
            im1 = pair[1]
            res_pair_dir = Path("res") / f"{pair[0].stem}-{pair[1].stem}"

            debug = False

            if debug:
                from copy import deepcopy
                import torch
                from deep_image_matching.hloc.extractors.superpoint import SuperPoint

                def to_tensor(image):
                    if len(image.shape) == 2:
                        image = image[None][None]
                    elif len(image.shape) == 3:
                        image = image.transpose(2, 0, 1)[None]
                    return torch.tensor(image / 255.0, dtype=torch.float).to(device)

                device = torch.device("cuda")
                cfg = {"name": "superpoint", "nms_radius": 3, "max_keypoints": 4096}

                image0 = cv2.imread(str(im0), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                image1 = cv2.imread(str(im1), cv2.IMREAD_GRAYSCALE).astype(np.float32)

                extractor = SuperPoint(cfg).eval().to(device)
                feats0 = extractor({"image": to_tensor(image0)})
                feats1 = extractor({"image": to_tensor(image1)})

                self._matcher.match(
                    image0,
                    image1,
                    feats0,
                    feats1,
                    general={"save_dir": res_pair_dir},
                )

                ktps0 = deepcopy(self._matcher._features0.keypoints)
                ktps1 = deepcopy(self._matcher._features1.keypoints)
                matches0 = deepcopy(self._matcher._matches0)
                matches01 = deepcopy(self._matcher._matches01)
                matches_dict = {
                    "matches0": matches0,
                    "matches01": matches01,
                }

                correspondences = ReorganizeMatches(ktps0.shape[0], matches_dict)

                (
                    self.keypoints[im0.name],
                    self.keypoints[im1.name],
                    self.correspondences[(im0, im1)],
                ) = ApplyGeometricVer(
                    ktps0,
                    ktps1,
                    correspondences,
                )

                # deepcopy status
                res = {}
                k = f"{pair[0].stem}_{pair[1].stem}"
                res[k] = (
                    deepcopy(self.keypoints),
                    deepcopy(self.correspondences),
                )

                logger.info(f"Pairs: {pair[0].name} - {pair[1].name} done.")

            else:
                image0 = cv2.imread(str(im0), cv2.COLOR_RGB2BGR)
                image1 = cv2.imread(str(im1), cv2.COLOR_RGB2BGR)

                if len(image0.shape) == 2:
                    image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
                if len(image1.shape) == 2:
                    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)

                self._matcher.match(
                    image0,
                    image1,
                    general={"save_dir": res_pair_dir},
                )

                ktps0 = self._matcher._features0.keypoints
                ktps1 = self._matcher._features1.keypoints
                matches0 = self._matcher._matches0
                matches01 = self._matcher._matches01
                matches_dict = {
                    "matches0": matches0,
                    "matches01": matches01,
                }

                # Store keypoints and matches
                self.keypoints[im0.name] = ktps0
                self.keypoints[im1.name] = ktps1
                self.correspondences[(im0, im1)] = ReorganizeMatches(
                    ktps0.shape[0], matches_dict
                )

                # Apply geometric verification
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
