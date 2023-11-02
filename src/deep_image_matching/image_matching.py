import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .extractors import DiskExtractor, SuperPointExtractor
from .geometric_verification import geometric_verification
from .image import ImageList
from .io.h5 import get_features
from .matchers import LightGlueMatcher, LOFTRMatcher, SuperGlueMatcher
from .pairs_generator import PairsGenerator

logger = logging.getLogger(__name__)


def make_correspondence_matrix(matches: np.ndarray) -> np.ndarray:
    kpts_number = matches.shape[0]
    n_tie_points = np.arange(kpts_number).reshape((-1, 1))
    matrix = np.hstack((n_tie_points, matches.reshape((-1, 1))))
    correspondences = matrix[~np.any(matrix == -1, axis=1)]
    return correspondences


def apply_geometric_verification(
    kpts0: np.ndarray, kpts1: np.ndarray, correspondences: np.ndarray, config: dict
) -> np.ndarray:
    mkpts0 = kpts0[correspondences[:, 0]]
    mkpts1 = kpts1[correspondences[:, 1]]
    _, inlMask = geometric_verification(
        kpts0=mkpts0,
        kpts1=mkpts1,
        method=config["geometric_verification"],
        threshold=config["gv_threshold"],
        confidence=config["gv_confidence"],
    )
    return correspondences[inlMask]


class ImageMatching:
    def __init__(
        # TODO: add default values for not necessary parameters
        self,
        imgs_dir: Path,
        matching_strategy: str,
        retrieval_option: str,
        local_features: str,
        matching_method: str,
        custom_config: dict,
        # max_feat_numb: int = 2048,
        pair_file: Path = None,
        overlap: int = 1,
    ):
        self.matching_strategy = matching_strategy
        self.retrieval_option = retrieval_option
        self.local_features = local_features
        self.matching_method = matching_method
        self.custom_config = custom_config
        # self.max_feat_numb = max_feat_numb
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

    def extract_features(self):
        if self.local_features == "superpoint":
            extractor = SuperPointExtractor(**self.custom_config)
        elif self.local_features == "disk":
            extractor = DiskExtractor(**self.custom_config)
        else:
            raise ValueError(
                "Invalid local feature extractor. Supported extractors: superpoint"
            )

        logger.info("Extracting features...")
        for img in tqdm(self.image_list):
            feature_path = extractor.extract(img)

        logger.info("Features extracted")

        return feature_path

    def match_pairs(self, feature_path: Path):
        # first_img = cv2.imread(str(self.image_list[0].absolute_path))
        # w_size = first_img.shape[1]
        # self.custom_config["general"]["w_size"] = w_size

        # Check that feature_path exists
        if not Path(feature_path).exists():
            raise ValueError(f"Feature path {feature_path} does not exist")
        else:
            feature_path = Path(feature_path)

        # Do not use geometric verification within the matcher, but do it after
        # matcher_cfg = deepcopy(self.custom_config)
        # matcher_cfg["geometric_verification"] = GeometricVerification.NONE
        matcher_cfg = self.custom_config

        # Initialize matcher
        if self.matching_method == "lightglue":
            matcher = LightGlueMatcher(
                local_features=self.local_features, **matcher_cfg
            )
        elif self.matching_method == "superglue":
            if self.local_features != "superpoint":
                raise ValueError(
                    "Invalid local features for SuperGlue matcher. SuperGlue supports only SuperPoint features."
                )
            matcher = SuperGlueMatcher(**matcher_cfg)
        elif self.matching_method == "loftr":
            matcher = LOFTRMatcher(**matcher_cfg)
        # elif self.local_features == "detect_and_describe":
        #     matcher_cfg["ALIKE"]["n_limit"] = self.max_feat_numb
        #     detector_and_descriptor = matcher_cfg["general"]["detector_and_descriptor"]
        #     local_feat_conf = matcher_cfg[detector_and_descriptor]
        #     local_feat_extractor = LocalFeatureExtractor(
        #         detector_and_descriptor,
        #         local_feat_conf,
        #         self.max_feat_numb,
        #     )
        #     matcher = DetectAndDescribe(**matcher_cfg)
        #     matcher_cfg["general"]["local_feat_extractor"] = local_feat_extractor
        else:
            raise ValueError(
                "Invalid local feature extractor. Supported extractors: lightglue, superglue, loftr, detect_and_describe"
            )

        logger.info("Matching features...")
        for pair in tqdm(self.pairs):
            logger.debug(f"Matching image pair: {pair[0].name} - {pair[1].name}")
            im0 = pair[0]
            im1 = pair[1]

            correspondences = matcher.match(
                feature_path=feature_path,
                img0=im0,
                img1=im1,
            )

            # Make correspondence matrix (no need it anymore as the matcher already does it)
            # correspondences = make_correspondence_matrix(matches)

            kpts0_h5 = get_features(feature_path, im0.name)["keypoints"]
            kpts1_h5 = get_features(feature_path, im1.name)["keypoints"]

            # Apply geometric verification and store results
            self.keypoints[im0.name] = kpts0_h5
            self.keypoints[im1.name] = kpts1_h5
            self.correspondences[(im0, im1)] = apply_geometric_verification(
                kpts0=self.keypoints[im0.name],
                kpts1=self.keypoints[im1.name],
                correspondences=correspondences,
                config=self.custom_config["general"],
            )

            logger.debug(f"Pairs: {pair[0].name} - {pair[1].name} done.")

        logger.info("Matching done!")

        return self.keypoints, self.correspondences
