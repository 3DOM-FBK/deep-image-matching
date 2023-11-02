import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import extractors, matchers
from .extractors.extractor_base import extractor_loader
from .io.h5 import get_features
from .matchers.matcher_base import matcher_loader
from .utils.geometric_verification import geometric_verification
from .utils.image import ImageList
from .utils.pairs_generator import PairsGenerator

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
        # if self.local_features == "superpoint":
        #     extractor = SuperPointExtractor(**self.custom_config)
        # elif self.local_features == "disk":
        #     extractor = DiskExtractor(**self.custom_config)
        # else:

        # Dynamically load the extractor
        try:
            Extractor = extractor_loader(extractors, self.local_features)
        except AttributeError:
            raise ValueError(
                f"Invalid local feature extractor. {self.local_features} is not supported."
            )

        # Initialize extractor
        extractor = Extractor(**self.custom_config)

        # Extract features
        logger.info("Extracting features...")
        for img in tqdm(self.image_list):
            feature_path = extractor.extract(img)

        logger.info("Features extracted")

        return feature_path

    def match_pairs(self, feature_path: Path):
        # Check that feature_path exists
        if not Path(feature_path).exists():
            raise ValueError(f"Feature path {feature_path} does not exist")
        else:
            feature_path = Path(feature_path)

        # if self.matching_method == "lightglue":
        #     matcher = LightGlueMatcher(
        #         local_features=self.local_features, **matcher_cfg
        #     )
        # elif self.matching_method == "superglue":
        #     if self.local_features != "superpoint":
        #         raise ValueError(
        #             "Invalid local features for SuperGlue matcher. SuperGlue supports only SuperPoint features."
        #         )
        #     matcher = SuperGlueMatcher(**matcher_cfg)
        # elif self.matching_method == "loftr":
        #     raise NotImplementedError("LOFTR is not implemented yet")
        # matcher = LOFTRMatcher(**matcher_cfg)
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
        # else:
        #     raise ValueError(
        #         "Invalid local feature extractor. Supported extractors: lightglue, superglue, loftr, detect_and_describe"
        #     )

        # Dynamically load the matcher
        try:
            Matcher = matcher_loader(matchers, self.matching_method)
        except AttributeError:
            raise ValueError(
                f"Invalid matcher. {self.local_features} is not supported."
            )

        # Initialize matcher
        matcher = Matcher(**self.custom_config)

        # Match pairs
        logger.info("Matching features...")
        for pair in tqdm(self.pairs):
            logger.debug(f"Matching image pair: {pair[0].name} - {pair[1].name}")
            im0 = pair[0]
            im1 = pair[1]

            # Run matching
            correspondences = matcher.match(
                feature_path=feature_path,
                img0=im0,
                img1=im1,
            )

            # Get original keypoints from h5 file
            kpts0 = get_features(feature_path, im0.name)["keypoints"]
            kpts1 = get_features(feature_path, im1.name)["keypoints"]

            # Check if there are enough correspondences
            min_matches_per_pair = 50
            if len(correspondences) < min_matches_per_pair:
                logger.warning(
                    f"Not enough correspondences found between {im0.name} and {im1.name} ({len(correspondences)}). Skipping image pair"
                )
                continue

            # Apply geometric verification
            correspondences = apply_geometric_verification(
                kpts0=kpts0,
                kpts1=kpts1,
                correspondences=correspondences,
                config=self.custom_config["general"],
            )

            # Save keypoints and correspondences
            self.keypoints[im0.name] = kpts0
            self.keypoints[im1.name] = kpts1
            self.correspondences[(im0, im1)] = correspondences

            logger.debug(f"Pairs: {pair[0].name} - {pair[1].name} done.")

        logger.info("Matching done!")

        return self.keypoints, self.correspondences
