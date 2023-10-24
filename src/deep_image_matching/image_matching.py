import cv2
import numpy as np
from pathlib import Path
import logging
from copy import deepcopy

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

DEBUG = True


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
        # extractor =

        for idx, pair in enumerate(self.image_list):
            logger.info(f"Extracting features from image: {pair.name}")

        # if as_half:
        #     for k in pred:
        #         dt = pred[k].dtype
        #         if (dt == np.float32) and (dt != np.float16):
        #             pred[k] = pred[k].astype(np.float16)

        # with h5py.File(str(feature_path), "a", libver="latest") as fd:
        #     try:
        #         if name in fd:
        #             del fd[name]
        #         grp = fd.create_group(name)
        #         for k, v in pred.items():
        #             grp.create_dataset(k, data=v)
        #         if "keypoints" in pred:
        #             grp["keypoints"].attrs["uncertainty"] = uncertainty
        #     except OSError as error:
        #         if "No space left on device" in error.args[0]:
        #             logger.error(
        #                 "Out of disk space: storing features on disk can take "
        #                 "significant space, did you enable the as_half flag?"
        #             )
        #             del grp, fd[name]
        #         raise error

    def match_pairs(self):
        # first_img = cv2.imread(str(self.image_list[0].absolute_path))
        # w_size = first_img.shape[1]
        # self.custom_config["general"]["w_size"] = w_size

        # Do not use geometric verification within the matcher, but do it after
        matcher_cfg = deepcopy(self.custom_config)
        matcher_cfg["geometric_verification"] = GeometricVerification.NONE

        # Initialize matcher
        if self.local_features == "lightglue":
            self._matcher = LightGlueMatcher(**matcher_cfg)
        elif self.local_features == "superglue":
            self._matcher = SuperGlueMatcher(**matcher_cfg)
        elif self.local_features == "loftr":
            self._matcher = LOFTRMatcher(**matcher_cfg)
        elif self.local_features == "detect_and_describe":
            matcher_cfg["ALIKE"]["n_limit"] = self.max_feat_numb
            detector_and_descriptor = matcher_cfg["general"]["detector_and_descriptor"]
            local_feat_conf = matcher_cfg[detector_and_descriptor]
            local_feat_extractor = LocalFeatureExtractor(
                detector_and_descriptor,
                local_feat_conf,
                self.max_feat_numb,
            )
            self._matcher = DetectAndDescribe(**matcher_cfg)
            matcher_cfg["general"]["local_feat_extractor"] = local_feat_extractor
        else:
            raise ValueError(
                "Invalid local feature extractor. Supported extractors: lightglue, superglue, loftr, detect_and_describe"
            )

        for idx, pair in enumerate(self.pairs):
            logger.info(f"Matching image pair: {pair[0].name} - {pair[1].name}")
            im0 = pair[0]
            im1 = pair[1]
            res_pair_dir = Path("res") / f"{pair[0].stem}-{pair[1].stem}"

            if DEBUG:
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

                kpts0 = deepcopy(self._matcher._features0.keypoints)
                kpts1 = deepcopy(self._matcher._features1.keypoints)
                matches0 = deepcopy(self._matcher._matches0)

                # Make correspondence matrix
                correspondences = make_correspondence_matrix(matches0)

                # Apply geometric verification and store results
                self.keypoints[im0.name] = kpts0
                self.keypoints[im1.name] = kpts1
                self.correspondences[(im0, im1)] = apply_geometric_verification(
                    kpts0=self.keypoints[im0.name],
                    kpts1=self.keypoints[im1.name],
                    correspondences=correspondences,
                    config=self.custom_config["general"],
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

                # Not needed anymore as the correspondence matrix is computed
                # matches01 = self._matcher._matches01
                # matches_dict = {
                #     "matches0": matches0,
                #     "matches01": matches01,
                # }

                # Store keypoints and matches
                self.keypoints[im0.name] = ktps0
                self.keypoints[im1.name] = ktps1

                # Make correspondence matrix
                correspondences = make_correspondence_matrix(matches0)

                # Apply geometric verification
                self.correspondences[(im0, im1)] = apply_geometric_verification(
                    kpts0=self.keypoints[im0.name],
                    kpts1=self.keypoints[im1.name],
                    correspondences=correspondences,
                    config=self.custom_config["general"],
                )

        return self.keypoints, self.correspondences
