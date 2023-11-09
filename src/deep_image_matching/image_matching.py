from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from . import extractors, logger, matchers
from .extractors.extractor_base import extractor_loader
from .io.h5 import get_features, get_matches
from .matchers.matcher_base import matcher_loader
from .utils.consts import GeometricVerification, Quality, TileSelection
from .utils.geometric_verification import geometric_verification
from .utils.image import ImageList
from .utils.pairs_generator import PairsGenerator


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
        method=config["geom_verification"],
        threshold=config["gv_threshold"],
        confidence=config["gv_confidence"],
    )
    return correspondences[inlMask]


def get_pairs_from_file(pair_file: Path) -> list:
    pairs = []
    with open(pair_file, "r") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            im1, im2 = line.strip().split(" ", 1)
            pairs.append((im1, im2))
    return pairs


class ImageMatching:
    default_conf_general = {
        "quality": Quality.MEDIUM,
        "tile_selection": TileSelection.NONE,
        "geom_verification": GeometricVerification.PYDEGENSAC,
        "output_dir": "output",
        "tiling_grid": [1, 1],
        "tiling_overlap": 0,
        "force_cpu": False,
        "do_viz": False,
        "fast_viz": True,
        "hide_matching_track": True,
        "do_viz_tiles": False,
    }

    def __init__(
        # TODO: add default values for not necessary parameters
        self,
        imgs_dir: Path,
        output_dir: Path,
        matching_strategy: str,
        retrieval_option: str,
        local_features: str,
        matching_method: str,
        pair_file: Path = None,
        custom_config: dict = {},
        overlap: int = 1,
    ):
        self.image_dir = Path(imgs_dir)
        self.output_dir = Path(output_dir)
        self.matching_strategy = matching_strategy
        self.retrieval_option = retrieval_option
        self.local_features = local_features
        self.matching_method = matching_method
        self.pair_file = Path(pair_file) if pair_file is not None else None
        self.overlap = overlap

        # Merge default and custom config
        self.custom_config = custom_config
        self.custom_config["general"] = {
            **self.default_conf_general,
            **custom_config["general"],
        }

        # Check that parameters are valid
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

        # Initialize output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractor
        try:
            Extractor = extractor_loader(extractors, self.local_features)
        except AttributeError:
            raise ValueError(
                f"Invalid local feature extractor. {self.local_features} is not supported."
            )
        self._extractor = Extractor(self.custom_config)

        # Initialize matcher
        try:
            Matcher = matcher_loader(matchers, self.matching_method)
        except AttributeError:
            raise ValueError(
                f"Invalid matcher. {self.local_features} is not supported."
            )
        if self.matching_method == "lightglue":
            self._matcher = Matcher(
                local_features=self.local_features, config=self.custom_config
            )
        else:
            self._matcher = Matcher(self.custom_config)

        # Print configuration
        logger.info("Running image matching with the following configuration:")
        logger.info(f"  Image folder: {self.image_dir}")
        logger.info(f"  Output folder: {self.output_dir}")
        logger.info(f"  Matching strategy: {self.matching_strategy}")
        logger.info(f"  Retrieval option: {self.retrieval_option}")
        logger.info(f"  Feature extraction method: {self.local_features}")
        logger.info(f"  Matching method: {self.matching_method}")
        logger.info(f"  Overlap: {self.overlap}")
        logger.info(f"  Number of images: {len(self.image_list)}")

    @property
    def img_names(self):
        return self.image_list.img_names

    def generate_pairs(self):
        if self.pair_file is not None and self.matching_strategy == "custom_pairs":
            if not self.pair_file.exists():
                raise FileExistsError(f"File {self.pair_file} does not exist")

            pairs = get_pairs_from_file(self.pair_file)
            self.pairs = [
                (self.image_dir / im1, self.image_dir / im2) for im1, im2 in pairs
            ]

        else:
            pairs_generator = PairsGenerator(
                self.image_list.img_paths,
                self.matching_strategy,
                self.retrieval_option,
                self.overlap,
            )
            self.pairs = pairs_generator.run()
            with open(self.pair_file, "w") as txt_file:
                for pair in self.pairs:
                    txt_file.write(f"{pair[0].name} {pair[1].name}\n")

        return self.pair_file

    def extract_features(self) -> Path:
        # Extract features
        logger.info("Extracting features...")
        for img in tqdm(self.image_list):
            feature_path = self._extractor.extract(img)

        logger.info("Features extracted")

        return feature_path

    def match_pairs(self, feature_path: Path) -> Path:
        # Check that feature_path exists
        feature_path = Path(feature_path)
        if not feature_path.exists():
            raise ValueError(f"Feature path {feature_path} does not exist")

        # Define matches path
        matches_path = feature_path.parent / "matches.h5"

        # Match pairs
        logger.info("Matching features...")
        logger.info("")
        for pair in tqdm(self.pairs):
            logger.debug(f"Matching image pair: {pair[0].name} - {pair[1].name}")
            im0 = pair[0]
            im1 = pair[1]

            # Run matching
            matches = self._matcher.match(
                feature_path=feature_path,
                matches_path=matches_path,
                img0=im0,
                img1=im1,
            )

            if matches is None:
                continue

            # Get original keypoints from h5 file
            kpts0 = get_features(feature_path, im0.name)["keypoints"]
            kpts1 = get_features(feature_path, im1.name)["keypoints"]
            correspondences = get_matches(matches_path, im0.name, im1.name)

            # Apply geometric verification
            correspondences_cleaned = apply_geometric_verification(
                kpts0=kpts0,
                kpts1=kpts1,
                correspondences=correspondences,
                config=self.custom_config["general"],
            )

            # Update matches in h5 file
            with h5py.File(str(matches_path), "a", libver="latest") as fd:
                group = fd.require_group(im0.name)
                if im1.name in group:
                    del group[im1.name]
                group.create_dataset(im1.name, data=correspondences_cleaned)

            logger.debug(f"Pairs: {pair[0].name} - {pair[1].name} done.")

        logger.info("Matching done!")

        return matches_path
