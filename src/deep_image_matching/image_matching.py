import logging
import os
import shutil
from pathlib import Path
from pprint import pprint

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import (
    extractors,
    matchers,
)
from .config import Config
from .constants import GeometricVerification, Quality, TileSelection, Timer
from .extractors import SuperPointExtractor, extractor_loader
from .io import get_features
from .matchers import LightGlueMatcher, matcher_loader
from .pairs_generator import PairsGenerator
from .utils import ImageList, get_pairs_from_file

logger = logging.getLogger("dim")
timer = Timer(logger=logger)


def make_correspondence_matrix(matches: np.ndarray) -> np.ndarray:
    kpts_number = matches.shape[0]
    n_tie_points = np.arange(kpts_number).reshape((-1, 1))
    matrix = np.hstack((n_tie_points, matches.reshape((-1, 1))))
    correspondences = matrix[~np.any(matrix == -1, axis=1)]
    return correspondences


class ImageMatcher:
    """
    ImageMatcher class for performing image matching and feature extraction.

    Methods:
        __init__(self, imgs_dir, output_dir, matching_strategy, local_features, matching_method, retrieval_option=None, pair_file=None, overlap=None, existing_colmap_model=None, custom_config={})
            Initializes the ImageMatcher class.
        generate_pairs(self, **kwargs) -> Path:
            Generates pairs of images for matching.
        rotate_upright_images(self)
            Rotates upright images.
        extract_features(self) -> Path:
            Extracts features from the images.
        match_pairs(self, feature_path, try_full_image=False) -> Path:
            Matches pairs of images.
        rotate_back_features(self, feature_path)
            Rotates back the features.

    """

    default_conf_general = {
        "quality": Quality.MEDIUM,
        "tile_selection": TileSelection.NONE,
        "geom_verification": GeometricVerification.PYDEGENSAC,
        "output_dir": "output",
        "tile_size": [2048, 1365],
        "tile_overlap": 0,
        "force_cpu": False,
        "do_viz": False,
        "fast_viz": True,
        "hide_matching_track": True,
        "do_viz_tiles": False,
    }

    def __init__(
        self,
        config: Config,
        # imgs_dir: Path,
        # output_dir: Path,
        # matching_strategy: str,
        # local_features: str,
        # matching_method: str,
        # retrieval_option: str = None,
        # pair_file: Path = None,
        # overlap: int = None,
        # existing_colmap_model: Path = None,
        # custom_config: dict = {},
    ):
        """
        Initializes the ImageMatcher class.

        Parameters:
            imgs_dir (Path): Path to the directory containing the images.
            output_dir (Path): Path to the output directory for the results.
            matching_strategy (str): The strategy for generating pairs of images for matching.
            local_features (str): The method for extracting local features from the images.
            matching_method (str): The method for matching pairs of images.
            retrieval_option (str, optional): The retrieval option for generating pairs of images. Defaults to None.
            pair_file (Path, optional): Path to the file containing custom pairs of images. Required when 'retrieval_option' is set to 'custom_pairs'. Defaults to None.
            overlap (int, optional): The overlap between tiles. Required when 'retrieval_option' is set to 'sequential'. Defaults to None.
            existing_colmap_model (Path, optional): Path to the existing COLMAP model. Required when 'retrieval_option' is set to 'covisibility'. Defaults to None.
            custom_config (dict, optional): Custom configuration settings. Defaults to {}.

        Raises:
            ValueError: If the 'overlap' option is required but not provided when 'retrieval_option' is set to 'sequential'.
            ValueError: If the 'pair_file' option is required but not provided when 'retrieval_option' is set to 'custom_pairs'.
            ValueError: If the 'pair_file' does not exist when 'retrieval_option' is set to 'custom_pairs'.
            ValueError: If the 'existing_colmap_model' option is required but not provided when 'retrieval_option' is set to 'covisibility'.
            ValueError: If the 'existing_colmap_model' does not exist when 'retrieval_option' is set to 'covisibility'.
            ValueError: If the image folder is empty or contains only one image.

        Returns:
            None
        """

        # Store configuration
        self.config = config
        self.image_dir = Path(config.general["image_dir"])
        self.output_dir = Path(config.general["output_dir"])
        self.strategy = config.general["matching_strategy"]
        self.extraction = config.extractor["name"]
        self.matching = config.matcher["name"]
        self.pair_file = config.general["pair_file"]

        # self.existing_colmap_model = config.general["db_path"]
        # if config.general["retrieval"] == "covisibility":
        #     if self.existing_colmap_model is None:
        #         raise ValueError("'existing_colmap_model' option is required when 'strategy' is set to covisibility")
        #     else:
        #         if not self.existing_colmap_model.exists():
        #             raise ValueError(f"File {self.existing_colmap_model} does not exist")

        # Initialize ImageList class
        self.image_list = ImageList(self.image_dir)
        images = self.image_list.img_names
        if len(images) == 0:
            raise ValueError(f"Image folder empty. Supported formats: {self.image_ext}")
        elif len(images) == 1:
            raise ValueError("Image folder must contain at least two images")

        # Initialize output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractor
        try:
            Extractor = extractor_loader(extractors, self.extraction)
        except AttributeError:
            raise ValueError(f"Invalid local feature extractor. {self.extraction} is not supported.")
        self._extractor = Extractor(self.config)

        # Initialize matcher
        try:
            Matcher = matcher_loader(matchers, self.matching)
        except AttributeError:
            raise ValueError(f"Invalid matcher. {self.matching} is not supported.")
        if self.matching == "lightglue":
            self._matcher = Matcher(local_features=self.extraction, config=self.config)
        else:
            self._matcher = Matcher(self.config)

        # Print configuration
        logger.info("Running image matching with the following configuration:")
        logger.info(f"  Image folder: {self.image_dir}")
        logger.info(f"  Output folder: {self.output_dir}")
        logger.info(f"  Number of images: {len(self.image_list)}")
        logger.info(f"  Matching strategy: {self.strategy}")
        logger.info(f"  Image quality: {self.config.general['quality'].name}")
        logger.info(f"  Tile selection: {self.config.general['tile_selection'].name}")
        logger.info(f"  Feature extraction method: {self.extraction}")
        logger.info(f"  Matching method: {self.matching}")
        logger.info(f"  Geometric verification: {self.config.general['geom_verification'].name}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")

    @property
    def img_names(self):
        return self.image_list.img_names

    def run(self):
        """
        Runs the image matching pipeline.

        Returns:
            None

        Raises:
            None
        """
        # Generate pairs to be matched
        pair_path = self.generate_pairs()
        timer.update("generate_pairs")

        # Try to rotate images so they will be all "upright", useful for deep-learning approaches that usually are not rotation invariant
        if self.config.general["upright"]:
            self.rotate_upright_images()
            timer.update("rotate_upright_images")

        # Extract features
        feature_path = self.extract_features()
        timer.update("extract_features")

        # Matching
        match_path = self.match_pairs(feature_path)

        # If features have been extracted on "upright" images, this function bring features back to their original image orientation
        if self.config.general["upright"]:
            self.rotate_back_features(feature_path)
            timer.update("rotate_back_features")

        # Print timing
        timer.print("Deep Image Matching")

        return feature_path, match_path

    def generate_pairs(self, **kwargs) -> Path:
        """
        Generates pairs of images for matching.

        Returns:
            Path: The path to the pair file containing the generated pairs of images.
        """
        if self.pair_file is not None and self.strategy == "custom_pairs":
            if not self.pair_file.exists():
                raise FileExistsError(f"File {self.pair_file} does not exist")

            pairs = get_pairs_from_file(self.pair_file)
            self.pairs = [(self.image_dir / im1, self.image_dir / im2) for im1, im2 in pairs]

        else:
            pairs_generator = PairsGenerator(
                self.image_list.img_paths,
                self.pair_file,
                self.strategy,
                self.config.general["retrieval"],
                self.config.general["overlap"],
                self.image_dir,
                self.output_dir,
                **kwargs,
            )
            self.pairs = pairs_generator.run()

        return self.pair_file

    def rotate_upright_images(self):
        """
        Rotates the images in the image directory to an upright position.

        This method rotates the images in the image directory to an upright position using the OpenCV library. The rotated images are saved in a separate directory called "upright_images" within the output directory.

        Returns:
            None

        Raises:
            None
        """
        logger.info("Rotating images upright...")
        path_to_upright_dir = self.output_dir / "upright_images"
        os.makedirs(path_to_upright_dir, exist_ok=False)
        images = os.listdir(self.image_dir)
        processed_images = []

        for img in images:
            shutil.copy(self.image_dir / img, path_to_upright_dir / img)

        rotations = [0, 90, 180, 270]
        cv2_rot_params = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]
        self.rotated_images = []
        SPextractor = SuperPointExtractor(
            config={
                "general": {},
                "extractor": {
                    "keypoint_threshold": 0.005,
                    "max_keypoints": 1024,
                },
            }
        )
        LGmatcher = LightGlueMatcher(
            config={
                "general": {},
                "matcher": {
                    "depth_confidence": 0.95,  # early stopping, disable with -1
                    "width_confidence": 0.99,  # point pruning, disable with -1
                    "filter_threshold": 0.1,  # match threshold
                },
            },
        )
        features = {
            "feat0": None,
            "feat1": None,
        }
        for pair in tqdm(self.pairs):
            matchesXrotation = []

            # Reference image
            ref_image = pair[0].name
            image0 = cv2.imread(str(path_to_upright_dir / ref_image))
            H, W = image0.shape[:2]
            new_width = 500
            new_height = int(H * 500 / W)
            image0 = cv2.resize(image0, (new_width, new_height))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            features["feat0"] = SPextractor._extract(image0)

            # Target image - find the best rotation
            target_img = pair[1].name
            if target_img not in processed_images:
                # processed_images.append(target_img)
                image1 = cv2.imread(str(path_to_upright_dir / target_img))
                for rotation, cv2rotation in zip(rotations, cv2_rot_params):
                    H, W = image1.shape[:2]
                    new_width = 500
                    new_height = int(H * 500 / W)
                    _image1 = cv2.resize(image1, (new_width, new_height))
                    _image1 = cv2.cvtColor(_image1, cv2.COLOR_BGR2GRAY)
                    # rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), rotation, 1.0)
                    # rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
                    if rotation != 0:
                        _image1 = cv2.rotate(_image1, cv2rotation)
                    features["feat1"] = SPextractor._extract(_image1)
                    matches = LGmatcher._match_pairs(features["feat0"], features["feat1"])
                    matchesXrotation.append((rotation, matches.shape[0]))
                index_of_max = max(range(len(matchesXrotation)), key=lambda i: matchesXrotation[i][1])
                n_matches = matchesXrotation[index_of_max][1]
                if index_of_max != 0 and n_matches > 100:
                    processed_images.append(target_img)
                    self.rotated_images.append((pair[1].name, rotations[index_of_max]))
                    rotated_image1 = cv2.rotate(image1, cv2_rot_params[index_of_max])
                    cv2.imwrite(str(path_to_upright_dir / target_img), rotated_image1)

        out_file = self.pair_file.parent / f"{self.pair_file.stem}_rot.txt"
        with open(out_file, "w") as txt_file:
            for element in self.rotated_images:
                txt_file.write(f"{element[0]} {element[1]}\n")

        # Update image directory to the dir with upright images
        # Features will be rotate accordingly on exporting, if the images have been rotated
        self.image_dir = path_to_upright_dir
        self.image_list = ImageList(path_to_upright_dir)
        images = self.image_list.img_names

        torch.cuda.empty_cache()
        logger.info(f"Images rotated and saved in {path_to_upright_dir}")

    def extract_features(self) -> Path:
        """
        Extracts features from the images using the specified local feature extraction method.

        Returns:
            Path: The path to the directory containing the extracted features.

        Raises:
            ValueError: If the local feature extraction method is invalid or not supported.

        """
        logger.info(f"Extracting features with {self.extraction}...")
        logger.info(f"{self.extraction} configuration: ")
        pprint(self.config.extractor)

        # Extract features
        for img in tqdm(self.image_list):
            feature_path = self._extractor.extract(img)

        torch.cuda.empty_cache()
        logger.info("Features extracted!")

        return feature_path

    def match_pairs(self, feature_path: Path, try_full_image: bool = False) -> Path:
        """
        Matches features using a specified matching method.

        Args:
            feature_path (Path): The path to the directory containing the extracted features.
            try_full_image (bool, optional): Whether to try matching the full image. Defaults to False.

        Returns:
            Path: The path to the directory containing the matches.

        Raises:
            ValueError: If the feature path does not exist.
        """

        logger.info(f"Matching features with {self.matching}...")
        logger.info(f"{self.matching} configuration: ")
        pprint(self.config.matcher)

        # Check that feature_path exists
        feature_path = Path(feature_path)
        if not feature_path.exists():
            raise ValueError(f"Feature path {feature_path} does not exist")

        # Define matches path
        matches_path = feature_path.parent / "matches.h5"

        # Match pairs
        logger.info("Matching features...")
        logger.info("")
        for i, pair in enumerate(tqdm(self.pairs)):
            name0 = pair[0].name if isinstance(pair[0], Path) else pair[0]
            name1 = pair[1].name if isinstance(pair[1], Path) else pair[1]
            im0 = self.image_dir / name0
            im1 = self.image_dir / name1

            logger.debug(f"Matching image pair: {name0} - {name1}")

            # Run matching
            self._matcher.match(
                feature_path=feature_path,
                matches_path=matches_path,
                img0=im0,
                img1=im1,
                try_full_image=try_full_image,
            )
            timer.update("Match pair")

            # NOTE: Geometric verif. has been moved to the end of the matching process

        torch.cuda.empty_cache()
        timer.print("matching")

        return matches_path

    def rotate_back_features(self, feature_path: Path) -> None:
        """
        Rotates back the features.

        This method rotates back the features extracted from the images that were previously rotated upright using the 'rotate_upright_images' method. The rotation is performed based on the theta value associated with each image in the 'rotated_images' list. The rotated features are then saved back to the feature file.

        Parameters:
            feature_path (Path): The path to the feature file containing the extracted features.

        Returns:
            None

        Raises:
            None
        """
        # images = self.image_list.img_names
        for img, theta in tqdm(self.rotated_images):
            features = get_features(feature_path, img)
            keypoints = features["keypoints"]
            rotated_keypoints = np.empty(keypoints.shape)
            im = cv2.imread(str(self.image_dir / img))
            H, W = im.shape[:2]

            if theta == 180:
                for r in range(keypoints.shape[0]):
                    x, y = keypoints[r, 0], keypoints[r, 1]
                    y_rot = H - y
                    x_rot = W - x
                    rotated_keypoints[r, 0], rotated_keypoints[r, 1] = x_rot, y_rot

            if theta == 90:
                for r in range(keypoints.shape[0]):
                    x, y = keypoints[r, 0], keypoints[r, 1]
                    y_rot = W - x
                    x_rot = y
                    rotated_keypoints[r, 0], rotated_keypoints[r, 1] = x_rot, y_rot

            if theta == 270:
                for r in range(keypoints.shape[0]):
                    x, y = keypoints[r, 0], keypoints[r, 1]
                    y_rot = x
                    x_rot = H - y
                    rotated_keypoints[r, 0], rotated_keypoints[r, 1] = x_rot, y_rot

            with h5py.File(feature_path, "r+", libver="latest") as fd:
                del fd[img]
                features["keypoints"] = rotated_keypoints
                grp = fd.create_group(img)
                for k, v in features.items():
                    if k == "im_path" or k == "feature_path":
                        grp.create_dataset(k, data=str(v))
                    if isinstance(v, np.ndarray):
                        grp.create_dataset(k, data=v)

        logger.info("Features rotated back.")
