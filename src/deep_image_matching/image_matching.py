# Aggiungere interfaccia con global descriptors
# Aggiungere check epipolare
# se un'immagine ripetutamente non si riesce ad agganciare, -> scartarla
# Estrarre tutte le features in una volta
# Un'immagine viene ruotata solo se per due volte passa il check
# Velocizzare il tutto
import gc
import os
import random
import shutil
from pathlib import Path
from pprint import pprint
from PIL import Image

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional

from . import (
    GeometricVerification,
    Quality,
    TileSelection,
    Timer,
    extractors,
    logger,
    matchers,
)
from .extractors.extractor_base import extractor_loader
from .extractors.superpoint import SuperPointExtractor
from .io.h5 import get_features
from .matchers.lightglue import LightGlueMatcher
from .matchers.matcher_base import matcher_loader
from .pairs_generator import PairsGenerator
from .utils.image import ImageList
from .utils.geometric_verification import geometric_verification


from multiprocessing import Pool
from multiprocessing import Pool, set_start_method
from functools import partial


def resize(path_to_img: str, new_width: int) -> np.ndarray:
    """
    Resize an image to a specified width while maintaining the aspect ratio.

    This function resizes the input image to a new width, and computes the corresponding height
    to maintain the original aspect ratio of the image. The resized image is then converted to
    grayscale using OpenCV's BGR to grayscale conversion.

    Parameters:
    - img (np.ndarray): Input image as a NumPy array with shape (H, W, C) where H is the height,
                       W is the width, and C is the number of channels (usually 3 for color images).
    - new_width (int): Desired width for the resized image.

    Returns:
    - np.ndarray: Resized and converted grayscale image as a NumPy array with shape (new_height,
                  new_width).
    """
    img = Image.open(path_to_img).convert("L")
    W, H = img.size
    new_height = int(H * new_width / W)
    resized_img = img.resize((new_width, new_height))

    return resized_img


def find_matches_per_rotation(
    path_to_img0: str,
    path_to_img1: str,
    rotations: List[int],
    cv2_rot_params: List[Optional[int]],
    SPextractor: SuperPointExtractor,
    LGmatcher: LightGlueMatcher,
    resize_size: int,
):
    features = {
        "feat0": None,
        "feat1": None,
    }

    image0 = np.array(resize(path_to_img0, resize_size))
    _image1 = resize(path_to_img1, resize_size)
    image1 = np.array(_image1)
    features["feat0"] = SPextractor._extract(image0)
    matchesXrotation = []
    for rotation, cv2rotation in zip(rotations, cv2_rot_params):
        # rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), rotation, 1.0)
        # rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        if rotation != 0:
            image1 = np.array(_image1.rotate(cv2rotation, expand=False))
        features["feat1"] = SPextractor._extract(image1)
        if (
            features["feat0"]["keypoints"].shape[0] > 8
            and features["feat1"]["keypoints"].shape[0] > 8
        ):
            matches = LGmatcher._match_pairs(features["feat0"], features["feat1"])

            F, inlMask = geometric_verification(
                features["feat0"]["keypoints"][matches[:, 0], :],
                features["feat1"]["keypoints"][matches[:, 1], :],
                GeometricVerification.PYDEGENSAC,
                threshold=1,
                confidence=0.9999,
                max_iters=10000,
                bool=False,
            )

            verified_matches = np.sum(inlMask)
            # print(matches.shape[0], verified_matches)

        else:
            verified_matches = 0

        # matchesXrotation.append((rotation, matches.shape[0]))
        matchesXrotation.append((rotation, verified_matches))
    return matchesXrotation


def upright(
    cluster0,
    path_to_upright_dir,
    rotations,
    cv2_rot_params,
    SPextractor,
    LGmatcher,
    pairs,
    resize_size,
    processed_pairs,
    cluster1,
):
    processed = []
    rotated_images = []

    for j, img1 in enumerate(cluster1):
        # for i, img0 in enumerate(cluster0):
        for i in range(len(cluster0)):
            img0 = cluster0[len(cluster0) - 1 - i]

            # if True: # bruteforce, global descriptors or other ways to reduce image pairs are not used
            # if (img0, img1) not in processed_pairs and (img1, img0) not in processed_pairs:
            if (
                (img0, img1) in pairs
                or (img1, img0) in pairs
                and (img0, img1) not in processed_pairs
                and (img1, img0) not in processed_pairs
            ):
                processed_pairs.append((img0, img1))
                # print('inside pairs')
                # print(j, i, len(cluster1), len(cluster0))
                matchesXrotation = find_matches_per_rotation(
                    str(path_to_upright_dir / img0),
                    str(path_to_upright_dir / img1),
                    rotations,
                    cv2_rot_params,
                    SPextractor,
                    LGmatcher,
                    resize_size,
                )
                index_of_max = max(
                    range(len(matchesXrotation)),
                    key=lambda i: matchesXrotation[i][1],
                )
                n_matches = matchesXrotation[index_of_max][1]
                if index_of_max != 0 and n_matches > 100:
                    print(f"ref {img0}     rotated {img1} {rotations[index_of_max]}")
                    rotated_images.append((img1, rotations[index_of_max]))
                    image1 = Image.open(str(path_to_upright_dir / img1)).convert("L")
                    # image1 = cv2.imread(str(path_to_upright_dir / img1))
                    # rotated_image1 = cv2.rotate(image1, cv2_rot_params[index_of_max])
                    # rotated_image1 = cv2.cvtColor(rotated_image1, cv2.COLOR_BGR2GRAY)
                    p = image1.rotate(cv2_rot_params[index_of_max], expand=False)
                    p.save(str(path_to_upright_dir / img1))
                    processed.append(img1)
                    break
                if index_of_max == 0 and n_matches > 100:
                    processed.append(img1)
                    image1 = Image.open(str(path_to_upright_dir / img1)).convert("L")
                    # image1 = cv2.imread(str(path_to_upright_dir / img1))
                    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    image1.save(str(path_to_upright_dir / img1))
                    # cv2.imwrite(str(path_to_upright_dir / img1), image1)
                    print(
                        f"ref {img0}     NOT rotated {img1} {rotations[index_of_max]}"
                    )
                    break
    return processed, rotated_images


def make_correspondence_matrix(matches: np.ndarray) -> np.ndarray:
    kpts_number = matches.shape[0]
    n_tie_points = np.arange(kpts_number).reshape((-1, 1))
    matrix = np.hstack((n_tie_points, matches.reshape((-1, 1))))
    correspondences = matrix[~np.any(matrix == -1, axis=1)]
    return correspondences


def get_pairs_from_file(pair_file: Path) -> list:
    pairs = []
    with open(pair_file, "r") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            im1, im2 = line.strip().split(" ", 1)
            pairs.append((im1, im2))
    return pairs


class ImageMatching:
    """
    ImageMatching class for performing image matching and feature extraction.

    Methods:
        __init__(self, imgs_dir, output_dir, matching_strategy, local_features, matching_method, retrieval_option=None, pair_file=None, overlap=None, existing_colmap_model=None, custom_config={})
            Initializes the ImageMatching class.
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
    # pair_file=pair_file,
    # retrieval_option=retrieval_option,
    # overlap=overlap,
    # existing_colmap_model=existing_colmap_model,

    def __init__(
        # TODO: add default values for not necessary parameters
        self,
        imgs_dir: Path,
        output_dir: Path,
        matching_strategy: str,
        local_features: str,
        matching_method: str,
        retrieval_option: str = None,
        pair_file: Path = None,
        overlap: int = None,
        existing_colmap_model: Path = None,
        custom_config: dict = {},
    ):
        """
        Initializes the ImageMatching class.

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
        self.image_dir = Path(imgs_dir)
        self.output_dir = Path(output_dir)
        self.matching_strategy = matching_strategy
        self.retrieval_option = retrieval_option
        self.local_features = local_features
        self.matching_method = matching_method
        self.pair_file = Path(pair_file) if pair_file else None
        self.overlap = overlap
        self.existing_colmap_model = existing_colmap_model

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
        elif retrieval_option == "covisibility":
            if self.existing_colmap_model is None:
                raise ValueError(
                    "'existing_colmap_model' option is required when 'strategy' is set to covisibility"
                )
            else:
                if not self.existing_colmap_model.exists():
                    raise ValueError(
                        f"File {self.existing_colmap_model} does not exist"
                    )

        # Initialize ImageList class
        self.image_list = ImageList(imgs_dir)
        images = self.image_list.img_names
        if len(images) == 0:
            raise ValueError(f"Image folder empty. Supported formats: {self.image_ext}")
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
                f"Invalid matcher. {self.matching_method} is not supported."
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
        logger.info(f"  Number of images: {len(self.image_list)}")
        logger.info(f"  Matching strategy: {self.matching_strategy}")
        logger.info(f"  Image quality: {self.custom_config['general']['quality'].name}")
        logger.info(
            f"  Tile selection: {self.custom_config['general']['tile_selection'].name}"
        )
        logger.info(f"  Feature extraction method: {self.local_features}")
        logger.info(f"  Matching method: {self.matching_method}")
        logger.info(
            f"  Geometric verification: {self.custom_config['general']['geom_verification'].name}"
        )
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")

    @property
    def img_names(self):
        return self.image_list.img_names

    def generate_pairs(self, **kwargs) -> Path:
        """
        Generates pairs of images for matching.

        Returns:
            Path: The path to the pair file containing the generated pairs of images.
        """
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
                self.pair_file,
                self.matching_strategy,
                self.retrieval_option,
                self.overlap,
                self.image_dir,
                self.output_dir,
                self.existing_colmap_model,
                **kwargs,
            )
            self.pairs = pairs_generator.run()

        return self.pair_file

    def rotate_upright_images(
        self, resize_size=500, n_cores=1, multi_processing=False
    ) -> None:
        """
        Try to rotate upright images. Useful for not rotation invariant approaches.
        Rotate images are saved in 'upright_images' dir in results folder

        Returns:
            None
        """
        gc.collect()
        logger.info("Rotating images upright...")
        pairs = [(item[0].name, item[1].name) for item in self.pairs]
        path_to_upright_dir = self.output_dir / "upright_images"
        os.makedirs(path_to_upright_dir, exist_ok=False)
        images = os.listdir(self.image_dir)

        logger.info(f"Copying images to {path_to_upright_dir}")
        for img in images:
            shutil.copy(self.image_dir / img, path_to_upright_dir / img)
        logger.info(f"{len(images)} images copied")

        rotations = [0, 90, 180, 270]
        # cv2_rot_params = [
        #    None,
        #    cv2.ROTATE_90_CLOCKWISE,
        #    cv2.ROTATE_180,
        #    cv2.ROTATE_90_COUNTERCLOCKWISE,
        # ]
        cv2_rot_params = [
            None,
            -90,
            180,
            90,
        ]
        self.rotated_images = []

        logger.info(f"Initializing Superpoint + LIghtGlue..")
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

        cluster0 = []
        cluster1 = os.listdir(path_to_upright_dir)

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

        # Random init
        random_first_img = random.randint(0, len(cluster1))
        # Choose first image
        # random_first_img = 1

        cluster0.append(cluster1[random_first_img])
        cluster1.pop(random_first_img)

        # Main loop
        processed_pairs = []
        max_iter = len(images)
        logger.info(f"Max n iter: {max_iter}")
        for iter in tqdm(range(max_iter)):
            rotated = []
            print(f"len(cluster0): {len(cluster0)}\t len(cluster1): {len(cluster1)}")
            last_cluster1_len = len(cluster1)

            # rotated.sort
            ##cluster0 = []
            # for r in reversed(rotated):
            #    cluster0.append(cluster1[r])
            # for r in reversed(rotated):
            #    cluster1.pop(r)

            if multi_processing:
                partial_upright = partial(
                    upright,
                    cluster0,
                    path_to_upright_dir,
                    rotations,
                    cv2_rot_params,
                    SPextractor,
                    LGmatcher,
                    pairs,
                    resize_size,
                )
                sublists = np.array_split(cluster1, n_cores)
                with Pool(n_cores) as p:
                    results = p.map(partial_upright, sublists)

                processed = [item[0] for item in results]
                rotated = [item[1] for item in results]

                processed = [item for sublist in processed for item in sublist if item]
                self.rotated_images = self.rotated_images + [
                    item[0] for item in rotated if item != []
                ]

            else:
                processed, rotated = upright(
                    cluster0,
                    path_to_upright_dir,
                    rotations,
                    cv2_rot_params,
                    SPextractor,
                    LGmatcher,
                    pairs,
                    resize_size,
                    processed_pairs,
                    cluster1,
                )

            for r in processed:
                cluster0.append(r)
            cluster1 = [name for name in cluster1 if name not in cluster0]

            if last_cluster1_len == len(cluster1) or len(cluster1) == 0:
                break

        out_file = self.pair_file.parent / f"{self.pair_file.stem}_rot.txt"
        with open(out_file, "w") as txt_file:
            for element in self.rotated_images:
                print(element)
                txt_file.write(f"{element[0]} {element[1]}\n")

        # Update image directory to the dir with upright images
        # Features will be rotate accordingly on exporting, if the images have been rotated
        self.image_dir = path_to_upright_dir
        self.image_list = ImageList(path_to_upright_dir)
        images = self.image_list.img_names

        torch.cuda.empty_cache()
        logger.info(f"Images rotated and saved in {path_to_upright_dir}")
        gc.collect()

    def extract_features(self) -> Path:
        """
        Extracts features from the images using the specified local feature extraction method.

        Returns:
            Path: The path to the directory containing the extracted features.

        Raises:
            ValueError: If the local feature extraction method is invalid or not supported.

        """
        logger.info(f"Extracting features with {self.local_features}...")
        logger.info(f"{self.local_features} configuration: ")
        pprint(self.custom_config["extractor"])

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
        timer = Timer(log_level="debug")

        logger.info(f"Matching features with {self.matching_method}...")
        logger.info(f"{self.matching_method} configuration: ")
        pprint(self.custom_config["matcher"])
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

        # TODO: Clean up features with no matches

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
