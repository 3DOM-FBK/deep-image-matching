import sys
import cv2
import numpy as np

from pathlib import Path
from typing import List, Union, Tuple
from src.deep_image_matching.image_retrieval import ImageRetrieval
from . import logger

from .extractors.keynetaffnethardnet import KeyNet
from .matchers.kornia_matcher import KorniaMatcher
from .utils.geometric_verification import geometric_verification
from src.deep_image_matching import GeometricVerification

MIN_N_MATCHES = 100

KeyNetAffNetHardNetConfig = {
    "extractor": {
        "name": "keynetaffnethardnet",
        "n_features": 8000,
        "upright" : False,
    },
    "matcher": {"name": "kornia_matcher", "match_mode": "smnn", "th": 0.95},
}

def SequentialPairs(img_list: List[Union[str, Path]], overlap: int) -> List[tuple]:
    pairs = []
    for i in range(len(img_list) - overlap):
        for k in range(overlap):
            j = i + k + 1
            im1 = img_list[i]
            im2 = img_list[j]
            pairs.append((im1, im2))
    return pairs

def BruteForce(img_list: List[Union[str, Path]]) -> List[tuple]:
    pairs = []
    for i in range(len(img_list) - 1):
        for j in range(i + 1, len(img_list)):
            im1 = img_list[i]
            im2 = img_list[j]
            pairs.append((im1, im2))
    return pairs

def MatchingLowres(brute_pairs: List[Tuple[Union[str, Path]]]):
    pairs = []
    KNextractor = KeyNet(KeyNetAffNetHardNetConfig)
    KorniaMatch = KorniaMatcher(KeyNetAffNetHardNetConfig)
    for pair in brute_pairs:
        im0_path = pair[0]
        im1_path = pair[1]

        im0 = cv2.imread(str(im0_path))
        H, W = im0.shape[:2]
        new_width = 500
        new_height = int(H * 500/W)
        im0 = cv2.resize(im0, (new_width, new_height))
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)

        im1 = cv2.imread(str(im1_path))
        H, W = im1.shape[:2]
        new_width = 500
        new_height = int(H * 500/W)
        im0 = cv2.resize(im0, (new_width, new_height))
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

        features0 = KNextractor._extract(im0)
        features1 = KNextractor._extract(im1)
        matches01_idx = KorniaMatch._match_pairs(features0, features1)
        mkpts0 = features0["keypoints"][matches01_idx[:, 0]]
        mkpts1 = features1["keypoints"][matches01_idx[:, 1]]
        _, inlMask = geometric_verification(
            kpts0=mkpts0,
            kpts1=mkpts1,
            method=GeometricVerification.PYDEGENSAC,
            threshold=4,
            confidence=0.99,
    )
        count_true = np.count_nonzero(inlMask)
        print(im0_path.name, im1_path.name, count_true)
        if count_true > MIN_N_MATCHES:
            pairs.append(pair)
    return pairs

class PairsGenerator:
    def __init__(
        self,
        img_paths: List[Path],
        pair_file: Path,
        strategy: str,
        retrieval_option: Union[str, None] = None,
        overlap: int = 1,
        image_dir: str = "",
        output_dir: str = "",
    ) -> None:
        self.img_paths = img_paths
        self.pair_file = pair_file
        self.strategy = strategy
        self.retrieval_option = retrieval_option
        self.overlap = overlap
        self.image_dir = image_dir
        self.output_dir = output_dir

    def bruteforce(self):
<<<<<<< HEAD:src/deep_image_matching/utils/pairs_generator.py
        logger.debug("Bruteforce matching, generating pairs ..")
        pairs = BruteForce(self.img_paths, self.overlap)
        logger.info(f"  Number of pairs: {len(pairs)}")
=======
        logger.info("Bruteforce matching, generating pairs ..")
        pairs = BruteForce(self.img_paths)
        logger.info(f"Number of pairs: {len(pairs)}")
>>>>>>> 36d107e (Added preselection at low res):src/deep_image_matching/pairs_generator.py
        return pairs

    def sequential(self):
        logger.debug("Sequential matching, generating pairs ..")
        pairs = SequentialPairs(self.img_paths, self.overlap)
        logger.info(f"  Number of pairs: {len(pairs)}")
        return pairs

    def retrieval(self):
        import hloc

        logger.info("Retrieval matching, generating pairs ..")
        brute_pairs = BruteForce(self.img_paths)
        with open(self.output_dir / "retrieval_pairs.txt", "w") as txt_file:
            for pair in brute_pairs:
                txt_file.write(f"{pair[0]} {pair[1]}\n")
        pairs = ImageRetrieval(self.image_dir, self.output_dir, self.retrieval_option, self.output_dir / "retrieval_pairs.txt")
        return pairs
    
    def matching_lowres(self):
        logger.info("Low resolution matching, generating pairs ..")
        brute_pairs = BruteForce(self.img_paths)
        pairs = MatchingLowres(brute_pairs)
        return pairs

    def run(self):
        generate_pairs = getattr(self, self.strategy)
        pairs = generate_pairs()

        with open(self.pair_file, "w") as txt_file:
            for pair in pairs:
                txt_file.write(f"{pair[0].name} {pair[1].name}\n")

        return pairs
