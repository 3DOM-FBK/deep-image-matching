from pathlib import Path
from typing import List, Union

from .. import logger


def SequentialPairs(img_list: List[Union[str, Path]], overlap: int) -> List[tuple]:
    pairs = []
    for i in range(len(img_list) - overlap):
        for k in range(overlap):
            j = i + k + 1
            im1 = img_list[i]
            im2 = img_list[j]
            pairs.append((im1, im2))
    return pairs


def BruteForce(img_list: List[Union[str, Path]], overlap: int) -> List[tuple]:
    pairs = []
    for i in range(len(img_list) - 1):
        for j in range(i + 1, len(img_list)):
            im1 = img_list[i]
            im2 = img_list[j]
            pairs.append((im1, im2))
    return pairs


class PairsGenerator:
    def __init__(
        self,
        img_paths: List[Path],
        pair_file: Path,
        strategy: str,
        retrieval_option: Union[str, None] = None,
        overlap: int = 1,
    ) -> None:
        self.img_paths = img_paths
        self.pair_file = pair_file
        self.strategy = strategy
        self.retrieval_option = retrieval_option
        self.overlap = overlap

    def bruteforce(self):
        logger.debug("Bruteforce matching, generating pairs ..")
        pairs = BruteForce(self.img_paths, self.overlap)
        logger.info(f"  Number of pairs: {len(pairs)}")
        return pairs

    def sequential(self):
        logger.debug("Sequential matching, generating pairs ..")
        pairs = SequentialPairs(self.img_paths, self.overlap)
        logger.info(f"  Number of pairs: {len(pairs)}")
        return pairs

    def retrieval(self):
        logger.debug("Retrieval matching, generating pairs ..")
        raise NotImplementedError("Retrieval needs to be implemented. Exit")

    def run(self):
        generate_pairs = getattr(self, self.strategy)
        pairs = generate_pairs()

        with open(self.pair_file, "w") as txt_file:
            for pair in pairs:
                txt_file.write(f"{pair[0].name} {pair[1].name}\n")

        return pairs
