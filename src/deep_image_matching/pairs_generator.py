from typing import List, Union
from pathlib import Path


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
        matching_strategy: str,
        retrieval_option: Union[str, None],
        overlap: int,
    ) -> None:
        self.img_paths = img_paths
        self.matching_strategy = matching_strategy
        self.retrieval_option = retrieval_option
        self.overlap = overlap

    def bruteforce(self):
        print("\nBruteforce matching, generating pairs ..")
        pairs = BruteForce(self.img_paths, self.overlap)
        print("N of pairs:", len(pairs))
        return pairs

    def sequential(self):
        print("\nSequential matching, generating pairs ..")
        pairs = SequentialPairs(self.img_paths, self.overlap)
        print("N of pairs:", len(pairs))
        return pairs

    def retrieval(self):
        print("Retrieval matching, generating pairs ..")
        raise NotImplementedError("Retrieval needs to be implemented. Exit")

    def run(self):
        generate_pairs = getattr(self, self.matching_strategy)
        pairs = generate_pairs()
        return pairs
