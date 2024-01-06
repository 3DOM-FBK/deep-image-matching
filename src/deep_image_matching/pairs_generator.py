from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from tqdm import tqdm

from . import Timer, logger
from .image_retrieval import ImageRetrieval
from .io.colmap_read_write_model import read_model


def pairs_from_sequential(
    img_list: List[Union[str, Path]], overlap: int
) -> List[tuple]:
    pairs = []
    for i in range(len(img_list) - overlap):
        for k in range(overlap):
            j = i + k + 1
            im1 = img_list[i]
            im2 = img_list[j]
            pairs.append((im1, im2))
    return pairs


def pairs_from_bruteforce(img_list: List[Union[str, Path]]) -> List[tuple]:
    pairs = []
    for i in range(len(img_list) - 1):
        for j in range(i + 1, len(img_list)):
            im1 = img_list[i]
            im2 = img_list[j]
            pairs.append((im1, im2))
    return pairs


def pairs_from_lowres(
    brute_pairs: List[Tuple[Union[str, Path]]],
    resize_max: int = 500,
    min_matches: int = 50,
):
    def read_tensor_image(
        path: Path, resize_to: int = 500, device="cuda"
    ) -> Tuple[np.ndarray, float]:
        device = (
            torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        )
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = img.shape[:2][::-1]
        scale = resize_to / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        img = cv2.resize(img, size_new)
        img = K.image_to_tensor(img, False).float() / 255.0
        img = img.to(device)

        return img, scale

    def get_matching_keypoints(lafs1, lafs2, idxs):
        mkpts1 = KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
        mkpts2 = KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
        return mkpts1, mkpts2

    timer = Timer(log_level="debug")

    pairs = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    KNextractor = KF.KeyNetAffNetHardNet(
        num_features=4000,
        upright=False,
        device=device,
    )

    for pair in tqdm(brute_pairs):
        im0_path = pair[0]
        im1_path = pair[1]

        im0, _ = read_tensor_image(im0_path, resize_max)
        im1, _ = read_tensor_image(im1_path, resize_max)
        hw1 = torch.tensor(im1.shape[2:])
        hw2 = torch.tensor(im1.shape[2:])
        adalam_config = {"device": device}

        with torch.inference_mode():
            lafs1, resps1, descs1 = KNextractor(im0)
            lafs2, resps2, descs2 = KNextractor(im1)
            dists, idxs = KF.match_adalam(
                descs1.squeeze(0),
                descs2.squeeze(0),
                lafs1,
                lafs2,  # Adalam takes into account also geometric information
                config=adalam_config,
                hw1=hw1,
                hw2=hw2,  # Adalam also benefits from knowing image size
            )
            timer.update("match pair")

        mkpts0, mkpts1 = get_matching_keypoints(lafs1, lafs2, idxs)

        # _, inlMask = geometric_verification(
        #     kpts0=mkpts0,
        #     kpts1=mkpts1,
        #     method=GeometricVerification.PYDEGENSAC,
        #     threshold=4,
        #     confidence=0.99,
        # )
        # count_true = np.count_nonzero(inlMask)
        # timer.update("geometric verification")

        # # print(im0_path.name, im1_path.name, count_true)
        # if count_true > min_matches:
        #     pairs.append(pair)
        if len(mkpts0) > min_matches:
            pairs.append(pair)

    timer.print("low-res pair generation")

    return pairs


def pairs_from_covisibility(model, output, num_matched):
    logger.info("Reading the COLMAP model...")
    cameras, images, points3D = read_model(model)

    logger.info("Extracting image pairs from covisibility info...")
    pairs = []
    for image_id, image in tqdm(images.items()):
        matched = image.point3D_ids != -1
        points3D_covis = image.point3D_ids[matched]

        covis = defaultdict(int)
        for point_id in points3D_covis:
            for image_covis_id in points3D[point_id].image_ids:
                if image_covis_id != image_id:
                    covis[image_covis_id] += 1

        if len(covis) == 0:
            logger.info(f"Image {image_id} does not have any covisibility.")
            continue

        covis_ids = np.array(list(covis.keys()))
        covis_num = np.array([covis[i] for i in covis_ids])

        if len(covis_ids) <= num_matched:
            top_covis_ids = covis_ids[np.argsort(-covis_num)]
        else:
            # get covisible image ids with top k number of common matches
            ind_top = np.argpartition(covis_num, -num_matched)
            ind_top = ind_top[-num_matched:]  # unsorted top k
            ind_top = ind_top[np.argsort(-covis_num[ind_top])]
            top_covis_ids = [covis_ids[i] for i in ind_top]
            assert covis_num[ind_top[0]] == np.max(covis_num)

        for i in top_covis_ids:
            pair = (image.name, images[i].name)
            pairs.append(pair)

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


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
        logger.debug("Bruteforce matching, generating pairs ..")
        pairs = pairs_from_bruteforce(self.img_paths)
        logger.info(f"Number of pairs: {len(pairs)}")
        return pairs

    def sequential(self):
        logger.debug("Sequential matching, generating pairs ..")
        pairs = pairs_from_sequential(self.img_paths, self.overlap)
        logger.info(f"  Number of pairs: {len(pairs)}")
        return pairs

    def retrieval(self):
        logger.info("Retrieval matching, generating pairs ..")
        brute_pairs = pairs_from_bruteforce(self.img_paths)
        with open(self.output_dir / "retrieval_pairs.txt", "w") as txt_file:
            for pair in brute_pairs:
                txt_file.write(f"{pair[0]} {pair[1]}\n")
        pairs = ImageRetrieval(
            self.image_dir,
            self.output_dir,
            self.retrieval_option,
            self.output_dir / "retrieval_pairs.txt",
        )
        return pairs

    def matching_lowres(self):
        logger.info("Low resolution matching, generating pairs ..")
        brute_pairs = pairs_from_bruteforce(self.img_paths)
        pairs = pairs_from_lowres(brute_pairs)
        return pairs

    def run(self):
        generate_pairs = getattr(self, self.strategy)
        pairs = generate_pairs()

        with open(self.pair_file, "w") as txt_file:
            for pair in pairs:
                txt_file.write(f"{pair[0].name} {pair[1].name}\n")

        return pairs
