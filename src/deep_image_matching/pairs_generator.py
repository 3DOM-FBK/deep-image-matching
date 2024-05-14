from collections import defaultdict, namedtuple
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from tqdm import tqdm

from .constants import Timer, logger
from .image_retrieval import ImageRetrieval
from .io.colmap_read_write_model import read_model
from .thirdparty.hloc.extractors.superpoint import SuperPoint
from .thirdparty.LightGlue.lightglue import LightGlue
from .utils.geometric_verification import geometric_verification


def pairs_from_sequential(img_list: List[Union[str, Path]], overlap: int) -> List[tuple]:
    pairs = []
    for i in range(len(img_list) - overlap):
        for k in range(overlap):
            j = i + k + 1
            im1 = img_list[i]
            im2 = img_list[j]
            pairs.append((im1, im2))
    return pairs


def pairs_from_bruteforce(img_list: List[Union[str, Path]]) -> List[tuple]:
    return list(combinations(img_list, 2))


def pairs_from_lowres(
    img_list: List[Union[str, Path]],
    resize_max: int = 1000,
    min_matches: int = 20,
    max_keypoints: int = 1024,
    use_superpoint: bool = True,
    do_geometric_verification: bool = False,
) -> List[tuple]:
    def read_tensor_image(path: Path, resize_to: int = 500, device="cuda") -> Tuple[np.ndarray, float]:
        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
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

    def frame2tensor(image: np.ndarray, device: str = "cpu"):
        if len(image.shape) == 2:
            image = image[None][None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)[None]
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)

    def sp2lg(feats: dict) -> dict:
        feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
        if feats["descriptors"].shape[-1] != 256:
            feats["descriptors"] = feats["descriptors"].T
        feats = {k: v[None] for k, v in feats.items()}
        return feats

    def rbd2np(data: dict) -> dict:
        """Remove batch dimension from elements in data"""
        return {
            k: v[0].cpu().numpy() if isinstance(v, (torch.Tensor, np.ndarray, list)) else v for k, v in data.items()
        }

    use_superpoint = True

    timer = Timer(log_level="debug")

    brute_pairs = pairs_from_bruteforce(img_list)

    pairs = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if use_superpoint:
        extractor = (
            SuperPoint(
                {
                    "nms_radius": 3,
                    "max_keypoints": 2048,
                    "keypoint_threshold": 0.0005,
                }
            )
            .eval()
            .to(device)
        )
        matcher = (
            LightGlue(
                features="superpoint",
                n_layers=7,
                depth_confidence=0.9,
                width_confidence=0.95,
                filter_threshold=0.3,
                flash=True,
            )
            .eval()
            .to(device)
        )
        features = namedtuple("features", ["keypoints", "descriptors", "scores"])

    else:
        extractor = KF.KeyNetAffNetHardNet(
            num_features=max_keypoints,
            upright=False,
            device=device,
        )
        features = namedtuple("features", ["lafs", "resps", "descs", "hw"])

    # Extract features
    features_dict = {}
    logger.info("Extracting features from downsampled images...")
    for img in tqdm(img_list):
        if use_superpoint:
            i0 = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            size = i0.shape[:2][::-1]
            scale = resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            i0 = cv2.resize(i0, size_new, interpolation=cv2.INTER_AREA)
            with torch.inference_mode():
                feats = extractor({"image": frame2tensor(i0, device)})
                features_dict[img.name] = sp2lg(feats)
                del feats
        else:
            with torch.inference_mode():
                im0, _ = read_tensor_image(img, resize_max)
                lafs1, resps1, descs1 = extractor(im0)
                features_dict[img.name] = features(
                    lafs1.detach().cpu(),
                    resps1.detach().cpu(),
                    descs1.detach().cpu(),
                    im0.shape[2:],
                )
                del im0, lafs1, resps1, descs1

        torch.cuda.empty_cache()
        timer.update("extraction")

    logger.info("Matching downsampled images...")
    for pair in tqdm(brute_pairs):
        im0_path = pair[0]
        im1_path = pair[1]

        if use_superpoint:
            feats0 = features_dict[im0_path.name]
            feats1 = features_dict[im1_path.name]
            with torch.inference_mode():
                res = matcher({"image0": feats0, "image1": feats1})
            res = rbd2np(res)
            kp0 = feats0["keypoints"].cpu().numpy()[0]
            kp1 = feats1["keypoints"].cpu().numpy()[0]
            mkpts0 = kp0[res["matches"][:, 0], :]
            mkpts1 = kp1[res["matches"][:, 1], :]
            del feats0, feats1, res, kp0, kp1

        else:
            adalam_config = {"device": device}
            lafs1, resps1, descs1, hw1 = (
                features_dict[im0_path.name].lafs.cuda(),
                features_dict[im0_path.name].resps.cuda(),
                features_dict[im0_path.name].descs.cuda(),
                features_dict[im0_path.name].hw,
            )
            lafs2, resps2, descs2, hw2 = (
                features_dict[im1_path.name].lafs.cuda(),
                features_dict[im1_path.name].resps.cuda(),
                features_dict[im1_path.name].descs.cuda(),
                features_dict[im1_path.name].hw,
            )
            with torch.inference_mode():
                dists, idxs = KF.match_adalam(
                    descs1.squeeze(0),
                    descs2.squeeze(0),
                    lafs1,
                    lafs2,  # Adalam takes into account also geometric information
                    config=adalam_config,
                    hw1=hw1,
                    hw2=hw2,  # Adalam also benefits from knowing image size
                )

            mkpts0, mkpts1 = get_matching_keypoints(lafs1, lafs2, idxs)
            del lafs1, resps1, descs1, hw1, lafs2, resps2, descs2, hw2

        if do_geometric_verification:
            _, inlMask = geometric_verification(
                kpts0=mkpts0,
                kpts1=mkpts1,
                threshold=4,
                confidence=0.99,
                quiet=True,
            )
            count_true = np.count_nonzero(inlMask)
            timer.update("geometric verification")

            # print(im0_path.name, im1_path.name, count_true)
            if count_true > min_matches:
                pairs.append((pair))
        else:
            if len(mkpts0) > min_matches:
                pairs.append(pair)

        torch.cuda.empty_cache()

        timer.update("matching")

    timer.print("low-res pair generation")

    return pairs


def pairs_from_covisibility(
    model: Union[str, Path],
    num_matched: int = 10,
) -> List[tuple]:
    """
    Generate image pairs based on covisibility information.

    Args:
        model (Union[str, Path]): The path to the COLMAP model file.
        output (Union[str, Path]): The path to the output file where the pairs will be written.
        num_matched (int, optional): The number of matched images to consider for each image. Defaults to 10.
    """
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

    return pairs


class PairsGenerator:
    def __init__(
        self,
        img_paths: List[Path],
        pair_file: Path,
        strategy: str,
        retrieval_option: Union[str, None] = None,
        overlap: int = 1,
        image_dir: Union[str, Path] = "",
        output_dir: Union[str, Path] = "",
        existing_colmap_model: Union[str, Path] = "",
        **kwargs,
    ) -> None:
        self.img_paths = img_paths
        self.pair_file = pair_file
        self.strategy = strategy
        self.retrieval_option = retrieval_option
        self.overlap = overlap
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.existing_colmap_model = existing_colmap_model

        self.config = kwargs

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
        pairs = pairs_from_lowres(self.img_paths)
        return pairs

    def covisibility(self):
        logger.info("Covisibility matching, generating pairs ..")
        num_matched = self.config.get("num_matched", 10)
        pairs = pairs_from_covisibility(
            model=self.existing_colmap_model,
            num_matched=num_matched,
        )
        return pairs

    def run(self):
        generate_pairs = getattr(self, self.strategy)
        pairs = generate_pairs()
        logger.info(f"Found {len(pairs)} pairs.")

        with open(self.pair_file, "w") as txt_file:
            for p1, p2 in pairs:
                if isinstance(p1, Path):
                    p1 = p1.name
                if isinstance(p2, Path):
                    p2 = p2.name
                txt_file.write(f"{p1} {p2}\n")

        return pairs


if __name__ == "__main__":
    pass
