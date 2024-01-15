from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Union

import cv2
import h5py
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from tqdm import tqdm

from deep_image_matching import Timer, logger
from deep_image_matching.hloc.extractors.superpoint import SuperPoint
from deep_image_matching.io.h5 import get_features
from deep_image_matching.thirdparty.LightGlue.lightglue import LightGlue
from deep_image_matching.utils.geometric_verification import geometric_verification


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: Optional[np.ndarray]
    scores: Optional[np.ndarray]
    shape: Tuple[int, int]
    lafs: Optional[np.ndarray]
    resps: Optional[np.ndarray]
    descs: Optional[np.ndarray]


def get_matching_keypoints(lafs1, lafs2, idxs):
    mkpts1 = KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
    mkpts2 = KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
    return mkpts1, mkpts2


def read_tensor_image(
    path: Path, resize_to: int = 500, device="cuda"
) -> Tuple[np.ndarray, float]:
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    size = img.shape[:2][::-1]
    scale = resize_to / max(size)
    size_new = tuple(int(round(x * scale)) for x in size)
    img = cv2.resize(img, size_new)
    img = K.image_to_tensor(img, False).float() / 255.0
    img = img.to(device)

    return img, scale


def feats2LG(feats: FeaturesDict) -> dict:
    # Remove elements from list/tuple
    feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
    # Move descriptors dimension to last
    if "descriptors" in feats.keys():
        if feats["descriptors"].shape[-1] != 256:
            feats["descriptors"] = feats["descriptors"].T
    # Add batch dimension
    feats = {k: v[None] for k, v in feats.items()}
    return feats


def rbd(feats: dict, to_numpy: bool = False) -> dict:
    """Remove batch dimension from features dict"""
    feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}
    if to_numpy:
        feats = {k: v.detach().cpu().numpy() for k, v in feats.items()}
    return feats


def save_features_h5(feature_path, features, im_name, as_half=True):
    if as_half:
        feat_dtype = np.float16
        for k in features:
            if not isinstance(features[k], np.ndarray):
                continue
            dt = features[k].dtype
            if (dt == np.float32) and (dt != feat_dtype):
                features[k] = features[k].astype(feat_dtype)
    else:
        feat_dtype = np.float32

    with h5py.File(str(feature_path), "a", libver="latest") as fd:
        try:
            if im_name in fd:
                del fd[im_name]
            grp = fd.create_group(im_name)
            for k, v in features.items():
                grp.create_dataset(
                    k,
                    data=v,
                    dtype=feat_dtype,
                    compression="gzip",
                    compression_opts=9,
                )
        except OSError as error:
            if "No space left on device" in error.args[0]:
                logger.error(
                    "Out of disk space: storing features on disk can take "
                    "significant space, did you enable the as_half flag?"
                )
                del grp, fd[im_name]
            raise error


def match_low_resolution(
    img_list: List[Union[str, Path]],
    resize_max: int = 2000,
    min_matches: int = 20,
    max_keypoints: int = 1024,
    use_superpoint: bool = True,
    do_geometric_verification: bool = False,
) -> List[tuple]:
    timer = Timer(log_level="debug")

    use_superpoint = True

    # Define paths
    feature_path = Path("sandbox/features_lowres.h5")
    matches_path = Path("sandbox/matches_lowres.h5")

    # Remove previous files (temporary solution)
    if feature_path.exists():
        feature_path.unlink()
    if matches_path.exists():
        matches_path.unlink()

    brute_pairs = list(combinations(img_list, 2))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if use_superpoint:
        sp_cfg = {
            "nms_radius": 3,
            "max_keypoints": 2048,
            "keypoint_threshold": 0.0005,
        }
        extractor = SuperPoint(sp_cfg).eval().to(device)
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
    else:
        extractor = KF.KeyNetAffNetHardNet(
            num_features=max_keypoints,
            upright=False,
            device=device,
        )

    # Extract features
    logger.info("Extracting features from downsampled images...")
    threads = []
    for img_path in tqdm(img_list):
        if use_superpoint:
            img, scale = read_tensor_image(img_path, resize_max)
            with torch.inference_mode():
                feats = extractor({"image": img})
            feats = rbd(feats, to_numpy=True)

            # Scale keypoints
            feats["keypoints"] = feats["keypoints"] / scale

            # Add dummy tile_idx and image_size
            feats["tile_idx"] = np.zeros((feats["keypoints"].shape[0], 1))
            feats["image_size"] = np.array(img.shape[2:])

            # Save features to disk
            save_features_h5(
                feature_path,
                feats,
                img_path.name,
                as_half=True,
            )

        else:
            with torch.inference_mode():
                im0, _ = read_tensor_image(img, resize_max)
                lafs, resps, descs = extractor(im0)
                feats = FeaturesDict(
                    lafs=lafs.detach().cpu(),
                    resps=resps.detach().cpu(),
                    descrs=descs.detach().cpu(),
                    shape=im0.shape[2:],
                )
                del im0, lafs, resps, descs

        torch.cuda.empty_cache()
        timer.update("extraction")

    logger.info("Matching downsampled images...")
    for im0_path, im1_path in tqdm(brute_pairs):
        if use_superpoint:
            # Load features from disk
            feats0 = feats2LG(
                get_features(feature_path, im0_path.name, as_tensor=True, device=device)
            )
            feats1 = feats2LG(
                get_features(feature_path, im1_path.name, as_tensor=True, device=device)
            )
            # Match
            with torch.inference_mode():
                res = matcher({"image0": feats0, "image1": feats1})
            matches = res["matches"][0].cpu().numpy()
            kp0 = feats0["keypoints"].cpu().numpy()[0]
            kp1 = feats1["keypoints"].cpu().numpy()[0]
            mkpts0 = kp0[matches[:, 0], :]
            mkpts1 = kp1[matches[:, 1], :]

        else:
            adalam_config = {"device": device}
            # lafs1, resps1, descs1, hw1 = (
            #     features_dict[im0_path.name].lafs.cuda(),
            #     features_dict[im0_path.name].resps.cuda(),
            #     features_dict[im0_path.name].descs.cuda(),
            #     features_dict[im0_path.name].hw,
            # )
            # lafs2, resps2, descs2, hw2 = (
            #     features_dict[im1_path.name].lafs.cuda(),
            #     features_dict[im1_path.name].resps.cuda(),
            #     features_dict[im1_path.name].descs.cuda(),
            #     features_dict[im1_path.name].hw,
            # )
            # with torch.inference_mode():
            #     dists, idxs = KF.match_adalam(
            #         descs1.squeeze(0),
            #         descs2.squeeze(0),
            #         lafs1,
            #         lafs2,  # Adalam takes into account also geometric information
            #         config=adalam_config,
            #         hw1=hw1,
            #         hw2=hw2,  # Adalam also benefits from knowing image size
            #     )

            # mkpts0, mkpts1 = get_matching_keypoints(lafs1, lafs2, idxs)
            # del lafs1, resps1, descs1, hw1, lafs2, resps2, descs2, hw2

        if do_geometric_verification:
            _, inlMask = geometric_verification(
                kpts0=mkpts0,
                kpts1=mkpts1,
                threshold=4,
                confidence=0.99,
                quiet=True,
            )
            matches = matches[inlMask]
            mkpts0 = mkpts0[inlMask]
            mkpts1 = mkpts1[inlMask]

        if len(matches) < min_matches:
            continue

        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(im0_path.name)
            group.create_dataset(im1_path.name, data=matches)

        timer.update("matching")

    timer.print("low-res pair generation")


if __name__ == "__main__":
    img_dir = Path("datasets/casalbagliano/images")
    img_paths = list(img_dir.glob("*"))

    match_low_resolution(img_paths, resize_max=1000, min_matches=20)

    print("Done")
