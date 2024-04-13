from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Union

import cv2
import h5py
import kornia as K
import numpy as np
import torch
from tqdm import tqdm

from deep_image_matching.constants import Timer, logger
from deep_image_matching.io.h5 import get_features
from deep_image_matching.thirdparty.hloc.extractors.superpoint import SuperPoint
from deep_image_matching.thirdparty.LightGlue.lightglue import LightGlue
from deep_image_matching.utils.geometric_verification import geometric_verification


class FeaturesDict(TypedDict):
    keypoints: np.ndarray
    descriptors: Optional[np.ndarray]
    scores: Optional[np.ndarray]


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
    feature_path: Path,
    matches_path: Path,
    resize_max: int = 2000,
    min_matches: int = 15,
    do_geometric_verification: bool = False,
    sp_cfg: dict = {},
    lg_cfg: dict = {},
    gv_cfg: dict = {},
) -> List[tuple]:
    timer = Timer(log_level="debug")

    # Generate all possible pairs
    brute_pairs = list(combinations(img_list, 2))

    # Initialize extractor and matcher models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    extractor = SuperPoint(sp_cfg).eval().to(device)
    matcher = LightGlue(**lg_cfg).eval().to(device)

    # ======= TEST FOR EXTRACTING FEATURES IN BATCH
    batch_size = 5

    # Read batch_size images and stack them into the batch tensor
    im0, sc0 = read_tensor_image(img_list[0], resize_max)
    h, w = im0.shape[2:]

    # Pre-allocate the batch tensor with the correct shape
    batch = torch.empty((batch_size, 1, h, w), dtype=torch.float32).to(device)

    # Fill the batch efficiently
    n = 0
    for i, img_path in enumerate(img_list):
        img, scale = read_tensor_image(img_path, resize_max)
        batch[n, ...] = img
        n += 1

        if n == batch_size:
            break  # Batch is full

    with torch.inference_mode():
        feats = extractor({"image": batch})

    # ============

    # Extract features
    logger.info("Extracting features from downsampled images...")
    for img_path in tqdm(img_list):
        img, scale = read_tensor_image(img_path, resize_max)
        with torch.inference_mode():
            feats = extractor({"image": img})
        feats = rbd(feats, to_numpy=True)

        # Scale keypoints
        feats["keypoints"] = feats["keypoints"] / scale

        # Add image_size and dummy tile_idx
        feats["image_size"] = np.array(img.shape[2:])
        feats["tile_idx"] = np.zeros((feats["keypoints"].shape[0], 1))

        # Save features to disk
        save_features_h5(
            feature_path,
            feats,
            img_path.name,
            as_half=True,
        )

        torch.cuda.empty_cache()
        timer.update("extraction")

    logger.info("Matching downsampled images...")
    for im0_path, im1_path in tqdm(brute_pairs):
        # Load features from disk
        feats0 = feats2LG(get_features(feature_path, im0_path.name, as_tensor=True, device=device))
        feats1 = feats2LG(get_features(feature_path, im1_path.name, as_tensor=True, device=device))
        # Match
        with torch.inference_mode():
            res = matcher({"image0": feats0, "image1": feats1})
        matches = res["matches"][0].cpu().numpy()
        kp0 = feats0["keypoints"].cpu().numpy()[0]
        kp1 = feats1["keypoints"].cpu().numpy()[0]
        mkpts0 = kp0[matches[:, 0], :]
        mkpts1 = kp1[matches[:, 1], :]

        if do_geometric_verification:
            # if the resising scale is too small, we need to increase the threshold by the scale factor / 2
            if scale < 0.4 and "threshold" in gv_cfg:
                gv_cfg["threshold"] = np.floor(gv_cfg["threshold"] / (scale * 2))
            _, inlMask = geometric_verification(
                kpts0=mkpts0,
                kpts1=mkpts1,
                **gv_cfg,
            )
            matches = matches[inlMask]
            mkpts0 = mkpts0[inlMask]
            mkpts1 = mkpts1[inlMask]

        if len(matches) < min_matches:
            logger.debug(f"Not enough matches, skipping pair {im0_path.name} - {im1_path.name}")
            continue

        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(im0_path.name)
            group.create_dataset(im1_path.name, data=matches)

        timer.update("matching")

    timer.print("low-res pair generation")

    return True


if __name__ == "__main__":
    from deep_image_matching import setup_logger
    from deep_image_matching.io.h5_to_db import export_to_colmap

    logger = setup_logger("dim", log_level="debug")

    data_dir = Path("datasets/polimi_lowres")

    img_dir = data_dir / "images"

    # Pass configuration to funcion. Temporary!
    sp_cfg = {
        "nms_radius": 2,
        "max_keypoints": 2048,
        "keypoint_threshold": 0.0005,
    }
    lg_cfg = {
        "features": "superpoint",
        "n_layers": 9,
        "filter_threshold": 0.5,
        "depth_confidence": -1,
        "width_confidence": -1,
        "flash": True,
    }
    gv_cfg = {
        "threshold": 4,
        "confidence": 0.9999,
        "quiet": False,
    }
    feature_path = data_dir / "features_lowres.h5"
    matches_path = data_dir / "matches_lowres.h5"

    # Remove existing files (temporary)
    if feature_path.exists():
        feature_path.unlink()
    if matches_path.exists():
        matches_path.unlink()

    # Get image paths
    img_paths = list(img_dir.glob("*"))

    # Match low resolution imagess
    match_low_resolution(
        img_paths,
        feature_path,
        matches_path,
        resize_max=2000,
        min_matches=15,
        do_geometric_verification=True,
        sp_cfg=sp_cfg,
        lg_cfg=lg_cfg,
        gv_cfg=gv_cfg,
    )

    export_to_colmap(
        img_dir,
        feature_path,
        matches_path,
        data_dir / "database.db",
    )

    print("Done")
