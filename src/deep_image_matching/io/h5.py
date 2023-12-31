from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from .. import logger


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator="_")


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def get_features(
    path: Path,
    name: str,
    as_tensor: bool = False,
    device: "str" = "cuda",
) -> dict:
    with h5py.File(str(path), "r", libver="latest") as fd:
        if name in fd:
            try:
                kpts = np.array(fd[name]["keypoints"]).astype(np.float32)
                descr = np.array(fd[name]["descriptors"]).astype(np.float32)

            except KeyError:
                raise ValueError(f"Cannot find keypoints and descriptors in {path}")

            try:
                feature_path = fd[name]["feature_path"][()].decode("utf-8")
                im_path = fd[name]["im_path"][()].decode("utf-8")
                feats = {
                    "feature_path": feature_path,
                    "im_path": im_path,
                    "keypoints": kpts,
                    "descriptors": descr,
                }
            except KeyError:
                logger.warning(
                    "Cannot find feature_path and im_path in the hdf5 file. "
                    "Returning only keypoints and descriptors. Matching with end-to-end matchers (loftr, roma) will not work."
                )
                feats = {"keypoints": kpts, "descriptors": descr}

            for k in ["tile_idx", "scores"]:
                if k in fd[name]:
                    feats[k] = np.array(fd[name][k]).astype(np.float32)
                else:
                    logger.warning(f"Cannot find {k} in {path}")
            k = "image_size"
            if k in fd[name]:
                feats[k] = np.array(fd[name][k]).astype(np.int32)
        else:
            raise ValueError(f"Cannot find image {name} in {path}")

        if as_tensor:
            if device not in ("cpu", "cuda"):
                raise ValueError(f"Unknown device {device}. It must be cpu or cuda.")
            device = "cuda" if torch.cuda.is_available() and not device else "cpu"
            feats = {
                k: torch.tensor(v, dtype=torch.float, device=device)
                for k, v in feats.items()
            }

        return feats


def get_keypoints(
    path: Path, name: str, return_uncertainty: bool = False
) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def get_matches(path: Path, name0: str, name1: str) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        group = hfile[name0]
        matches = group[name1][()]

    return matches


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f"Could not find pair {(name0, name1)}... "
        "Maybe you matched with a different list of pairs? "
    )
