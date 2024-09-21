# Code by Song Shuang, Geospatial Data Analytics Group, Ohio State Univeristy

import sys
import pydegensac
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import argparse


def Points(img1_path, img2_path, matches_path) -> None:

    assert img1_path.exists(), f"{img1_path} does not exist"
    assert img2_path.exists(), f"{img2_path} does not exist"

    img1 = rio.open(img1_path).read(1)
    img2 = rio.open(img2_path).read(1)

    img1_low, img1_high = np.percentile(img1, [1, 99])
    img2_low, img2_high = np.percentile(img2, [1, 99])

    matches = np.loadtxt(matches_path, delimiter=" ")
    src_pts = matches[:, :2]
    dst_pts = matches[:, 2:4]

    color_pts = np.random.rand(src_pts.shape[0], 1)

    fig, ax = plt.subplots(1, 2)
    ax[0].matshow(img1, cmap="gray", vmin=img1_low, vmax=img1_high)
    ax[1].matshow(img2, cmap="gray", vmin=img2_low, vmax=img2_high)

    ax[0].scatter(src_pts[:, 0], src_pts[:, 1], c=color_pts, cmap="hsv", s=5)
    ax[1].scatter(dst_pts[:, 0], dst_pts[:, 1], c=color_pts, cmap="hsv", s=5)

    plt.show()

def Lines(img1_path, img2_path, matches_path) -> None:

    assert img1_path.exists(), f"{img1_path} does not exist"
    assert img2_path.exists(), f"{img2_path} does not exist"

    img1 = rio.open(img1_path).read(1)
    img2 = rio.open(img2_path).read(1)

    merged_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]))
    merged_img[:img1.shape[0], :img1.shape[1]] = img1
    merged_img[:img2.shape[0], img1.shape[1]:] = img2

    img1_low, img1_high = np.percentile(img1, [1, 99])
    img2_low, img2_high = np.percentile(img2, [1, 99])

    matches = np.loadtxt(matches_path, delimiter=" ")
    src_pts = matches[:, :2]
    dst_pts = matches[:, 2:4]
    F, inlMask = pydegensac.findFundamentalMatrix(
        src_pts,
        dst_pts,
        px_th=1.0,
        conf=0.99,
        max_iters=100000,
    )
    src_pts = src_pts[inlMask, :]
    dst_pts = dst_pts[inlMask, :]

    dst_pts[:, 0] += img1.shape[1]

    fig, ax = plt.subplots()
    ax.matshow(merged_img, cmap="gray", vmin=min(img1_low, img2_low), vmax=max(img1_high, img2_high))
    ax.scatter(src_pts[:, 0], src_pts[:, 1], c="red", cmap="hsv", s=1)
    ax.scatter(dst_pts[:, 0], dst_pts[:, 1], c="red", cmap="hsv", s=1)

    c=0
    for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
        ax.plot([x1, x2], [y1, y2], "g-", linewidth=0.5)
        c+=1
        if c>1000:
            break

    plt.show()





def parse_args():
    parser = argparse.ArgumentParser(description="Show matches between two images.")
    parser.add_argument("--img1", type=str, required=True, help="Path to the first image.")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second image.")
    parser.add_argument("--matches", type=str, required=True, help="Path to the matches file.")
    parser.add_argument("--method", type=str, required=False, default="default", help="Method to use for matching 'points' or 'lines'")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
        
    matches_path = Path(args.matches)
    img1_path = Path(args.img1)
    img2_path = Path(args.img2)
    method = args.method

    assert img1_path.exists(), f"{img1_path} does not exist"
    assert img2_path.exists(), f"{img2_path} does not exist"

    if method == "points":
        Points(img1_path, img2_path, matches_path)
    elif method == "lines":
        Lines(img1_path, img2_path, matches_path)
    else:
        raise ValueError(f"Invalid method: {method}")

