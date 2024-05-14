# Code by Song Shuang, Geospatial Data Analytics Group, Ohio State Univeristy

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio


def _main() -> int:
    matches_path = Path(r"./img1_img2.txt")

    img1_path = Path(r"./images") / "img1.jpg"
    img2_path = Path(r"./images") / "img2.jpg"

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


if __name__ == "__main__":
    _main()
