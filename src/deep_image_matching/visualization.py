import importlib
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def viz_matches_mpl(
    image0: np.ndarray,
    image1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    save_path: str = None,
    hide_fig: bool = True,
    **config,
) -> None:
    if hide_fig:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")  # Use the Agg backend for rendering

    # Get config
    colors = config.get("c", config.get("color", ["r", "r"]))
    if isinstance(colors, str):
        colors = [colors, colors]
    s = config.get("s", 2)
    figsize = config.get("figsize", (20, 8))

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], s=s, c=colors[0])
    ax[0].axis("equal")
    ax[1].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], s=s, c=colors[1])
    ax[1].axis("equal")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if hide_fig is False:
        plt.show()
    else:
        plt.close(fig)


def viz_matches_cv2(
    image0: np.ndarray,
    image1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    save_path: str = None,
    pts_col: Tuple[int] = (0, 0, 255),
    point_size: int = 1,
    line_col: Tuple[int] = (0, 255, 0),
    line_thickness: int = 1,
    margin: int = 10,
    autoresize: bool = True,
    max_long_edge: int = 2000,
    jpg_quality: int = 80,
) -> np.ndarray:
    """Plot matching points between two images using OpenCV.

    Args:
        image0: The first image.
        image1: The second image.
        pts0: List of 2D points in the first image.
        pts1: List of 2D points in the second image.
        pts_col: RGB color of the points.
        point_size: Size of the circles representing the points.
        line_col: RGB color of the matching lines.
        line_thickness: Thickness of the lines connecting the points.
        path: Path to save the output image.
        margin: Margin between the two images in the output.

    Returns:
        np.ndarrya: The output image.
    """
    if image0.ndim > 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    if image1.ndim > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    if autoresize:
        H0, W0 = image0.shape[:2]
        H1, W1 = image1.shape[:2]
        max_size = max(H0, W0, H1, W1)
        scale_factor = max_long_edge / max_size
        new_W0, new_H0 = int(W0 * scale_factor), int(H0 * scale_factor)
        new_W1, new_H1 = int(W1 * scale_factor), int(H1 * scale_factor)

        image0 = cv2.resize(image0, (new_W0, new_H0))
        image1 = cv2.resize(image1, (new_W1, new_H1))

        # Scale the keypoints accordingly
        pts0 = (pts0 * scale_factor).astype(int)
        pts1 = (pts1 * scale_factor).astype(int)

    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin :] = image1
    out = np.stack([out] * 3, -1)

    mkpts0, mkpts1 = np.round(pts0).astype(int), np.round(pts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        if line_thickness > -1:
            # draw lines between matching keypoints
            cv2.line(
                out,
                (x0, y0),
                (x1 + margin + W0, y1),
                color=line_col,
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_size, pts_col, -1, lineType=cv2.LINE_AA)
        cv2.circle(
            out,
            (x1 + margin + W0, y1),
            point_size,
            pts_col,
            -1,
            lineType=cv2.LINE_AA,
        )
    if save_path is not None:
        cv2.imwrite(str(save_path), out, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])

    return out
