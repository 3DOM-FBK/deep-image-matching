import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .utils import compute_T_with_imagesize


def extract_corrs(img1, img2, bi_check=True, numkp=2000):
    # Create correspondences with SIFT
    sift = cv2.xfeatures2d.SIFT_create(
            nfeatures=numkp, contrastThreshold=1e-5)

    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    xy1 = np.array([_kp.pt for _kp in kp1])
    xy2 = np.array([_kp.pt for _kp in kp2])

    distmat = np.sqrt(
        np.sum(desc1**2, axis=1, keepdims=True) + \
        np.sum(desc2**2, axis=1) - \
        2 * np.dot(desc1, desc2.T)
    )
    idx_sort0 = np.argsort(distmat, axis=1)[:, 0] # choose the best from 0 to 1
    corrs = np.concatenate([xy1, xy2.take(idx_sort0, axis=0)], axis=1)
    if bi_check:
        idx_sort1 = np.argsort(distmat, axis=0)[0, :] # choose the best from 1 to 0
        bi_mat = idx_sort1[idx_sort0] == np.arange(idx_sort0.shape[0])
        corrs = corrs[bi_mat]
    return corrs


def visualize_corrs(img1, img2, corrs, mask=None):
    if mask is None:
        mask = np.ones(len(corrs)).astype(bool)
 
    scale1 = 1.0
    scale2 = 1.0
    if img1.shape[1] > img2.shape[1]:
        scale2 = img1.shape[1] / img2.shape[1]
        w = img1.shape[1]
    else:
        scale1 = img2.shape[1] / img1.shape[1]
        w = img2.shape[1]
    # Resize if too big
    max_w = 400
    if w > max_w:
        scale1 *= max_w / w
        scale2 *= max_w / w
    img1 = np.array(Image.fromarray(img1).resize((int(img1.shape[1] * scale1), int(img1.shape[0] * scale1))))
    img2 = np.array(Image.fromarray(img2).resize((int(img2.shape[1] * scale2), int(img2.shape[0] * scale2))))

#   img1 = imresize(img1, scale1)
#   img2 = imresize(img2, scale2)

    x1, x2 = corrs[:, :2], corrs[:, 2:]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img1.dtype)
    img[:h1, :w1] = img1
    img[h1:, :w2] = img2
    # Move keypoints to coordinates to image coordinates
    x1 = x1 * scale1
    x2 = x2 * scale2
    # recompute the coordinates for the second image
    x2p = x2 + np.array([[0, h1]])
    fig = plt.figure(frameon=False)
    fig = plt.imshow(img)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.1],
    ]
    lw = .5
    alpha = 1

    # Draw outliers
    _x1 = x1[~mask]
    _x2p = x2p[~mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[1],
    )

    # Draw Inliers
    _x1 = x1[mask]
    _x2p = x2p[mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[0],
    )

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def get_T_from_K(K):
    cx = K[0, 2]
    cy = K[1, 2]
    w = cx * 2 + 1.0
    h = cy * 2 + 1.0 
    T = compute_T_with_imagesize(w, h)
    return T


def norm_points_with_T(x, T):
    x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
    return x
