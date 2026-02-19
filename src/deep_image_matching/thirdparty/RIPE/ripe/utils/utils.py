import random
from typing import List

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize

from . import get_pylogger

log = get_pylogger(__name__)


def gridify(x, window_size):
    """Turn a tensor of BxCxHxW into a tensor of
    BxCx(H//window_size)x(W//window_size)x(window_size**2)

    Params:
        x: Input tensor of shape BxCxHxW
        window_size: Size of the window

    Returns:
        x: Output tensor of shape BxCx(H//window_size)x(W//window_size)x(window_size**2)
    """

    assert x.dim() == 4, "Input tensor x must have 4 dimensions"

    B, C, H, W = x.shape
    x = (
        x.unfold(2, window_size, window_size)
        .unfold(3, window_size, window_size)
        .reshape(B, C, H // window_size, W // window_size, window_size**2)
    )

    return x


def get_grid(B, H, W, device):
    x1_n = torch.meshgrid(*[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)])
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def cv2_matches_from_kornia(match_dists: torch.Tensor, match_idxs: torch.Tensor) -> List[cv2.DMatch]:
    return [cv2.DMatch(idx[0].item(), idx[1].item(), d.item()) for idx, d in zip(match_idxs, match_dists)]


def to_cv_kpts(kpts, scores):
    kp = kpts.cpu().numpy().astype(np.int16)
    s = scores.cpu().numpy()

    cv_kp = [cv2.KeyPoint(kp[i][0], kp[i][1], 6, 0, s[i]) for i in range(len(kp))]

    return cv_kp


def resize_image(image, min_size=512, max_size=768):
    """Resize image to a new size while maintaining the aspect ratio.

    Params:
        image (torch.tensor): Image to be resized.
        min_size (int): Minimum size of the smaller dimension.
        max_size (int): Maximum size of the larger dimension.

    Returns:
        image: Resized image.
    """

    h, w = image.shape[-2:]

    aspect_ratio = w / h

    if w > h:
        new_w = max(min_size, min(max_size, w))
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = max(min_size, min(max_size, h))
        new_w = int(new_h * aspect_ratio)

    new_size = (new_h, new_w)

    image = resize(image, new_size)

    return image


def get_rewards(
    reward,
    kps1,
    kps2,
    selected_mask1,
    selected_mask2,
    padding_mask1,
    padding_mask2,
    rel_idx_matches,
    abs_idx_matches,
    ransac_inliers,
    label,
    penalty=0.0,
    use_whitening=False,
    selected_only=False,
    filter_mode=None,
):
    with torch.no_grad():
        reward *= 1.0 if label else -1.0

        dense_returns = torch.zeros((len(kps1), len(kps2)), device=kps1.device)

        if filter_mode == "ignore":
            dense_returns[
                abs_idx_matches[:, 0][ransac_inliers],
                abs_idx_matches[:, 1][ransac_inliers],
            ] = reward
        elif filter_mode == "punish":
            in_padding_area = (
                padding_mask1[abs_idx_matches[:, 0]] & padding_mask2[abs_idx_matches[:, 1]]
            )  # both in the image area (not in padding area)

            dense_returns[
                abs_idx_matches[:, 0][ransac_inliers & in_padding_area],
                abs_idx_matches[:, 1][ransac_inliers & in_padding_area],
            ] = reward
            dense_returns[
                abs_idx_matches[:, 0][ransac_inliers & ~in_padding_area],
                abs_idx_matches[:, 1][ransac_inliers & ~in_padding_area],
            ] = -1.0
        else:
            raise ValueError(f"Unknown filter mode: {filter_mode}")

        if selected_only:
            dense_returns = dense_returns[selected_mask1, :][:, selected_mask2]
        if filter_mode == "ignore" and not selected_only:
            dense_returns = dense_returns[padding_mask1, :][:, padding_mask2]

        if penalty != 0.0:
            # pos. pair: small penalty for not finding a match
            # neg. pair: small reward for not finding a match
            penalty_val = penalty if label else -penalty

            dense_returns[dense_returns == 0.0] = penalty_val

        if use_whitening:
            dense_returns = (dense_returns - dense_returns.mean()) / (dense_returns.std() + 1e-6)

    return dense_returns


def get_other_random_id(idx: int, len_dataset: int, min_dist: int = 20):
    for _ in range(10):
        tgt_id = random.randint(0, len_dataset - 1)
        if abs(idx - tgt_id) >= min_dist:
            return tgt_id

    raise ValueError(f"Could not find target image with distance >= {min_dist} from source image {idx}")


def cv_resize_and_pad_to_shape(image, new_shape, padding_color=(0, 0, 0)):
    """Resizes image to new_shape with maintaining the aspect ratio and pads with padding_color if
    needed.

    Params:
        image: Image to be resized.
        new_shape: Expected (height, width) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    h, w = image.shape[:2]

    scale_h = new_shape[0] / h
    scale_w = new_shape[1] / w

    scale = None
    if scale_w * h > new_shape[0]:
        scale = scale_h
    elif scale_h * w > new_shape[1]:
        scale = scale_w
    else:
        scale = max(scale_h, scale_w)

    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    image = cv2.resize(image, (new_w, new_h))

    missing_h = new_shape[0] - new_h
    missing_w = new_shape[1] - new_w

    top, bottom = missing_h // 2, missing_h - (missing_h // 2)
    left, right = missing_w // 2, missing_w - (missing_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image
