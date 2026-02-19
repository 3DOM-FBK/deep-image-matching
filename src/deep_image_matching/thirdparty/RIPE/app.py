# This is a small gradio interface to access our RIPE keypoint extractor.
# You can either upload two images or use one of the example image pairs.

import os

import gradio as gr
from PIL import Image

from ripe import vgg_hyper

SEED = 32000
os.environ["PYTHONHASHSEED"] = str(SEED)

import random
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
import cv2
import kornia.feature as KF
import kornia.geometry as KG

from ripe.utils.utils import cv2_matches_from_kornia, to_cv_kpts

MIN_SIZE = 512
MAX_SIZE = 768

description_text = """
<p align='center'>
  <h1 align='center'>🌊🌺 ICCV 2025 🌺🌊</h1>
  <p align='center'>
    <a href='https://scholar.google.com/citations?user=ybMR38kAAAAJ'>Johannes Künzel</a> · 
    <a href='https://scholar.google.com/citations?user=5yTuyGIAAAAJ'>Anna Hilsmann</a> · 
    <a href='https://scholar.google.com/citations?user=BCElyCkAAAAJ'>Peter Eisert</a>
  </p>
  <h2 align='center'>
    <a href='???'>Arxiv</a> | 
    <a href='???'>Project Page</a> | 
    <a href='???'>Code</a>
  </h2>
</p>

<br/>
<div align='center'>

### This demo showcases our new keypoint extractor model, RIPE (Reinforcement Learning on Unlabeled Image Pairs for Robust Keypoint Extraction).

### RIPE is trained without requiring pose or depth supervision or artificial augmentations. By leveraging reinforcement learning, it learns to extract keypoints solely based on whether an image pair depicts the same scene or not.

### For more detailed information, please refer to our [paper](link to be added).

The demo code extracts the 2048-top keypoints from the two input images. It uses the mutual nearest neighbor (MNN) descriptor matcher from kornia to find matches between the two images.
If the number of matches is greater than 8, it applies RANSAC to filter out outliers based on the inlier threshold provided by the user.
Images are resized to fit within a maximum size of 2048x2048 pixels with maintained aspect ratio.

</div>
"""

path_weights = Path(
    "/media/jwkuenzel/work/projects/CVG_Reinforced_Keypoints/output/train/ablation_iccv/inlier_threshold/1571243/2025-02-19/14-00-10_789013/model_inlier_threshold_best.pth"
)

model = vgg_hyper(path_weights)


def get_new_image_size(image, min_size=1600, max_size=2048):
    """
    Get a new size for the image that is scaled to fit between min_size and max_size while maintaining the aspect ratio.

    Args:
        image (PIL.Image): Input image.
        min_size (int): Minimum allowed size for width and height.
        max_size (int): Maximum allowed size for width and height.

    Returns:
        tuple: New size (width, height) for the image.
    """
    width, height = image.size

    aspect_ratio = width / height
    if width > height:
        new_width = max(min_size, min(max_size, width))
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max(min_size, min(max_size, height))
        new_width = int(new_height * aspect_ratio)

    new_size = (new_width, new_height)

    return new_size


def extract_keypoints(image1, image2, inl_th):
    """
    Extract keypoints from two input images using the RIPE model.

    Args:
        image1 (PIL.Image): First input image.
        image2 (PIL.Image): Second input image.
        inl_th (float): RANSAC inlier threshold.

    Returns:
        dict: A dictionary containing keypoints and matches.
    """
    log_text = "Extracting keypoints and matches with RIPE\n"

    log_text += f"Image 1 size: {image1.size}\n"
    log_text += f"Image 2 size: {image2.size}\n"

    # check not larger than 2048x2048
    new_size = get_new_image_size(image1, min_size=MIN_SIZE, max_size=MAX_SIZE)
    image1 = image1.resize(new_size)

    new_size = get_new_image_size(image2, min_size=MIN_SIZE, max_size=MAX_SIZE)
    image2 = image2.resize(new_size)

    log_text += f"Resized Image 1 size: {image1.size}\n"
    log_text += f"Resized Image 2 size: {image2.size}\n"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    image1_original = image1.copy()
    image2_original = image2.copy()

    # convert PIL images to numpy arrays
    image1_original = np.array(image1_original)
    image2_original = np.array(image2_original)

    # convert PIL images to tensors
    image1 = torch.tensor(np.array(image1)).permute(2, 0, 1).float() / 255.0
    image2 = torch.tensor(np.array(image2)).permute(2, 0, 1).float() / 255.0

    image1 = image1.to(dev).unsqueeze(0)  # Add batch dimension
    image2 = image2.to(dev).unsqueeze(0)  # Add batch dimension

    kpts_1, desc_1, score_1 = model.detectAndCompute(image1, threshold=0.5, top_k=2048)
    kpts_2, desc_2, score_2 = model.detectAndCompute(image2, threshold=0.5, top_k=2048)

    log_text += f"Number of keypoints in image 1: {kpts_1.shape[0]}\n"
    log_text += f"Number of keypoints in image 2: {kpts_2.shape[0]}\n"

    matcher = KF.DescriptorMatcher("mnn")  # threshold is not used with mnn
    match_dists, match_idxs = matcher(desc_1, desc_2)

    log_text += f"Number of MNN matches: {match_idxs.shape[0]}\n"

    cv2_matches = cv2_matches_from_kornia(match_dists, match_idxs)

    do_ransac = match_idxs.shape[0] > 8

    if do_ransac:
        matched_pts_1 = kpts_1[match_idxs[:, 0]]
        matched_pts_2 = kpts_2[match_idxs[:, 1]]

        H, mask = KG.ransac.RANSAC(model_type="fundamental", inl_th=inl_th)(matched_pts_1, matched_pts_2)
        matchesMask = mask.int().ravel().tolist()

        log_text += f"RANSAC found {mask.sum().item()} inliers out of {mask.shape[0]} matches with an inlier threshold of {inl_th}.\n"
    else:
        log_text += "Not enough matches for RANSAC, skipping RANSAC step.\n"

    kpts_1 = to_cv_kpts(kpts_1, score_1)
    kpts_2 = to_cv_kpts(kpts_2, score_2)

    keypoints_raw_1 = cv2.drawKeypoints(image1_original, kpts_1, image1_original, color=(0, 255, 0))
    keypoints_raw_2 = cv2.drawKeypoints(image2_original, kpts_2, image2_original, color=(0, 255, 0))

    # pad height smaller image to match the height of the larger image
    if keypoints_raw_1.shape[0] < keypoints_raw_2.shape[0]:
        pad_height = keypoints_raw_2.shape[0] - keypoints_raw_1.shape[0]
        keypoints_raw_1 = np.pad(
            keypoints_raw_1, ((0, pad_height), (0, 0), (0, 0)), mode="constant", constant_values=255
        )
    elif keypoints_raw_1.shape[0] > keypoints_raw_2.shape[0]:
        pad_height = keypoints_raw_1.shape[0] - keypoints_raw_2.shape[0]
        keypoints_raw_2 = np.pad(
            keypoints_raw_2, ((0, pad_height), (0, 0), (0, 0)), mode="constant", constant_values=255
        )

    # concatenate keypoints images horizontally
    keypoints_raw = np.concatenate((keypoints_raw_1, keypoints_raw_2), axis=1)
    keypoints_raw_pil = Image.fromarray(keypoints_raw)

    result_raw = cv2.drawMatches(
        image1_original,
        kpts_1,
        image2_original,
        kpts_2,
        cv2_matches,
        None,
        matchColor=(0, 255, 0),
        matchesMask=None,
        # matchesMask=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if not do_ransac:
        result_ransac = None
    else:
        result_ransac = cv2.drawMatches(
            image1_original,
            kpts_1,
            image2_original,
            kpts_2,
            cv2_matches,
            None,
            matchColor=(0, 255, 0),
            matchesMask=matchesMask,
            singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    # convert to PIL Image
    result_raw_pil = Image.fromarray(result_raw)
    if result_ransac is not None:
        result_ransac_pil = Image.fromarray(result_ransac)
    else:
        result_ransac_pil = None

    return log_text, result_ransac_pil, result_raw_pil, keypoints_raw_pil


demo = gr.Interface(
    fn=extract_keypoints,
    inputs=[
        gr.Image(type="pil", label="Image 1"),
        gr.Image(type="pil", label="Image 2"),
        gr.Slider(
            minimum=0.1,
            maximum=3.0,
            step=0.1,
            value=0.5,
            label="RANSAC inlier threshold",
            info="Threshold for RANSAC inlier detection. Lower values may yield fewer inliers but more robust matches.",
        ),
    ],
    outputs=[
        gr.Textbox(type="text", label="Log"),
        gr.Image(type="pil", label="Keypoints and Matches (RANSAC)"),
        gr.Image(type="pil", label="Keypoints and Matches"),
        gr.Image(type="pil", label="Keypoint Detection Results"),
    ],
    title="RIPE: Reinforcement Learning on Unlabeled Image Pairs for Robust Keypoint Extraction",
    description=description_text,
    examples=[
        [
            "assets_gradio/all_souls_000013.jpg",
            "assets_gradio/all_souls_000055.jpg",
        ],
        [
            "assets_gradio/167170681_0e5c42fd21_o.jpg",
            "assets_gradio/170804731_6bf4fbecd4_o.jpg",
        ],
        [
            "assets_gradio/4171014767_0fe879b783_o.jpg",
            "assets_gradio/4174108353_20422632d6_o.jpg",
        ],
    ],
    flagging_mode="never",
    theme="default",
)
demo.launch()
