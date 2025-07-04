import cv2
import torch
import numpy as np
import kornia as K
from kornia import feature as KF
from pathlib import Path
from enum import Enum
from typing import Tuple

from ..ncc import refinement_laf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Quality(Enum):
    """Enumeration for matching quality."""

    LOWEST = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4


def get_size_by_quality(
    quality: Quality,
    size: Tuple[int, int],  # usually (width, height)
):
    quality_size_map = {
        Quality.HIGHEST: 2,
        Quality.HIGH: 1,
        Quality.MEDIUM: 1 / 2,
        Quality.LOW: 1 / 4,
        Quality.LOWEST: 1 / 8,
    }
    f = quality_size_map[quality]
    return (int(size[0] * f), int(size[1] * f))

def load_image_np(img_path: Path, as_float: bool = True, grayscale: bool = False):
    image = cv2.imread(str(img_path))
    if as_float:
        image = image.astype(np.float32)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def frame2tensor(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    image = K.image_to_tensor(np.array(image), False).float() / 255.0
    image = image.to(device)
    if image.shape[1] > 2:
        image = K.color.bgr_to_rgb(image)
        image = K.color.rgb_to_grayscale(image)
    return image

def resize_image(quality: Quality, image: np.ndarray, interp: str = "cv2_area") -> Tuple[np.ndarray]:
    """
    Resize images based on the specified quality.
    Args:
        quality (Quality): The quality level for resizing.
        image (np.ndarray): The first image.
    Returns:
        Tuple[np.ndarray]: Resized images.
    """
    # If quality is HIGHEST, force interpolation to cv2_cubic
    # if quality == Quality.HIGHEST:
    #     interp = "cv2_cubic"
    # if quality == Quality.HIGH:
    #     return image  # No resize
    new_size = get_size_by_quality(quality, image.shape[:2])
    return resize_image(image, (new_size[1], new_size[0]), interp=interp)

class loftr_module:
    def __init__(self, **args):
        self.outdoor = True
                
        for k, v in args.items():
           setattr(self, k, v)
           
        if self.outdoor == True:
            pretrained = 'outdoor'
        else:
            pretrained = 'indoor_new'

        with torch.inference_mode():
            self.matcher = KF.LoFTR(pretrained=pretrained).to(device).eval()


    def get_id(self):
        return ('loftr_outdoor_' + str(self.outdoor)).lower()


    def run(self, **args):
        with torch.inference_mode():
            image0 = load_image_np(Path(args['im1']))
            image1 = load_image_np(Path(args['im2']))
            # image0_ = resize_image(Quality.MEDIUM, image0)
            # image1_ = resize_image(Quality.MEDIUM, image1)
            timg0_ = frame2tensor(image0, device)
            timg1_ = frame2tensor(image1, device)
            input_dict = {"image0": timg0_, "image1": timg1_}
            correspondences = self.matcher(input_dict)
        
        kps1 = correspondences["keypoints0"].squeeze().detach().to(device)
        kps2 = correspondences["keypoints1"].squeeze().detach().to(device)

        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!

        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_laf}
