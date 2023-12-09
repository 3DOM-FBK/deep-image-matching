import torch
import cv2
import numpy as np
from PIL import Image
from DeDoDe import dedode_detector_L
from DeDoDe.utils import *

def draw_kpts(im, kpts):    
    kpts = [cv2.KeyPoint(x,y,1.) for x,y in kpts.cpu().numpy()]
    im = np.array(im)
    ret = cv2.drawKeypoints(im, kpts, None)
    return ret


if __name__ == "__main__":
    device = get_best_device()
    detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
    im_path = "assets/im_A.jpg"
    im = Image.open(im_path)
    out = detector.detect_from_path(im_path, num_keypoints = 10_000)
    W,H = im.size
    kps = out["keypoints"]
    kps = detector.to_pixel_coords(kps, H, W)
    Image.fromarray(draw_kpts(im, kps[0])).save("demo/keypoints.png")
