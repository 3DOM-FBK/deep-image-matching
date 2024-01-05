import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import cv2
import numpy as np


def draw_matches(im_A, kpts_A, im_B, kpts_B):    
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B.cpu().numpy()]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, 
                    matches_A_to_B, None)
    return ret

if __name__ == "__main__":
    device = get_best_device()
    detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
    descriptor = dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth", map_location = device))
    matcher = DualSoftMaxMatcher()

    im_A_path = "assets/im_A.jpg"
    im_B_path = "assets/im_B.jpg"
    im_A = Image.open(im_A_path)
    im_B = Image.open(im_B_path)
    W_A, H_A = im_A.size
    W_B, H_B = im_B.size


    detections_A = detector.detect_from_path(im_A_path, num_keypoints = 10_000)
    keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
    detections_B = detector.detect_from_path(im_B_path, num_keypoints = 10_000)
    keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]
    description_A = descriptor.describe_keypoints_from_path(im_A_path, keypoints_A)["descriptions"]
    description_B = descriptor.describe_keypoints_from_path(im_B_path, keypoints_B)["descriptions"]
    matches_A, matches_B, batch_ids = matcher.match(keypoints_A, description_A,
        keypoints_B, description_B,
        P_A = P_A, P_B = P_B,
        normalize = True, inv_temp=20, threshold = 0.01)#Increasing threshold -> fewer matches, fewer outliers

    matches_A, matches_B = matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)

    Image.fromarray(draw_matches(im_A, matches_A, im_B, matches_B)).save("demo/matches.png")