import os
import sys
import torch
import numpy as np
import math
import cv2

from models.liftfeat_wrapper import LiftFeat,MODEL_PATH

import argparse

parser=argparse.ArgumentParser(description='HPatch dataset evaluation script')
parser.add_argument('--name',type=str,default='LiftFeat',help='experiment name')
parser.add_argument('--img1',type=str,default='./assert/ref.jpg',help='reference image path')
parser.add_argument('--img2',type=str,default='./assert/query.jpg',help='query image path')
parser.add_argument('--size',type=str,default=None,help='Resize images to w,h, None means disable resize')
parser.add_argument('--use_opencv_match',action='store_true',help='Enable OpenCV match function')
parser.add_argument('--gpu',type=str,default='0',help='GPU ID')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    h, w = img1.shape[:2]
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, h*0.01+w*0.01, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


def opencv_knn_match(descs1,descs2,kpts1,kpts2):
    bf = cv2.BFMatcher()
    
    matches = bf.knnMatch(descs1,descs2,k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)
    
    mkpts1 = [];mkpts2 = []
    
    for m in good_matches:
        mkpt1=kpts1[m.queryIdx];mkpt2=kpts2[m.trainIdx]
        mkpts1.append(mkpt1);mkpts2.append(mkpt2)
        
    mkpts1 = np.array(mkpts1)
    mkpts2 = np.array(mkpts2)
    
    return mkpts1,mkpts2


if __name__=="__main__":
    if args.size:
        print(f'resize images to {args.size}')
        w=int(args.size.split(',')[0])
        h=int(args.size.split(',')[1])
        dst_size=(w,h)
    else:
        print(f'disable resize')
        
    if args.use_opencv_match:
        print(f'Use OpenCV knnMatch')
    else:
        print(f'Use original match function')
    
    liftfeat=LiftFeat(weight=MODEL_PATH,detect_threshold=0.05)
    
    img1=cv2.imread(args.img1)
    img2=cv2.imread(args.img2)
    
    if args.size:
        img1=cv2.resize(img1,dst_size)
        img2=cv2.resize(img2,dst_size)
    
    if args.use_opencv_match:
        data1 = liftfeat.extract(img1)
        data2 = liftfeat.extract(img2)
        kpts1,descs1=data1['keypoints'].cpu().numpy(),data1['descriptors'].cpu().numpy()
        kpts2,descs2=data2['keypoints'].cpu().numpy(),data2['descriptors'].cpu().numpy()
        
        mkpts1,mkpts2 = opencv_knn_match(descs1,descs2,kpts1,kpts2)
    else:
        mkpts1,mkpts2=liftfeat.match_liftfeat(img1,img2)
    
    
    canvas=warp_corners_and_draw_matches(mkpts1,mkpts2,img1,img2)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=[12,12])
    plt.imshow(canvas[...,::-1])
    
    plt.savefig(os.path.join(os.path.dirname(__file__),'match.jpg'), dpi=300, bbox_inches='tight')
    
    plt.show()
    