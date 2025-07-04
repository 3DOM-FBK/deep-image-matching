import cv2 as cv
import numpy as np
import argparse
import sys
import scipy.io as sio
import torch
import time
import kornia.feature as KF
from .adalam import AdalamFilter
from PIL import Image


def extract_keypoints(impath):
    im = cv.imread(impath, cv.IMREAD_COLOR)
    d = cv.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)
    kp1, desc1 = d.detectAndCompute(im, mask=np.ones(shape=im.shape[:-1] + (1,),
                                                              dtype=np.uint8))
    pts = np.array([k.pt for k in kp1], dtype=np.float32)
    ors = np.array([k.angle for k in kp1], dtype=np.float32)
    scs = np.array([k.size for k in kp1], dtype=np.float32)
    return pts, ors, scs, desc1, im


def show_matches(img1, img2, k1, k2, target_dim=800.):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2
        current_width = w1 + w2 * scale_to_align
        scale_to_fit = target_height / h1
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]

    target_1, target_2, scale1, scale2, offset = resize_horizontal(h1, w1, h2, w2, target_dim)

    im1 = cv.resize(img1, target_1, interpolation=cv.INTER_AREA)
    im2 = cv.resize(img2, target_2, interpolation=cv.INTER_AREA)

    h1, w1 = target_1[::-1]
    h2, w2 = target_2[::-1]

    vis = np.ones((max(h1, h2), w1 + w2, 3), np.uint8) * 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:w1 + w2] = im2

    p1 = [np.int32(k * scale1) for k in k1]
    p2 = [np.int32(k * scale2 + offset) for k in k2]

    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv.line(vis, (x1, y1), (x2, y2), [0, 255, 0], 1)

    cv.imshow("AdaLAM example", vis)
    cv.waitKey()


class adalam_module:
    def __init__(self, **args):
        self.matcher = AdalamFilter()
        self.th = self.matcher.config['th']        
        self.orientation_difference_threshold = self.matcher.config['orientation_difference_threshold']
        self.scale_rate_threshold = self.matcher.config['scale_rate_threshold']
            
        for k, v in args.items():
           setattr(self, k, v)
           self.matcher.config[k] = v
        
        
    def get_id(self):
        return ('adalam_th_' + str(self.th) + '_scale_th_' + str(self.scale_rate_threshold) + '_ori_th_' + str(self.orientation_difference_threshold)).lower()
    

    def run(self, **args):                     
        if torch.is_tensor(args['pt1']) and (args['pt1'].shape[0] >= 3):           
            k1 = np.ascontiguousarray(args['pt1'].detach().cpu())
            k2 = np.ascontiguousarray(args['pt2'].detach().cpu())
                        
            o1 = torch.zeros(0)
            o2 = torch.zeros(0)
            s1 = torch.zeros(0)
            s2 = torch.zeros(0)
            self.matcher.config['orientation_difference_threshold'] = None
            self.matcher.config['scale_rate_threshold'] = None            

            if 'kp1' in args.keys():
                self.matcher.config['orientation_difference_threshold'] = self.orientation_difference_threshold
                self.matcher.config['scale_rate_threshold'] = self.scale_rate_threshold                
                o1 = KF.get_laf_orientation(args['kp1'].unsqueeze(0)).squeeze().detach().cpu()
                s1 = KF.get_laf_scale(args['kp1'].unsqueeze(0)).squeeze().detach().cpu()
                o2 = KF.get_laf_orientation(args['kp2'].unsqueeze(0)).squeeze().detach().cpu()
                s2 = KF.get_laf_scale(args['kp2'].unsqueeze(0)).squeeze().detach().cpu()

            sz1 = Image.open(args['im1']).size
            sz2 = Image.open(args['im2']).size
            
            l = args['pt1'].shape[0]            
            putative_matches = np.arange(l, dtype=np.int64)
            if 'val' in args.keys():
                scores = np.ascontiguousarray(args['val'].squeeze().detach().cpu())
            else:
                scores = np.zeros(l)
            mnn = np.zeros(l, dtype=bool)                    
    
            idx = self.matcher.match_and_filter(k1=k1, k2=k2, o1=o1, o2=o2, d1=None, d2=None, s1=s1, s2=s2, im1shape=sz1, im2shape=sz2, putative_matches=putative_matches, scores=scores, mnn=mnn)

            mask = np.zeros(l, dtype=bool)                    
            mask[idx[:,0]] = True
    
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
        else:            
            mask = []
            pt1 = args['pt1']
            pt2 = args['pt2']     
            Hs = args['Hs']
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
