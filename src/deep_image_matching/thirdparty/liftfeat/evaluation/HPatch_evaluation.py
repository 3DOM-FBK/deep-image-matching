import cv2
import os
from tqdm import tqdm
import torch
import numpy as np
import sys
import poselib

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import argparse
import datetime

parser=argparse.ArgumentParser(description='HPatch dataset evaluation script')
parser.add_argument('--name',type=str,default='LiftFeat',help='experiment name')
parser.add_argument('--gpu',type=str,default='0',help='GPU ID')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

top_k = None
n_i = 52
n_v = 56

DATASET_ROOT = os.path.join(os.path.dirname(__file__),'../data/HPatch')

from evaluation.eval_utils import *
from models.liftfeat_wrapper import LiftFeat


poselib_config = {"ransac_th": 3.0, "options": {}}

class PoseLibHomographyEstimator:
    def __init__(self, conf):
        self.conf = conf

    def estimate(self, mkpts0,mkpts1):
        M, info = poselib.estimate_homography(
            mkpts0,
            mkpts1,
            {
                "max_reproj_error": self.conf["ransac_th"],
                **self.conf["options"],
            },
        )
        success = M is not None
        if not success:
            M = np.eye(3,dtype=np.float32)
            inl = np.zeros(mkpts0.shape[0],dtype=np.bool_)
        else:
            inl = info["inliers"]

        estimation = {
            "success": success,
            "M_0to1": M,
            "inliers": inl,
        }

        return estimation
    
    
estimator=PoseLibHomographyEstimator(poselib_config)


def poselib_homography_estimate(mkpts0,mkpts1):
    data=estimator.estimate(mkpts0,mkpts1)
    return data


def generate_standard_image(img,target_size=(1920,1080)):
    sh,sw=img.shape[0],img.shape[1]
    rh,rw=float(target_size[1])/float(sh),float(target_size[0])/float(sw)
    ratio=min(rh,rw)
    nh,nw=int(ratio*sh),int(ratio*sw)
    ph,pw=target_size[1]-nh,target_size[0]-nw
    nimg=cv2.resize(img,(nw,nh))
    nimg=cv2.copyMakeBorder(nimg,0,ph,0,pw,cv2.BORDER_CONSTANT,value=(0,0,0))
    
    return nimg,ratio,ph,pw
    

def benchmark_features(match_fn):
    lim = [1, 9]
    rng = np.arange(lim[0], lim[1] + 1)

    seq_names = sorted(os.listdir(DATASET_ROOT))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    i_err_homo = {thr: 0 for thr in rng}
    v_err_homo = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        # load reference image
        ref_img = cv2.imread(os.path.join(DATASET_ROOT, seq_name, "1.ppm"))
        ref_img_shape=ref_img.shape

        # load query images
        for im_idx in range(2, 7):
            # read ground-truth homography
            homography = np.loadtxt(os.path.join(DATASET_ROOT, seq_name, "H_1_" + str(im_idx)))
            query_img = cv2.imread(os.path.join(DATASET_ROOT, seq_name, f"{im_idx}.ppm"))
            
            mkpts_a,mkpts_b=match_fn(ref_img,query_img)

            pos_a = mkpts_a
            pos_a_h = np.concatenate([pos_a, np.ones([pos_a.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]

            pos_b = mkpts_b

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(pos_a.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == "i":
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

            # estimate homography
            gt_homo = homography
            pred_homo, _ = cv2.findHomography(mkpts_a,mkpts_b,cv2.USAC_MAGSAC)
            if pred_homo is None:
                homo_dist = np.array([float("inf")])
            else:
                corners = np.array(
                    [
                        [0, 0],
                        [ref_img_shape[1] - 1, 0],
                        [0, ref_img_shape[0] - 1],
                        [ref_img_shape[1] - 1, ref_img_shape[0] - 1],
                    ]
                )
                real_warped_corners = homo_trans(corners, gt_homo)
                warped_corners = homo_trans(corners, pred_homo)
                homo_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

            for thr in rng:
                if seq_name[0] == "i":
                    i_err_homo[thr] += np.mean(homo_dist <= thr)
                else:
                    v_err_homo[thr] += np.mean(homo_dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, i_err_homo, v_err_homo, [seq_type, n_feats, n_matches]


if __name__ == "__main__":
    errors = {}
    
    weights=os.path.join(os.path.dirname(__file__),'../weights/LiftFeat.pth')
    liftfeat=LiftFeat(weight=weights)

    errors = benchmark_features(liftfeat.match_liftfeat)

    i_err, v_err, i_err_hom, v_err_hom, _ = errors
    
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f'\n==={cur_time}==={args.name}===')
    print(f"MHA@3 MHA@5 MHA@7")
    for thr in [3, 5, 7]:
        ill_err_hom = i_err_hom[thr] / (n_i * 5)
        view_err_hom = v_err_hom[thr] / (n_v * 5)
        print(f"{ill_err_hom * 100:.2f}%-{view_err_hom * 100:.2f}%")
