import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from copy import deepcopy
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import json

import scipy.io as scio
import poselib

import argparse
import datetime

parser=argparse.ArgumentParser(description='MegaDepth dataset evaluation script')
parser.add_argument('--name',type=str,default='LiftFeat',help='experiment name')
parser.add_argument('--gpu',type=str,default='0',help='GPU ID')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from models.liftfeat_wrapper import LiftFeat
from evaluation.eval_utils import *

from torch.utils.data import Dataset,DataLoader

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

DATASET_ROOT = os.path.join(os.path.dirname(__file__),'../data/megadepth_test_1500')
DATASET_JSON = os.path.join(os.path.dirname(__file__),'../data/megadepth_1500.json')

class MegaDepth1500(Dataset):
    """
        Streamlined MegaDepth-1500 dataloader. The camera poses & metadata are stored in a formatted json for facilitating 
        the download of the dataset and to keep the setup as simple as possible.
    """
    def __init__(self, json_file, root_dir):
        # Load the info & calibration from the JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! \n \
              > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = deepcopy(self.data[idx])

        h1, w1 = data['size0_hw']
        h2, w2 = data['size1_hw']

        # Here we resize the images to max_dim = 1200, as described in the paper, and adjust the image such that it is divisible by 32
        # following the protocol of the LoFTR's Dataloader (intrinsics are corrected accordingly). 
        # For adapting this with different resolution, you would need to re-scale intrinsics below.
        image0 = cv2.resize(cv2.imread(f"{self.root_dir}/{data['pair_names'][0]}"),(w1, h1))

        image1 = cv2.resize(cv2.imread(f"{self.root_dir}/{data['pair_names'][1]}"),(w2, h2))

        data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1)
        data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1)

        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data

if __name__ == "__main__":
    weights=os.path.join(os.path.dirname(__file__),'../weights/LiftFeat.pth')
    liftfeat=LiftFeat(weight=weights)
    
    dataset = MegaDepth1500(json_file = DATASET_JSON, root_dir = DATASET_ROOT)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics = {}
    R_errs = []
    t_errs = []
    inliers = []
    
    results=[]

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for d in tqdm.tqdm(loader, desc="processing"):
        error_infos = compute_pose_error(liftfeat.match_liftfeat,d)
        results.append(error_infos)
            
    print(f'\n==={cur_time}==={args.name}===')
    d_err_auc,errors=compute_maa(results)
    for s_k,s_v in d_err_auc.items():
        print(f'{s_k}: {s_v*100}')
