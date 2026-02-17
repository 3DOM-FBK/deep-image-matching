"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
"""


import sys
import os

ALIKE_PATH = '/home/yepeng_liu/code_python/multimodal_remote/ALIKE'
sys.path.append(ALIKE_PATH)

import torch
import torch.nn as nn
from alike import ALike
import cv2
import numpy as np

import pdb

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-l.pth')},
}


class ALikeExtractor(nn.Module):
    def __init__(self,model_type,device) -> None:
        super().__init__()
        self.net=ALike(**configs[model_type],device=device,top_k=4096,scores_th=0.1,n_limit=8000)
        
    
    @torch.inference_mode()
    def extract_alike_kpts(self,img):
        pred0=self.net(img,sub_pixel=True)
        return pred0['keypoints']


