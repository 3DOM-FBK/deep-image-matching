
"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
"""

import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F

import tqdm
import math
import cv2

import sys
sys.path.append('/home/yepeng_liu/code_python/laiwenpeng/LiftFeat')
from ..utils.featurebooster import FeatureBooster
from ..utils.config import featureboost_config

# from models.model_dfb import LiftFeatModel
# from models.interpolator import InterpolateSparse2d
# from third_party.config import featureboost_config

"""
foundational functions
"""
def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        radius: an integer scalar, the radius of the NMS window.
    """

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    indices = torch.multinomial(scores, k, replacement=False)
    return keypoints[indices], scores[indices]


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2 * radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
        scores[:, None], width, 1, radius, divisor_override=1
    )
    ar = torch.arange(-radius, radius + 1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
        scores[:, None], kernel_x.transpose(2, 3), padding=radius
    )
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints


# Legacy (broken) sampling of the descriptors
def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 定义特征提取层，减少通道数同时增加特征提取能力
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        # 使用BN层
        self.bn = nn.BatchNorm2d(in_channels//2)
        # 使用LeakyReLU激活函数
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.leaky_relu(self.bn(self.conv(x)))

        return x


class KeypointHead(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer1=BaseLayer(in_channels,32)
        self.layer2=BaseLayer(32,32)
        self.layer3=BaseLayer(32,64)
        self.layer4=BaseLayer(64,64)
        self.layer5=BaseLayer(64,128)
        
        self.conv=nn.Conv2d(128,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(65)
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.bn(self.conv(x))
        return x
    
    
class DescriptorHead(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer=nn.Sequential(
            BaseLayer(in_channels,32),
            BaseLayer(32,32,activation=False),
            BaseLayer(32,64,activation=False),
            BaseLayer(64,out_channels,activation=False)
        )
        
    def forward(self,x):
        x=self.layer(x)
        # x=nn.functional.softmax(x,dim=1)
        return x
    
    
class HeatmapHead(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super().__init__()
        self.convHa = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bnHa = nn.BatchNorm2d(mid_channels)
        self.convHb = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bnHb = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self,x):
        x = self.leaky_relu(self.bnHa(self.convHa(x)))
        x = self.leaky_relu(self.bnHb(self.convHb(x)))
        
        x = torch.sigmoid(x)
        return x
        

class DepthHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsampleDa = UpsampleLayer(in_channels)
        self.upsampleDb = UpsampleLayer(in_channels//2)
        self.upsampleDc = UpsampleLayer(in_channels//4)
        
        self.convDepa = nn.Conv2d(in_channels//2+in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.bnDepa = nn.BatchNorm2d(in_channels//2)
        self.convDepb = nn.Conv2d(in_channels//4+in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1)
        self.bnDepb = nn.BatchNorm2d(in_channels//4)
        self.convDepc = nn.Conv2d(in_channels//8+in_channels//4, 3, kernel_size=3, stride=1, padding=1)
        self.bnDepc = nn.BatchNorm2d(3)
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x0 = F.interpolate(x, scale_factor=2,mode='bilinear',align_corners=False)
        x1 = self.upsampleDa(x)
        x1 = torch.cat([x0,x1],dim=1)
        x1 = self.leaky_relu(self.bnDepa(self.convDepa(x1)))
        
        x1_0 = F.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=False)
        x2 = self.upsampleDb(x1)
        x2 = torch.cat([x1_0,x2],dim=1)
        x2 = self.leaky_relu(self.bnDepb(self.convDepb(x2)))
        
        x2_0 = F.interpolate(x2,scale_factor=2,mode='bilinear',align_corners=False)
        x3 = self.upsampleDc(x2)
        x3 = torch.cat([x2_0,x3],dim=1)
        x = self.leaky_relu(self.bnDepc(self.convDepc(x3)))
        
        x = F.normalize(x,p=2,dim=1)
        return x
    

class BaseLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False,activation=True):
        super().__init__()
        if activation:
            self.layer=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
                nn.BatchNorm2d(out_channels,affine=False),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
                nn.BatchNorm2d(out_channels,affine=False)
            )
        
    def forward(self,x):
        return self.layer(x)


class LiftFeatSPModel(nn.Module):
    default_conf = {
        "has_detector": True,
        "has_descriptor": True,
        "descriptor_dim": 64,
        # Inference
        "sparse_outputs": True,
        "dense_outputs": False,
        "nms_radius": 4,
        "refinement_radius": 0,
        "detection_threshold": 0.005,
        "max_num_keypoints": -1,
        "max_num_keypoints_val": None,
        "force_num_keypoints": False,
        "randomize_keypoints_training": False,
        "remove_borders": 4,
        "legacy_sampling": True,  # True to use the old broken sampling
    }

    def __init__(self, featureboost_config, use_kenc=False, use_normal=True, use_cross=True):
        super().__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.descriptor_dim = 64
        
        self.norm = nn.InstanceNorm2d(1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1,c2,c3,c4,c5 = 24,24,64,64,128

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)
        
        self.upsample4 = UpsampleLayer(c4)
        self.upsample5 = UpsampleLayer(c5)
        self.conv_fusion45 = nn.Conv2d(c5//2+c4,c4,kernel_size=3,stride=1,padding=1)
        self.conv_fusion34 = nn.Conv2d(c4//2+c3,c3,kernel_size=3,stride=1,padding=1)

        # detector
        self.keypoint_head = KeypointHead(in_channels=c3,out_channels=65)
        # descriptor
        self.descriptor_head = DescriptorHead(in_channels=c3,out_channels=self.descriptor_dim)
        # # heatmap
        # self.heatmap_head = HeatmapHead(in_channels=c3,mid_channels=c3,out_channels=1)
        # depth
        self.depth_head = DepthHead(c3)
        
        self.fine_matcher =  nn.Sequential(
                                            nn.Linear(128, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 64),
                                        )
        
        # feature_booster
        self.feature_boost = FeatureBooster(featureboost_config, use_kenc=use_kenc, use_cross=use_cross, use_normal=use_normal)
        
    def feature_extract(self, x):
        x1 = self.relu(self.conv1a(x))
        x1 = self.relu(self.conv1b(x1))
        x1 = self.pool(x1)
        x2 = self.relu(self.conv2a(x1))
        x2 = self.relu(self.conv2b(x2))
        x2 = self.pool(x2)
        x3 = self.relu(self.conv3a(x2))
        x3 = self.relu(self.conv3b(x3))
        x3 = self.pool(x3)
        x4 = self.relu(self.conv4a(x3))
        x4 = self.relu(self.conv4b(x4))
        x4 = self.pool(x4)
        x5 = self.relu(self.conv5a(x4))
        x5 = self.relu(self.conv5b(x5))
        x5 = self.pool(x5)
        return x3,x4,x5
    
    def fuse_multi_features(self,x3,x4,x5):
        # upsample x5 feature
        x5 = self.upsample5(x5)
        x4 = torch.cat([x4,x5],dim=1)
        x4 = self.conv_fusion45(x4)
        
        # upsample x4 feature
        x4 = self.upsample4(x4)
        x3 = torch.cat([x3,x4],dim=1)
        x = self.conv_fusion34(x3)
        return x
    
    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws).reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


    def forward1(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        with torch.no_grad():
            x = x.mean(dim=1, keepdim = True)
            x = self.norm(x)
        
        x3,x4,x5 = self.feature_extract(x)
        
        # features fusion
        x = self.fuse_multi_features(x3,x4,x5)
        
        # keypoint 
        keypoint_map = self.keypoint_head(x)
        # descriptor
        des_map = self.descriptor_head(x)
        # # heatmap
        # heatmap = self.heatmap_head(x)
        
        # import pdb;pdb.set_trace()
        # depth
        d_feats = self.depth_head(x)
       
        return des_map, keypoint_map, d_feats
        # return des_map, keypoint_map, heatmap, d_feats
    
    def forward2(self, descs, kpts, normals):
        # import pdb;pdb.set_trace()
        normals_feat=self._unfold2d(normals, ws=8)
        normals_v=normals_feat.squeeze(0).permute(1,2,0).reshape(-1,normals_feat.shape[1])
        descs_v=descs.squeeze(0).permute(1,2,0).reshape(-1,descs.shape[1])
        kpts_v=kpts.squeeze(0).permute(1,2,0).reshape(-1,kpts.shape[1])
        descs_refine = self.feature_boost(descs_v, kpts_v, normals_v)
        return descs_refine
    
    def forward(self,x):
        M1,K1,D1=self.forward1(x)
        descs_refine=self.forward2(M1,K1,D1)
        return descs_refine,M1,K1,D1
    

if __name__ == "__main__":
    img_path=os.path.join(os.path.dirname(__file__),'../assert/ref.jpg')
    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(800,608))
    import pdb;pdb.set_trace()
    img=torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()/255.0
    img=img.cuda() if torch.cuda.is_available() else img
    liftfeat_sp=LiftFeatSPModel(featureboost_config).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    des_map, keypoint_map, d_feats=liftfeat_sp.forward1(img)
    des_fine=liftfeat_sp.forward2(des_map,keypoint_map,d_feats)
    print(des_map.shape)
    