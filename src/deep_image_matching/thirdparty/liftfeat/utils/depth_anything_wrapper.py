import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import sys

sys.path.append("/home/yepeng_liu/code_python/third_repos/Depth-Anything-V2")
from depth_anything_v2.dpt_opt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

import time

VITS_MODEL_PATH = "/home/yepeng_liu/code_python/third_repos/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
VITB_MODEL_PATH = "/home/yepeng_liu/code_python/third_repos/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"
VITL_MODEL_PATH = "/home/yepeng_liu/code_python/third_repos/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"

model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

class DepthAnythingExtractor(nn.Module):
    def __init__(self, encoder_type, device, input_size, process_size=(608,800)):
        super().__init__()
        self.net = DepthAnythingV2(**model_configs[encoder_type])
        self.device = device
        if encoder_type == "vits":
            print(f"loading {VITS_MODEL_PATH}")
            self.net.load_state_dict(torch.load(VITS_MODEL_PATH, map_location="cpu"))
        elif encoder_type == "vitb":
            print(f"loading {VITB_MODEL_PATH}")
            self.net.load_state_dict(torch.load(VITB_MODEL_PATH, map_location="cpu"))
        elif encoder_type == "vitl":
            print(f"loading {VITL_MODEL_PATH}")
            self.net.load_state_dict(torch.load(VITL_MODEL_PATH, map_location="cpu"))
        else:
            raise RuntimeError("unsupport encoder type")
        self.net.to(self.device).eval()
        self.tranform = Compose([
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        self.process_size=process_size
        self.input_size=input_size
        
    @torch.inference_mode()
    def infer_image(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        img = self.tranform({'image': img})['image']
        
        img = torch.from_numpy(img).unsqueeze(0)
        
        img = img.to(self.device)
        
        with torch.no_grad():
            depth = self.net.forward(img)
        
        depth = F.interpolate(depth[:, None], self.process_size, mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    @torch.inference_mode()
    def compute_normal_map_torch(self, depth_map, scale=1.0):
        """
        通过深度图计算法向量 (PyTorch 实现)

        参数：
            depth_map (torch.Tensor): 深度图，形状为 (H, W)
            scale (float): 深度值的比例因子，用于调整深度图中的梯度计算

        返回：
            torch.Tensor: 法向量图，形状为 (H, W, 3)
        """
        if depth_map.ndim != 2:
            raise ValueError("输入 depth_map 必须是二维张量。")
        
        # 计算深度图的梯度
        dzdx = torch.diff(depth_map, dim=1, append=depth_map[:, -1:]) * scale
        dzdy = torch.diff(depth_map, dim=0, append=depth_map[-1:, :]) * scale

        # 初始化法向量图
        H, W = depth_map.shape
        normal_map = torch.zeros((H, W, 3), dtype=depth_map.dtype, device=depth_map.device)
        normal_map[:, :, 0] = -dzdx  # x 分量
        normal_map[:, :, 1] = -dzdy  # y 分量
        normal_map[:, :, 2] = 1.0    # z 分量

        # 归一化法向量
        norm = torch.linalg.norm(normal_map, dim=2, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1.0, device=depth_map.device), norm)  # 避免除以零
        normal_map /= norm

        return normal_map

    @torch.inference_mode()
    def extract(self, img):
        depth = self.infer_image(img)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_t=torch.from_numpy(depth).float().to(self.device)
        normal_map = self.compute_normal_map_torch(depth_t,1.0)
        return depth_t,normal_map
    
    
if __name__=="__main__":
    img_path=os.path.join(os.path.dirname(__file__),'../assert/ref.jpg')
    img=cv2.imread(img_path)
    img=cv2.resize(img,(800,608))
    import pdb;pdb.set_trace()
    DAExtractor=DepthAnythingExtractor('vitb',torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),256)
    depth_t,norm=DAExtractor.extract(img)
    norm=norm.cpu().numpy()
    norm=(norm+1)/2*255
    norm=norm.astype(np.uint8)
    cv2.imwrite(os.path.join(os.path.dirname(__file__),"norm.png"),norm)
    start=time.perf_counter()
    for i in range(20):
        depth_t,norm=DAExtractor.extract(img)
    end=time.perf_counter()
    print(f"cost {end-start} seconds")
    