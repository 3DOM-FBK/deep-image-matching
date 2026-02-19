from .matchers import DualSoftmaxMatcher, DenseMatcher, LightGlue
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import kornia

class RDD_helper(nn.Module):
    def __init__(self, RDD):
        super().__init__()
        self.matcher = DualSoftmaxMatcher(inv_temperature = 20, thr = 0.01)
        self.dense_matcher = DenseMatcher(inv_temperature=20, thr=0.01)
        self.RDD = RDD
        self.lg_matcher = None
        
    @torch.inference_mode()
    def match(self, img0, img1, thr=0.01, resize=None, top_k=4096):
        if top_k is not None and top_k != self.RDD.top_k:
            self.RDD.top_k = top_k
            self.RDD.set_softdetect(top_k=top_k)
        
        img0, scale0 = self.parse_input(img0, resize)
        img1, scale1 = self.parse_input(img1, resize)

        out0 = self.RDD.extract(img0)[0]
        out1 = self.RDD.extract(img1)[0]
        
        # get top_k confident matches
        mkpts0, mkpts1, conf = self.matcher(out0, out1, thr)
        
        scale0 = 1.0 / scale0
        scale1 = 1.0 / scale1
        
        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1
        
        return mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), conf.cpu().numpy()
    
    @torch.inference_mode()
    def match_lg(self, img0, img1, thr=0.01, resize=None, top_k=4096):
        if self.lg_matcher is None:
            lg_conf = {
                "name": "lightglue",  # just for interfacing
                "input_dim": 256,  # input descriptor dimension (autoselected from weights)
                "descriptor_dim": 256,
                "add_scale_ori": False,
                "n_layers": 9,
                "num_heads": 4,
                "flash": True,  # enable FlashAttention if available.
                "mp": False,  # enable mixed precision
                "filter_threshold": 0.01,  # match threshold
                "depth_confidence": -1,  # depth confidence threshold
                "width_confidence": -1,  # width confidence threshold
                "weights": './weights/RDD_lg-v2.pth',  # path to the weights
            }
            self.lg_matcher = LightGlue('rdd', **lg_conf).to(self.RDD.device)

        if top_k is not None and top_k != self.RDD.top_k:
            self.RDD.top_k = top_k
            self.RDD.set_softdetect(top_k=top_k)
            
        img0, scale0 = self.parse_input(img0, resize=resize)
        img1, scale1 = self.parse_input(img1, resize=resize)
        
        size0 = torch.tensor(img0.shape[-2:])[None]
        size1 = torch.tensor(img1.shape[-2:])[None]
        
        out0 = self.RDD.extract(img0)[0]
        out1 = self.RDD.extract(img1)[0]

        # get top_k confident matches
        image0_data = {
            'keypoints': out0['keypoints'][None],
            'descriptors': out0['descriptors'][None],
            'image_size': size0,
        }
        
        image1_data = {
            'keypoints': out1['keypoints'][None],
            'descriptors': out1['descriptors'][None],
            'image_size': size1,
        }
        
        pred = {}
        
        with torch.no_grad():
            pred.update({'image0': image0_data, 'image1': image1_data})
            pred.update(self.lg_matcher({**pred}))
        
        kpts0 = pred['image0']['keypoints'][0]
        kpts1 = pred['image1']['keypoints'][0]
        
        matches = pred['matches'][0]

        mkpts0 = kpts0[matches[... , 0]]
        mkpts1 = kpts1[matches[... , 1]]
        conf = pred['scores'][0]
        
        valid_mask = conf > thr
        mkpts0 = mkpts0[valid_mask]
        mkpts1 = mkpts1[valid_mask]
        conf = conf[valid_mask]
        
        scale0 = 1.0 / scale0
        scale1 = 1.0 / scale1
        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1
        
        return mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), conf.cpu().numpy()
    
    @torch.inference_mode()
    def match_dense(self, img0, img1, thr=0.01, resize=None, anchor='mnn'):
        
        img0, scale0 = self.parse_input(img0, resize=resize)
        img1, scale1 = self.parse_input(img1, resize=resize)

        out0 = self.RDD.extract_dense(img0)[0]
        out1 = self.RDD.extract_dense(img1)[0]
        
        # get top_k confident matches
        mkpts0, mkpts1, conf = self.dense_matcher(out0, out1, thr, err_thr=self.RDD.stride, anchor=anchor)
        
        scale0 = 1.0 / scale0
        scale1 = 1.0 / scale1
        
        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1
        
        return mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), conf.cpu().numpy()
        
    @torch.inference_mode()
    def match_3rd_party(self, img0, img1, model='aliked', resize=None, thr=0.01):
        img0, scale0 = self.parse_input(img0, resize=resize)
        img1, scale1 = self.parse_input(img1, resize=resize)

        out0 = self.RDD.extract_3rd_party(img0, model=model)[0]
        out1 = self.RDD.extract_3rd_party(img1, model=model)[0]
        
        mkpts0, mkpts1, conf = self.matcher(out0, out1, thr)
        
        scale0 = 1.0 / scale0
        scale1 = 1.0 / scale1
        
        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1
        
        return mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), conf.cpu().numpy()
    
    def parse_input(self, x, resize=None):
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0,3,1,2)/255
        
        h, w = x.shape[-2:]
        size = h, w
        
        if resize is not None:
            size = self.get_new_image_size(h, w, resize)
            x = kornia.geometry.transform.resize(
                x,
                size,
                side='long',
                antialias=True,
                align_corners=None,
                interpolation='bilinear',
            )
        scale = torch.Tensor([x.shape[-1] / w, x.shape[-2] / h]).to(self.RDD.device)
        
        return x, scale
    
    def get_new_image_size(self, h, w, resize=1600):
        aspect_ratio = w / h
        if h > w:
            size = (resize, int(resize * aspect_ratio))
        else:
            size = (int(resize / aspect_ratio), resize)

        size = list(map(lambda x: int(x // 32 * 32), size)) # make sure size is divisible by 32
        return size