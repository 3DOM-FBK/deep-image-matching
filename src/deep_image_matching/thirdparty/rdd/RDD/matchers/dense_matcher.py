import torch
import torch.nn as nn
import torch.nn.functional as F
import poselib
from .lightglue import LightGlue

class DenseMatcher(nn.Module):
    def __init__(self, inv_temperature = 20, thr = 0.01):
        super().__init__()
        self.inv_temperature = inv_temperature
        self.thr = thr
        self.lg_matcher = None

    def forward(self, info0, info1, thr = None, err_thr=4, min_num_inliers=30, anchor='mnn'):
        
        desc0 = info0['descriptors']
        desc1 = info1['descriptors']
        
        if anchor == 'mnn':
            inds, P = self.dual_softmax(desc0, desc1, thr=thr)
            mconf = P[inds[:,0], inds[:,1]]
        elif anchor == 'lightglue':
            # Use LightGlue's matching strategy
            inds, mconf = self.lightglue(info0, info1, device=info0['keypoints'].device)
        else:
            raise ValueError(f"Unknown anchor type: {anchor}. Use 'mnn' or 'lightglue'.")
        
        mkpts_0 = info0['keypoints'][inds[:,0]]
        mkpts_1 = info1['keypoints'][inds[:,1]]
        Fm, inliers = self.get_fundamental_matrix(mkpts_0, mkpts_1)
        
        if inliers.sum() >= min_num_inliers:
            desc1_dense = info0['descriptors_dense']
            desc2_dense = info1['descriptors_dense']

            inds_dense, P_dense = self.dual_softmax(desc1_dense, desc2_dense, thr=thr)
            
            mkpts_0_dense = info0['keypoints_dense'][inds_dense[:,0]]
            mkpts_1_dense = info1['keypoints_dense'][inds_dense[:,1]]
            mconf_dense = P_dense[inds_dense[:,0], inds_dense[:,1]]
            
            mkpts_0_dense, mkpts_1_dense, mconf_dense = self.refine_matches(mkpts_0_dense, mkpts_1_dense, mconf_dense, Fm, err_thr=err_thr)
            mkpts_0 = mkpts_0[inliers]
            mkpts_1 = mkpts_1[inliers]
            mconf = mconf[inliers]
            # concatenate the matches
            mkpts_0 = torch.cat([mkpts_0, mkpts_0_dense], dim=0)
            mkpts_1 = torch.cat([mkpts_1, mkpts_1_dense], dim=0)
            mconf = torch.cat([mconf, mconf_dense], dim=0)

        return mkpts_0, mkpts_1, mconf
    
    def lightglue(self, info0, info1, device='cpu'):
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
            self.lg_matcher = LightGlue('rdd', **lg_conf).to(device)
        
        image0_data = {
            'keypoints': info0['keypoints'][None],
            'descriptors': info0['descriptors'][None],
            'image_size': info0['image_size'],
        }
        
        image1_data = {
            'keypoints': info1['keypoints'][None],
            'descriptors': info1['descriptors'][None],
            'image_size': info1['image_size'],
        }
        
        pred = {}
        
        with torch.no_grad():
            pred.update({'image0': image0_data, 'image1': image1_data})
            pred.update(self.lg_matcher({**pred}))
            
        matches = pred['matches'][0]
        conf = pred['scores'][0]
        
        return matches, conf

    def get_fundamental_matrix(self, kpts_0, kpts_1):
        Fm, info = poselib.estimate_fundamental(kpts_0.cpu().numpy(), kpts_1.cpu().numpy(), {'max_epipolar_error': 1, 'progressive_sampling': True}, {})
        inliers = info['inliers']
        Fm = torch.tensor(Fm, device=kpts_0.device, dtype=kpts_0.dtype)
        inliers = torch.tensor(inliers, device=kpts_0.device, dtype=torch.bool)
        return Fm, inliers    
    
    def dual_softmax(self, desc0, desc1, thr = None):
        if thr is None:
            thr = self.thr
        dist_mat = (desc0 @ desc1.t()) * self.inv_temperature
        P = dist_mat.softmax(dim = -2) * dist_mat.softmax(dim= -1)
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P >= thr))
        
        return inds, P
    
    @torch.inference_mode()
    def refine_matches(self, mkpts_0, mkpts_1, mconf, Fm, err_thr=4):    
        mkpts_0_h = torch.cat([mkpts_0, torch.ones(mkpts_0.shape[0], 1, device=mkpts_0.device)], dim=1)  # (N, 3)
        mkpts_1_h = torch.cat([mkpts_1, torch.ones(mkpts_1.shape[0], 1, device=mkpts_1.device)], dim=1)  # (N, 3)
        
        lines_1 = torch.matmul(Fm, mkpts_0_h.T).T
        
        a, b, c = lines_1[:, 0], lines_1[:, 1], lines_1[:, 2]  

        x1, y1 = mkpts_1[:, 0], mkpts_1[:, 1]
        
        denom = a**2 + b**2 + 1e-8  
        
        x_offset = (b * (b * x1 - a * y1) - a * c) / denom - x1
        y_offset = (a * (a * y1 - b * x1) - b * c) / denom - y1

        inds = (x_offset.abs() < err_thr) | (y_offset.abs() < err_thr)

        x_offset = x_offset[inds]
        y_offset = y_offset[inds]

        mkpts_0 = mkpts_0[inds]
        mkpts_1 = mkpts_1[inds]
        
        refined_mkpts_1 = mkpts_1 + torch.stack([x_offset, y_offset], dim=1)
  
        return mkpts_0, refined_mkpts_1, mconf[inds]
