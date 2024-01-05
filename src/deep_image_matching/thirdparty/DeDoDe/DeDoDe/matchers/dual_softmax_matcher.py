import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np
from DeDoDe.utils import dual_softmax_matcher, to_pixel_coords, to_normalized_coords

class DualSoftMaxMatcher(nn.Module):        
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = dual_softmax_matcher(descriptions_A, descriptions_B, 
                                 normalize = normalize, inv_temperature=inv_temp,
                                 )
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))
        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]
        return matches_A, matches_B, batch_inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)