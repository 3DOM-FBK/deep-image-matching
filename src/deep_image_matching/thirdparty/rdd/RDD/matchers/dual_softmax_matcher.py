import torch
import torch.nn as nn
import torch.nn.functional as F

class DualSoftmaxMatcher(nn.Module):
    def __init__(self, inv_temperature = 20, thr = 0.01):
        super().__init__()
        self.inv_temperature = inv_temperature
        self.thr = thr

    def forward(self, info0, info1, thr = None):
        desc0 = info0['descriptors']
        desc1 = info1['descriptors']
        
        inds, P = self.dual_softmax(desc0, desc1, thr)
        mkpts0 = info0['keypoints'][inds[:,0]]
        mkpts1 = info1['keypoints'][inds[:,1]]
        mconf = P[inds[:,0], inds[:,1]]
        
        return mkpts0, mkpts1, mconf

    def dual_softmax(self, desc0, desc1, thr = None):
        if thr is None:
            thr = self.thr
        dist_mat = (desc0 @ desc1.t()) * self.inv_temperature
        P = dist_mat.softmax(dim = -2) * dist_mat.softmax(dim= -1)
        
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P >= thr))
        
        return inds, P