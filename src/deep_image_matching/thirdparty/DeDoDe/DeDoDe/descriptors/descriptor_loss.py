import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from DeDoDe.utils import *
import DeDoDe

class DescriptorLoss(nn.Module):
    
    def __init__(self, detector, num_keypoints = 5000, normalize_descriptions = False, inv_temp = 1, device = get_best_device()) -> None:
        super().__init__()
        self.detector = detector
        self.tracked_metrics = {}
        self.num_keypoints = num_keypoints
        self.normalize_descriptions = normalize_descriptions
        self.inv_temp = inv_temp
    
    def warp_from_depth(self, batch, kpts_A, kpts_B):
        mask_A_to_B, kpts_A_to_B = warp_kpts(kpts_A, 
                    batch["im_A_depth"],
                    batch["im_B_depth"],
                    batch["T_1to2"],
                    batch["K1"],
                    batch["K2"],)
        mask_B_to_A, kpts_B_to_A = warp_kpts(kpts_B, 
                    batch["im_B_depth"],
                    batch["im_A_depth"],
                    batch["T_1to2"].inverse(),
                    batch["K2"],
                    batch["K1"],)
        return (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A)
    
    def warp_from_homog(self, batch, kpts_A, kpts_B):
        kpts_A_to_B = homog_transform(batch["Homog_A_to_B"], kpts_A)
        kpts_B_to_A = homog_transform(batch["Homog_A_to_B"].inverse(), kpts_B)
        return (None, kpts_A_to_B), (None, kpts_B_to_A)

    def supervised_loss(self, outputs, batch):
        kpts_A, kpts_B = self.detector.detect(batch, num_keypoints = self.num_keypoints)['keypoints'].clone().chunk(2)
        desc_grid_A, desc_grid_B = outputs["description_grid"].chunk(2)
        desc_A = F.grid_sample(desc_grid_A.float(), kpts_A[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT
        desc_B = F.grid_sample(desc_grid_B.float(), kpts_B[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT
        if "im_A_depth" in batch:
            (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = self.warp_from_depth(batch, kpts_A, kpts_B)
        elif "Homog_A_to_B" in batch:
            (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = self.warp_from_homog(batch, kpts_A, kpts_B)
            
        with torch.no_grad():
            D_B = torch.cdist(kpts_A_to_B, kpts_B)
            D_A = torch.cdist(kpts_A, kpts_B_to_A)
            inds = torch.nonzero((D_B == D_B.min(dim=-1, keepdim = True).values) 
                                 * (D_A == D_A.min(dim=-2, keepdim = True).values)
                                 * (D_B < 0.01)
                                 * (D_A < 0.01))
            
        logP_A_B = dual_log_softmax_matcher(desc_A, desc_B, 
                                            normalize = self.normalize_descriptions,
                                            inv_temperature = self.inv_temp)
        neg_log_likelihood = -logP_A_B[inds[:,0], inds[:,1], inds[:,2]].mean()
        self.tracked_metrics["neg_log_likelihood"] = (0.99 * self.tracked_metrics.get("neg_log_likelihood", neg_log_likelihood.detach().item()) + 0.01 * neg_log_likelihood.detach().item())
        if np.random.rand() > 0.99:
            print(self.tracked_metrics["neg_log_likelihood"])
        return neg_log_likelihood
    
    def forward(self, outputs, batch):
        losses = self.supervised_loss(outputs, batch)
        return losses