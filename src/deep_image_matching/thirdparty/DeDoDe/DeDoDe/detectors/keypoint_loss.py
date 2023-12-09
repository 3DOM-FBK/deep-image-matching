import torch
import torch.nn as nn
import math

from DeDoDe.utils import *
import DeDoDe

class KeyPointLoss(nn.Module):
    
    def __init__(self, smoothing_size = 1, use_max_logit = False, entropy_target = 80, 
                 num_matches = 1024, jacobian_density_adjustment = False,
                 matchability_weight = 1, device = "cuda") -> None:
        super().__init__()
        X = torch.linspace(-1,1,smoothing_size, device = device)
        G = (-X**2 / (2 *1/2**2)).exp()
        G = G/G.sum()
        self.use_max_logit = use_max_logit
        self.entropy_target = entropy_target
        self.smoothing_kernel = G[None, None, None,:]
        self.smoothing_size = smoothing_size
        self.tracked_metrics = {}
        self.center = None
        self.num_matches = num_matches
        self.jacobian_density_adjustment = jacobian_density_adjustment
        self.matchability_weight = matchability_weight
        
    def compute_consistency(self, logits_A, logits_B_to_A, mask = None):
        
        masked_logits_A = torch.full_like(logits_A, -torch.inf)
        masked_logits_A[mask] = logits_A[mask]

        masked_logits_B_to_A = torch.full_like(logits_B_to_A, -torch.inf)
        masked_logits_B_to_A[mask] = logits_B_to_A[mask]

        log_p_A = masked_logits_A.log_softmax(dim=-1)[mask]
        log_p_B_to_A = masked_logits_B_to_A.log_softmax(dim=-1)[mask]

        return self.compute_jensen_shannon_div(log_p_A, log_p_B_to_A)
    
    def compute_joint_neg_log_likelihood(self, logits_A, logits_B_to_A, detections_A = None, detections_B_to_A = None, mask = None, device = "cuda", dtype = torch.float32, num_matches = None):
        B, K, HW = logits_A.shape
        logits_A, logits_B_to_A = logits_A.to(dtype), logits_B_to_A.to(dtype)
        mask = mask[:,None].expand(B, K, HW).reshape(B, K*HW)
        log_p_B_to_A = self.masked_log_softmax(logits_B_to_A.reshape(B,K*HW), mask = mask)
        log_p_A = self.masked_log_softmax(logits_A.reshape(B,K*HW), mask = mask)
        log_p = log_p_A + log_p_B_to_A
        if detections_A is None:
            detections_A = torch.zeros_like(log_p_A)
        if detections_B_to_A is None:
            detections_B_to_A = torch.zeros_like(log_p_B_to_A)
        detections_A = detections_A.reshape(B, HW)
        detections_A[~mask] = 0         
        detections_B_to_A = detections_B_to_A.reshape(B, HW)
        detections_B_to_A[~mask] = 0
        log_p_target = log_p.detach() + 50*detections_A + 50*detections_B_to_A
        num_matches = self.num_matches if num_matches is None else num_matches
        best_k = -(-log_p_target).flatten().kthvalue(k = B * num_matches, dim=-1).values
        p_target = (log_p_target > best_k[..., None]).float().reshape(B,K*HW)/num_matches
        return self.compute_cross_entropy(log_p_A[mask], p_target[mask]) + self.compute_cross_entropy(log_p_B_to_A[mask], p_target[mask])
                
    def compute_jensen_shannon_div(self, log_p, log_q):
        return 1/2 * (self.compute_kl_div(log_p, log_q) + self.compute_kl_div(log_q, log_p))
    
    def compute_kl_div(self, log_p, log_q):
        return (log_p.exp()*(log_p-log_q)).sum(dim=-1)
    
    def masked_log_softmax(self, logits, mask):
        masked_logits = torch.full_like(logits, -torch.inf)
        masked_logits[mask] = logits[mask]
        log_p = masked_logits.log_softmax(dim=-1)
        return log_p
    
    def masked_softmax(self, logits, mask):
        masked_logits = torch.full_like(logits, -torch.inf)
        masked_logits[mask] = logits[mask]
        log_p = masked_logits.softmax(dim=-1)
        return log_p
    
    def compute_detection_img(self, detections, mask, B, H, W, device = "cuda"):
        kernel_size = 5
        X = torch.linspace(-2,2,kernel_size, device = device)
        G = (-X**2 / (2 * (1/2)**2)).exp() # half pixel std
        G = G/G.sum()
        det_smoothing_kernel = G[None, None, None,:]
        det_img = torch.zeros((B,1,H,W), device = device)
        for b in range(B):
            valid_detections = (detections[b][mask[b]]).int()
            det_img[b,0][valid_detections[:,1], valid_detections[:,0]] = 1
        det_img = F.conv2d(det_img, weight = det_smoothing_kernel, padding = (kernel_size//2, 0))
        det_img = F.conv2d(det_img, weight = det_smoothing_kernel.mT, padding = (0, kernel_size//2))
        return det_img

    def compute_cross_entropy(self, log_p_hat, p):
        return -(log_p_hat * p).sum(dim=-1)

    def compute_matchability(self, keypoint_p, has_depth, B, K, H, W, device = "cuda"):
        smooth_keypoint_p = F.conv2d(keypoint_p.reshape(B,1,H,W), weight = self.smoothing_kernel, padding = (self.smoothing_size//2,0))
        smooth_keypoint_p = F.conv2d(smooth_keypoint_p, weight = self.smoothing_kernel.mT, padding = (0,self.smoothing_size//2))
        log_p_hat = (smooth_keypoint_p+1e-8).log().reshape(B,H*W).log_softmax(dim=-1)
        smooth_has_depth = F.conv2d(has_depth.reshape(B,1,H,W), weight = self.smoothing_kernel, padding = (0,self.smoothing_size//2))
        smooth_has_depth = F.conv2d(smooth_has_depth, weight = self.smoothing_kernel.mT, padding = (self.smoothing_size//2,0)).reshape(B,H*W)
        p = smooth_has_depth/smooth_has_depth.sum(dim=-1,keepdim=True)
        return self.compute_cross_entropy(log_p_hat, p) - self.compute_cross_entropy((p+1e-12).log(), p)

    def supervised_loss(self, outputs, batch):
        keypoint_logits_A, keypoint_logits_B = outputs["keypoint_logits"].chunk(2)
        B, K, H, W = keypoint_logits_A.shape

        detections_A, detections_B = batch["detections_A"], batch["detections_B"]
        
        gt_warp_A_to_B, valid_mask_A_to_B = get_gt_warp(                
                    batch["im_A_depth"],
                    batch["im_B_depth"],
                    batch["T_1to2"],
                    batch["K1"],
                    batch["K2"],
                    H=H,
                    W=W,
                )
        gt_warp_B_to_A, valid_mask_B_to_A = get_gt_warp(                
            batch["im_B_depth"],
            batch["im_A_depth"],
            batch["T_1to2"].inverse(),
            batch["K2"],
            batch["K1"],
            H=H,
            W=W,
        )
        keypoint_logits_A = keypoint_logits_A.reshape(B, K, H*W)
        keypoint_logits_B = keypoint_logits_B.reshape(B, K, H*W)
        keypoint_logits = torch.cat((keypoint_logits_A, keypoint_logits_B))

        B = 2*B
        gt_warp = torch.cat((gt_warp_A_to_B, gt_warp_B_to_A))
        valid_mask = torch.cat((valid_mask_A_to_B, valid_mask_B_to_A))
        valid_mask = valid_mask.reshape(B,H*W)
        binary_mask = valid_mask == 1
        detections = torch.cat((detections_A, detections_B))
        legit_detections = ((detections > 0).prod(dim = -1) * (detections[...,0] < W) * (detections[...,1] < H)).bool()
        det_imgs_A, det_imgs_B = self.compute_detection_img(detections, legit_detections, B, H, W).chunk(2)
        det_imgs = torch.cat((det_imgs_A, det_imgs_B))
        det_imgs_backwarped = F.grid_sample(torch.cat((det_imgs_B, det_imgs_A)).reshape(B,1,H,W), 
            gt_warp[...,-2:].reshape(B,H,W,2).float(), align_corners = False, mode = "bicubic")

        keypoint_logits_backwarped = F.grid_sample(torch.cat((keypoint_logits_B, keypoint_logits_A)).reshape(B,K,H,W), 
            gt_warp[...,-2:].reshape(B,H,W,2).float(), align_corners = False, mode = "bicubic")
        
        keypoint_logits_backwarped = (keypoint_logits_backwarped).reshape(B,K,H*W)


        depth = F.interpolate(torch.cat((batch["im_A_depth"][:,None],batch["im_B_depth"][:,None]),dim=0), size = (H,W), mode = "bilinear", align_corners=False)
        has_depth = (depth > 0).float().reshape(B,H*W)
        
        joint_log_likelihood_loss = self.compute_joint_neg_log_likelihood(keypoint_logits, keypoint_logits_backwarped, 
                                                                          mask = binary_mask, detections_A = det_imgs, 
                                                                          detections_B_to_A = det_imgs_backwarped).mean()
        keypoint_p = keypoint_logits.reshape(B, K*H*W).softmax(dim=-1).reshape(B, K, H*W).sum(dim=1)
        matchability_loss = self.compute_matchability(keypoint_p, has_depth, B, K, H, W).mean()
        B = B//2
        kpts_A = sample_keypoints(keypoint_p[:B].reshape(B,H,W), 
                                use_nms = False, sample_topk = True, num_samples = 4*2048)
        kpts_B = sample_keypoints(keypoint_p[B:].reshape(B,H,W),
                                use_nms = False, sample_topk = True, num_samples = 4*2048)
        kpts_A_to_B = F.grid_sample(gt_warp_A_to_B[...,2:].float().permute(0,3,1,2), kpts_A[...,None,:], 
                                    align_corners=False, mode = 'bilinear')[...,0].mT
        legit_A_to_B = F.grid_sample(valid_mask_A_to_B.reshape(B,1,H,W), kpts_A[...,None,:], 
                                    align_corners=False, mode = 'bilinear')[...,0,:,0]
        percent_inliers = (torch.cdist(kpts_A_to_B, kpts_B).min(dim=-1).values[legit_A_to_B > 0] < 0.01).float().mean()
        self.tracked_metrics["mega_percent_inliers"] = (0.9 * self.tracked_metrics.get("mega_percent_inliers", percent_inliers) + 0.1 * percent_inliers)

        tot_loss = joint_log_likelihood_loss + self.matchability_weight * matchability_loss# 
        if torch.rand(1) > 1:
            print(f"Precent Inlier: {self.tracked_metrics.get('mega_percent_inliers', 0)}")
            print(f"{joint_log_likelihood_loss=} {matchability_loss=}")
            print(f"Total Loss: {tot_loss.item()}")
        return  tot_loss
    
    def forward(self, outputs, batch):
        
        if not isinstance(outputs, list):
            outputs = [outputs]
        losses = 0
        for output in outputs:
            losses = losses + self.supervised_loss(output, batch)
        return losses