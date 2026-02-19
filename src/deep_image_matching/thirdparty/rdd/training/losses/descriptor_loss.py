import torch
import torch.nn.functional as F

from training.utils import *
from torch import nn

class DescriptorLoss(nn.Module):
    def __init__(self, inv_temp = 20, dual_softmax_weight = 5, heatmap_weight = 1):
        super().__init__()
        self.inv_temp = inv_temp
        self.dual_softmax_weight = dual_softmax_weight
        self.heatmap_weight = heatmap_weight
    
    def forward(self, m1, m2, h1, h2, pts1, pts2):
        loss_ds = dual_softmax_loss(m1, m2, temp=20, normalize=True) * self.dual_softmax_weight
                    
        loss_h1, acc1 = heatmap_loss(h1, pts1)
        loss_h2, acc2 = heatmap_loss(h2, pts2)
        loss_h = (loss_h1 + loss_h2) / 2 * self.heatmap_weight
        
        acc_kp = 0.5 * (acc1 + acc2)
        
        return loss_ds, loss_h, acc_kp

def dual_softmax_loss(X, Y, temp = 1, normalize = False):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    if normalize:
        X = X/X.norm(dim=-1,keepdim=True)
        Y = Y/Y.norm(dim=-1,keepdim=True)
    
    dist_mat = (X @ Y.t()) * temp

    P = dist_mat.softmax(dim = -2) * dist_mat.softmax(dim= -1)
    
    conf_gt = torch.eye(len(X), device = X.device)
    pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
    
    conf_gt = torch.clamp(conf_gt, 1e-6, 1-1e-6)
    
    # focal loss
    alpha = 0.25
    gamma = 2
    pos_conf = P[pos_mask]
    loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()

    return loss_pos.mean()
    
def heatmap_loss(kpts, pts):
    C, H, W = kpts.shape

    with torch.no_grad():
        
        labels = torch.zeros((1, H, W), dtype=torch.long, device=kpts.device)
        labels[:, (pts[:,1]).long(), (pts[:,0]).long()] = 1
        
    kpts = kpts.view(-1)
    labels = labels.view(-1)
    
    BCE_loss = F.binary_cross_entropy(kpts, labels.float(), reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = 0.25 * (1 - pt) ** 2* BCE_loss
    
    with torch.no_grad():
        predictions = (kpts > 0.5)
        true_positives = ((predictions == 1) & (labels == 1)).sum().item()
        false_positives = ((predictions == 1) & (labels == 0)).sum().item()

        # Calculate Precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    return F_loss.mean(), precision
