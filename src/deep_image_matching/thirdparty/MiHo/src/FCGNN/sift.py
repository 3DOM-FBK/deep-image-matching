import torch
import torch.nn as nn
import cv2
import numpy as np

class SIFT(nn.Module):
    def __init__(self):
        super().__init__()

        try:
            self.sift = cv2.SIFT_create()
        except:
            self.sift = cv2.xfeatures2d.SIFT_create()

    def forward(self, img1, img2, device='cpu'):

        kp1, des1 = self.sift.detectAndCompute(img1,None)
        kp2, des2 = self.sift.detectAndCompute(img2,None)

        desc1 = torch.from_numpy(des1).float().to(device)
        desc2 = torch.from_numpy(des2).float().to(device)

        match_ids, _ = mnn_matching(desc1, desc2, threshold=0)

        match_ids = match_ids.cpu().numpy()

        kp1 = np.array([kp.pt for kp in kp1])
        kp2 = np.array([kp.pt for kp in kp2])

        p1 = kp1[match_ids[:, 0]]
        p2 = kp2[match_ids[:, 1]]

        matches = np.hstack((p1, p2))
        matches = torch.tensor(matches).float().to(device)

        return matches

def mnn_matching(desc1, desc2, threshold=None):

    desc1 = desc1 / desc1.norm(dim=1, keepdim=True)
    desc2 = desc2 / desc2.norm(dim=1, keepdim=True)

    martix = desc1 @ desc2.t()

    nn12 = martix.max(dim=1)[1]
    nn21 = martix.max(dim=0)[1]

    ids1 = torch.arange(0, martix.shape[0], device=desc1.device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    scores = martix.max(dim=1)[0][mask]

    if threshold is not None:
        mask = scores > threshold
        matches = matches[mask]    
        scores = scores[mask]

    return matches, scores
