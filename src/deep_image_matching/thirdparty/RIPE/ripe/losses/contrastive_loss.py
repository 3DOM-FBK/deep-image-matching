import torch
import torch.nn as nn
import torch.nn.functional as F


def second_nearest_neighbor(desc1, desc2):
    if desc2.shape[0] < 2:  # We cannot perform snn check, so output empty matches
        raise ValueError("desc2 should have at least 2 descriptors")

    dist = torch.cdist(desc1, desc2, p=2)

    vals, idxs = torch.topk(dist, 2, dim=1, largest=False)
    idxs_in_2 = idxs[:, 1]
    idxs_in_1 = torch.arange(0, idxs_in_2.size(0), device=dist.device)

    matches_idxs = torch.cat([idxs_in_1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)

    return vals[:, 1].view(-1, 1), matches_idxs


def contrastive_loss(
    desc1,
    desc2,
    matches,
    inliers,
    label,
    logits_1,
    logits_2,
    pos_margin=1.0,
    neg_margin=1.0,
):
    if inliers.sum() < 8:  # if there are too few inliers, calculate loss on all matches
        inliers = torch.ones_like(inliers)

    matched_inliers_descs1 = desc1[matches[:, 0][inliers]]
    matched_inliers_descs2 = desc2[matches[:, 1][inliers]]

    if logits_1 is not None and logits_2 is not None:
        matched_inliers_logits1 = logits_1[matches[:, 0][inliers]]
        matched_inliers_logits2 = logits_2[matches[:, 1][inliers]]
        logits = torch.minimum(matched_inliers_logits1, matched_inliers_logits2)
    else:
        logits = torch.ones_like(matches[:, 0][inliers])

    if label:
        snn_match_dists_1, idx1 = second_nearest_neighbor(matched_inliers_descs1, desc2)
        snn_match_dists_2, idx2 = second_nearest_neighbor(matched_inliers_descs2, desc1)

        dists = torch.hstack((snn_match_dists_1, snn_match_dists_2))
        min_dists_idx = torch.min(dists, dim=1).indices.unsqueeze(1)

        dists_hard = torch.gather(dists, 1, min_dists_idx).squeeze(-1)
        dists_pos = F.pairwise_distance(matched_inliers_descs1, matched_inliers_descs2)

        contrastive_loss = torch.clamp(pos_margin + dists_pos - dists_hard, min=0.0)

        contrastive_loss = contrastive_loss * logits

        contrastive_loss = contrastive_loss.sum() / (logits.sum() + 1e-8)  # small epsilon to avoid division by zero
    else:
        dists = F.pairwise_distance(matched_inliers_descs1, matched_inliers_descs2)
        contrastive_loss = torch.clamp(neg_margin - dists, min=0.0)

        contrastive_loss = contrastive_loss * logits

        contrastive_loss = contrastive_loss.sum() / (logits.sum() + 1e-8)  # small epsilon to avoid division by zero

    return contrastive_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=1.0, neg_margin=1.0):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, desc1, desc2, matches, inliers, label, logits_1=None, logits_2=None):
        return contrastive_loss(
            desc1,
            desc2,
            matches,
            inliers,
            label,
            logits_1,
            logits_2,
            self.pos_margin,
            self.neg_margin,
        )
