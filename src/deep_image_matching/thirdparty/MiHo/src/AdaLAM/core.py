from .ransac import ransac
from .utils import dist_matrix, orientation_diff
import numpy as np
import torch

def select_seeds(dist1, R1, scores1, n1, fnn12, mnn, th):
    im1neighmap = dist1 < R1 ** 2  # (n1, n1)
    # find out who scores higher than whom
    im1scorescomp = scores1.unsqueeze(1) > scores1.unsqueeze(0)  # (n1, n1)
    # find out who scores higher than all of its neighbors: seed points
    if mnn is not None:
        im1bs = (~ torch.any(im1neighmap & im1scorescomp & mnn.unsqueeze(0), dim=1)) & mnn & (scores1 < th) # (n1,)
    else:
        im1bs =(~ torch.any(im1neighmap & im1scorescomp, dim=1)) & (scores1 < th)


    # collect all seeds in both images and the 1NN of the seeds of the other image
    im1seeds = torch.where(im1bs)[0]  # (n1bs) index format
    im2seeds = fnn12[im1bs]  # (n1bs) index format
    return im1seeds, im2seeds


def extract_neighborhood_sets(o1, o2, s1, s2, dist1, im1seeds, im2seeds, k1, k2, R1, R2, fnn12, ORIENTATION_THR,
                              SCALE_RATE_THR, SEARCH_EXP, MIN_INLIERS):
    dst1 = dist1[im1seeds, :]
    dst2 = dist_matrix(k2[fnn12[im1seeds]], k2[fnn12])
    local_neighs_mask = (dst1 < (SEARCH_EXP * R1) ** 2) \
                        & (dst2 < (SEARCH_EXP * R2) ** 2)

    if ORIENTATION_THR is not None and ORIENTATION_THR < 180:
        relo = orientation_diff(o1, o2[fnn12])
        orientation_diffs = torch.abs(orientation_diff(relo.unsqueeze(0), relo[im1seeds].unsqueeze(1)))
        local_neighs_mask = local_neighs_mask & (orientation_diffs < ORIENTATION_THR)
    if SCALE_RATE_THR is not None and SCALE_RATE_THR < 10:
        rels = s2[fnn12] / s1
        scale_rates = rels[im1seeds].unsqueeze(1) / rels.unsqueeze(0)
        local_neighs_mask = local_neighs_mask & (scale_rates < SCALE_RATE_THR) \
                            & (scale_rates > 1 / SCALE_RATE_THR)  # (ns, n1)

    numn1 = torch.sum(local_neighs_mask, dim=1)
    valid_seeds = numn1 >= MIN_INLIERS

    local_neighs_mask = local_neighs_mask[valid_seeds, :]

    rdims = numn1[valid_seeds]

    return local_neighs_mask, rdims, im1seeds[valid_seeds], im2seeds[valid_seeds]


def extract_local_patterns(fnn12, fnn_to_seed_local_consistency_map_corr, k1, k2, im1seeds, im2seeds, scores):
    ransidx, tokp1 = torch.where(fnn_to_seed_local_consistency_map_corr)
    tokp2 = fnn12[tokp1]

    im1abspattern = k1[tokp1]
    im2abspattern = k2[tokp2]

    im1loc = im1abspattern - k1[im1seeds[ransidx]]
    im2loc = im2abspattern - k2[im2seeds[ransidx]]

    expanded_local_scores = scores[tokp1] + ransidx.type(scores.dtype)

    sorting_perm = torch.argsort(expanded_local_scores)

    im1loc = im1loc[sorting_perm]
    im2loc = im2loc[sorting_perm]
    tokp1 = tokp1[sorting_perm]
    tokp2 = tokp2[sorting_perm]

    return im1loc, im2loc, ransidx, tokp1, tokp2

def adalam_core(k1, k2, fnn12, scores1, config, mnn=None, im1shape=None, im2shape=None, o1=None, o2=None, s1=None, s2=None):
    AREA_RATIO = config['area_ratio']
    SEARCH_EXP = config['search_expansion']
    RANSAC_ITERS = config['ransac_iters']
    MIN_INLIERS = config['min_inliers']
    MIN_CONF = config['min_confidence']
    ORIENTATION_THR = config['orientation_difference_threshold']
    SCALE_RATE_THR = config['scale_rate_threshold']
    REFIT = config['refit']
    TH = config['th']

    if im1shape is None:
        k1mins, _ = torch.min(k1, dim=0)
        k1maxs, _ = torch.max(k1, dim=0)
        im1shape = (k1maxs - k1mins).cpu().numpy()
    if im2shape is None:
        k2mins, _ = torch.min(k2, dim=0)
        k2maxs, _ = torch.max(k2, dim=0)
        im2shape = (k2maxs - k2mins).cpu().numpy()

    R1 = np.sqrt(np.prod(im1shape[:2]) / AREA_RATIO / np.pi)
    R2 = np.sqrt(np.prod(im2shape[:2]) / AREA_RATIO / np.pi)

    n1 = k1.shape[0]
    n2 = k2.shape[0]

    dist1 = dist_matrix(k1, k1)
    im1seeds, im2seeds = select_seeds(dist1, R1, scores1, n1, fnn12, mnn, TH)

    local_neighs_mask, rdims, im1seeds, im2seeds = extract_neighborhood_sets(o1, o2, s1, s2, dist1,
                                                                             im1seeds, im2seeds,
                                                                             k1, k2, R1, R2, fnn12,
                                                                             ORIENTATION_THR,
                                                                             SCALE_RATE_THR,
                                                                             SEARCH_EXP, MIN_INLIERS)

    if rdims.shape[0] == 0:
        # No seed point survived. Just output ratio-test matches. This should happen very rarely.
        absolute_im1idx = torch.where(scores1 < TH)[0]
        absolute_im2idx = fnn12[absolute_im1idx]
        return torch.stack([absolute_im1idx, absolute_im2idx], dim=1)

    im1loc, im2loc, ransidx, tokp1, tokp2 = extract_local_patterns(fnn12,
                                                                   local_neighs_mask,
                                                                   k1, k2, im1seeds,
                                                                   im2seeds, scores1)
    im1loc = im1loc / (R1 * SEARCH_EXP)
    im2loc = im2loc / (R2 * SEARCH_EXP)

    inlier_idx, _, \
    inl_count_sign, inlier_counts = ransac(xsamples=im1loc,
                                           ysamples=im2loc,
                                           rdims=rdims, iters=RANSAC_ITERS,
                                           refit=REFIT, config=config)


    ics = inl_count_sign[ransidx[inlier_idx]]
    ica = inlier_counts[ransidx[inlier_idx]].float()
    passed_inliers_mask = (ics >= (1-1/MIN_CONF)) & (ica * ics >= MIN_INLIERS)
    accepted_inliers = inlier_idx[passed_inliers_mask]

    absolute_im1idx = tokp1[accepted_inliers]
    absolute_im2idx = tokp2[accepted_inliers]

    # inlier_seeds_idx = torch.unique(ransidx[accepted_inliers])

    # absolute_im1idx = torch.cat([absolute_im1idx, im1seeds[inlier_seeds_idx]])
    # absolute_im2idx = torch.cat([absolute_im2idx, im2seeds[inlier_seeds_idx]])
    
    final_matches = torch.stack([absolute_im1idx, absolute_im2idx], dim=1)
    if final_matches.shape[0] > 1:
        return torch.unique(final_matches, dim=0)
    return final_matches
