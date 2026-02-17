import torch

def match_features(feats1, feats2, min_cossim=0.82):
    cossim = feats1 @ feats2.t()
    cossim_t = feats2 @ feats1.t()
    _, match12 = cossim.max(dim=1)
    _, match21 = cossim_t.max(dim=1)
    idx0 = torch.arange(len(match12), device=match12.device)
    mutual = match21[match12] == idx0
    # import pdb; pdb.set_trace()
    if min_cossim > 0:
        best_sim, _ = cossim.max(dim=1)          
        good = best_sim > min_cossim
        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]
    else:
        idx0 = idx0[mutual]
        idx1 = match12[mutual]

    match_scores = cossim[idx0, idx1]
    return idx0, idx1, match_scores