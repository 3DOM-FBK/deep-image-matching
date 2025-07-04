from PIL import Image
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import warnings
from .ncc import refinement_miho


# cv2.ocl.setUseOpenCL(False)
# matplotlib.use('tkagg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS_ = torch.finfo(torch.float32).eps
sqrt2 = np.sqrt(2)


def get_error_duplex(H12, pt1, pt2, ptm, sidx_par):
    l2 = sidx_par.size()[0]        
    n = pt1.size()[1]
    
    ptm_reproj = torch.cat((torch.matmul(H12[:l2], pt1), torch.matmul(H12[l2:], pt2)), dim=0)
    sign_ptm = torch.sign(ptm_reproj[:, 2])

    pt12_reproj = torch.linalg.solve(H12, ptm)
    sign_pt12 = torch.sign(pt12_reproj[:, 2])

    idx_aux = torch.arange(l2*2, device=device)*n + sidx_par[:, 0].repeat(2)

    sa = sign_ptm.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    sb = sign_pt12.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    
    ssa = sa.unsqueeze(1) == sign_ptm
    ssb = sb.unsqueeze(1) == sign_pt12

    mask = torch.logical_and(ssa, ssb)
    
    err_m = ptm_reproj[:, :2] / ptm_reproj[:, 2].unsqueeze(1) - ptm[:2]
    err_12 = torch.cat((pt12_reproj[:l2, :2] / pt12_reproj[:l2, 2].unsqueeze(1) - pt1[:2], pt12_reproj[l2:, :2] / pt12_reproj[l2:, 2].unsqueeze(1) - pt2[:2]), dim=0)

    err = torch.maximum(torch.sum(err_m ** 2, dim=1), torch.sum(err_12 ** 2, dim=1))
    err[torch.logical_or(~torch.isfinite(err), ~mask)] = float('inf')

    return torch.maximum(err[:l2], err[l2:])


def get_inlier_duplex(H12, pt1, pt2, ptm, sidx_par, th):
    l2 = sidx_par.size()[0]        
    n = pt1.size()[1]
    
    ptm_reproj = torch.cat((torch.matmul(H12[:l2], pt1), torch.matmul(H12[l2:], pt2)), dim=0)
    sign_ptm = torch.sign(ptm_reproj[:, 2])
    
    pt12_reproj, bad_matrix = torch.linalg.solve_ex(H12, ptm.unsqueeze(0))
    pt12_reproj[bad_matrix > 0, 2] = 0 #
    sign_pt12 = torch.sign(pt12_reproj[:, 2])

    idx_aux = torch.arange(l2*2, device=device).unsqueeze(1)*n + sidx_par.repeat(2,1)

    sa = sign_ptm.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    sb = sign_pt12.flatten()[idx_aux.flatten()].reshape(idx_aux.size())
    
    ssa = torch.all(sa[:, 0].unsqueeze(1) == sa, dim=1)
    ssb = torch.all(sb[:, 0].unsqueeze(1) == sb, dim=1)

    ssa_ = sa[:, 0].unsqueeze(1) == sign_ptm
    ssb_ = sb[:, 0].unsqueeze(1) == sign_pt12

    mask = (ssa_ & ssb_) & (ssa & ssb).unsqueeze(1)
    
    err_m = ptm_reproj[:, :2] / ptm_reproj[:, 2].unsqueeze(1) - ptm[:2]
    err_12 = torch.cat((pt12_reproj[:l2, :2] / pt12_reproj[:l2, 2].unsqueeze(1) - pt1[:2], pt12_reproj[l2:, :2] / pt12_reproj[l2:, 2].unsqueeze(1) - pt2[:2]), dim=0)

    err = torch.maximum(torch.sum(err_m ** 2, dim=1), torch.sum(err_12 ** 2, dim=1))
    err[~torch.isfinite(err)] = float('inf')
    err_ = err < th

    final_mask = mask & err_
    return torch.logical_and(final_mask[:, :l2], final_mask[:, l2:]).squeeze(dim=0)


def compute_homography_duplex(pt1, pt2, ptm, sidx_par):
    
    if sidx_par.dtype != torch.bool:
        l0 = sidx_par.size()[0]
        l1 = sidx_par.size()[1]
        
        pt1_par = pt1[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
        pt2_par = pt2[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
        ptm_par = ptm[:, sidx_par.flatten()].reshape(3, l0, l1).permute(1, 0, 2)
    else:
        l0 = 1
        l1 = sidx_par.sum()
        
        pt1_par = pt1[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)
        pt2_par = pt2[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)
        ptm_par = ptm[:, sidx_par].reshape(3, l0, l1).permute(1, 0, 2)
        

    c1 = torch.mean(pt1_par[:, :2], dim=2)
    c2 = torch.mean(pt2_par[:, :2], dim=2)
    cm = torch.mean(ptm_par[:, :2], dim=2)

    norm_diff_1 = torch.sqrt(torch.sum((pt1_par[:, :2] - c1.unsqueeze(2))**2, dim=1))
    norm_diff_2 = torch.sqrt(torch.sum((pt2_par[:, :2] - c2.unsqueeze(2))**2, dim=1))
    norm_diff_m = torch.sqrt(torch.sum((ptm_par[:, :2] - cm.unsqueeze(2))**2, dim=1))

    s1 = sqrt2 / (torch.mean(norm_diff_1, dim=1) + EPS_)
    s2 = sqrt2 / (torch.mean(norm_diff_2, dim=1) + EPS_)
    sm = sqrt2 / (torch.mean(norm_diff_m, dim=1) + EPS_)

    T12 = torch.zeros((l0*2, 3, 3), dtype=torch.float32, device=device)
    T12[:l0, 0, 0] = s1
    T12[:l0, 1, 1] = s1        
    T12[:l0, 2, 2] = 1
    T12[:l0, 0, 2] = -c1[:, 0] * s1
    T12[:l0, 1, 2] = -c1[:, 1] * s1

    T12[l0:, 0, 0] = s2
    T12[l0:, 1, 1] = s2
    T12[l0:, 2, 2] = 1
    T12[l0:, 0, 2] = -c2[:, 0] * s2
    T12[l0:, 1, 2] = -c2[:, 1] * s2

    Tm = torch.zeros((l0*2, 3, 3), dtype=torch.float32, device=device)
    Tm[l0:, 0, 0] = 1/sm
    Tm[l0:, 1, 1] = 1/sm
    Tm[l0:, 2, 2] = 1
    Tm[l0:, 0, 2] = cm[:, 0]
    Tm[l0:, 1, 2] = cm[:, 1]

    Tm[:l0, 0, 0] = 1/sm
    Tm[:l0, 1, 1] = 1/sm
    Tm[:l0, 2, 2] = 1
    Tm[:l0, 0, 2] = cm[:, 0]
    Tm[:l0, 1, 2] = cm[:, 1]


    p1x = s1.unsqueeze(1) * (pt1_par[:, 0] - c1[:, 0].unsqueeze(1))
    p1y = s1.unsqueeze(1) * (pt1_par[:, 1] - c1[:, 1].unsqueeze(1))

    p2x = s2.unsqueeze(1) * (pt2_par[:, 0] - c2[:, 0].unsqueeze(1))
    p2y = s2.unsqueeze(1) * (pt2_par[:, 1] - c2[:, 1].unsqueeze(1))

    pmx = sm.unsqueeze(1) * (ptm_par[:, 0] - cm[:, 0].unsqueeze(1))
    pmy = sm.unsqueeze(1) * (ptm_par[:, 1] - cm[:, 1].unsqueeze(1))


    A = torch.zeros((l0*2, l1*3, 9), dtype=torch.float32, device=device)


    A[:l0, :l1, 3] = -p1x
    A[:l0, :l1, 4] = -p1y
    A[:l0, :l1, 5] = -1

    A[:l0, :l1, 6] = torch.mul(pmy, p1x)
    A[:l0, :l1, 7] = torch.mul(pmy, p1y)
    A[:l0, :l1, 8] = pmy

    A[:l0, l1:2*l1, 0] = p1x
    A[:l0, l1:2*l1, 1] = p1y
    A[:l0, l1:2*l1, 2] = 1

    A[:l0, l1:2*l1, 6] = -torch.mul(pmx, p1x)
    A[:l0, l1:2*l1, 7] = -torch.mul(pmx, p1y)
    A[:l0, l1:2*l1, 8] = -pmx

    A[:l0, 2*l1:, 0] = -torch.mul(pmy, p1x)
    A[:l0, 2*l1:, 1] = -torch.mul(pmy, p1y)
    A[:l0, 2*l1:, 2] = -pmy

    A[:l0, 2*l1:, 3] = torch.mul(pmx, p1x)
    A[:l0, 2*l1:, 4] = torch.mul(pmx, p1y)
    A[:l0, 2*l1:, 5] = pmx


    A[l0:, :l1, 3] = -p2x
    A[l0:, :l1, 4] = -p2y
    A[l0:, :l1, 5] = -1

    A[l0:, :l1, 6] = torch.mul(pmy, p2x)
    A[l0:, :l1, 7] = torch.mul(pmy, p2y)
    A[l0:, :l1, 8] = pmy

    A[l0:, l1:2*l1, 0] = p2x
    A[l0:, l1:2*l1, 1] = p2y
    A[l0:, l1:2*l1, 2] = 1

    A[l0:, l1:2*l1, 6] = -torch.mul(pmx, p2x)
    A[l0:, l1:2*l1, 7] = -torch.mul(pmx, p2y)
    A[l0:, l1:2*l1, 8] = -pmx

    A[l0:, 2*l1:, 0] = -torch.mul(pmy, p2x)
    A[l0:, 2*l1:, 1] = -torch.mul(pmy, p2y)
    A[l0:, 2*l1:, 2] = -pmy

    A[l0:, 2*l1:, 3] = torch.mul(pmx, p2x)
    A[l0:, 2*l1:, 4] = torch.mul(pmx, p2y)
    A[l0:, 2*l1:, 5] = pmx


    _, D, V = torch.linalg.svd(A, full_matrices=True)
    H12 = V[:, -1].reshape(l0*2, 3, 3)
    H12 = Tm @ H12 @ T12

    sv = torch.amax(D[:, -2].reshape(2, l0), dim=0)
    
    return H12, sv


def data_normalize(pts):
    c = torch.mean(pts, dim=1)
    norm_diff = torch.sqrt((pts[0] - c[0])**2 + (pts[1] - c[1])**2)
    s = torch.sqrt(torch.tensor(2.0)) / (torch.mean(norm_diff) + EPS_)

    T = torch.tensor([
        [s, 0, -c[0] * s],
        [0, s, -c[1] * s],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    return T


def steps(pps, inl, p):
    e = 1 - inl
    r = torch.log(torch.tensor(1.0) - p) / torch.log(torch.tensor(1.0) - (torch.tensor(1.0) - e)**pps)
    return r


def compute_homography(pts1, pts2):
    T1 = data_normalize(pts1)
    T2 = data_normalize(pts2)

    npts1 = torch.matmul(T1, pts1)
    npts2 = torch.matmul(T2, pts2)

    l = npts1.shape[1]
    A = torch.zeros((l*3, 9), dtype=torch.float32, device=device)
    A[:l, 3:6] = -torch.mul(torch.tile(npts2[2], (3, 1)).t(), npts1.t())
    A[:l, 6:] = torch.mul(torch.tile(npts2[1], (3, 1)).t(), npts1.t())
    A[l:2*l, :3] = torch.mul(torch.tile(npts2[2], (3, 1)).t(), npts1.t())
    A[l:2*l, 6:] = -torch.mul(torch.tile(npts2[0], (3, 1)).t(), npts1.t())
    A[2*l:, :3] = -torch.mul(torch.tile(npts2[1], (3, 1)).t(), npts1.t())
    A[2*l:, 3:6] = torch.mul(torch.tile(npts2[0], (3, 1)).t(), npts1.t())

    try:
        _, D, V = torch.linalg.svd(A, full_matrices=True)
        H = V[-1, :].reshape(3, 3)
        H = torch.inverse(T2) @ H @ T1
    except:
        H = None
        D = torch.zeros(9, dtype=torch.float32)

    return H, D


def get_inliers(pt1, pt2, H, ths, sidx):

    l = pt1.shape[1]
    ths_ = [ths] if not isinstance(ths, list) else ths
    m = len(ths_)

    pt2_ = torch.matmul(H, pt1)
    s2_ = torch.sign(pt2_[2])
    s2 = s2_[sidx[0]]

    if not torch.all(s2_[sidx] == s2):
        nidx = torch.zeros((m, l), dtype=torch.bool, device=device)
        if not isinstance(ths, list):
            nidx = nidx[0]
        return nidx

    tmp2_ = pt2_[:2] / pt2_[2] - pt2[:2]
    err2 = torch.sum(tmp2_**2, axis=0)

    pt1_ = torch.matmul(torch.inverse(H), pt2)
    s1_ = torch.sign(pt1_[2])
    s1 = s1_[sidx[0]]

    if not torch.all(s1_[sidx] == s1):
        nidx = torch.zeros((m, l), dtype=torch.bool, device=device)
        if not isinstance(ths, list):
            nidx = nidx[0]
        return nidx

    tmp1_ = pt1_[:2] / pt1_[2] - pt1[:2]
    err1 = torch.sum(tmp1_**2, axis=0)

    err = torch.maximum(err1, err2)
    err[~torch.isfinite(err)] = float('inf')

    nidx = torch.zeros((m, l), dtype=torch.bool, device=device)
    for i in range(m):
        nidx[i] = torch.all(torch.stack((err < ths_[i], s2_ == s2, s1_ == s1)), dim=0)

    if not isinstance(ths, list):
        nidx = nidx[0]

    return nidx


def get_error(pt1, pt2, H, sidx):
    l = pt1.shape[1]

    pt2_ = torch.matmul(H, pt1)
    s2_ = torch.sign(pt2_[2])
    s2 = s2_[sidx[0]]

    if not torch.all(s2_[sidx] == s2):
        return torch.full((l,), float('inf'))

    tmp2_ = pt2_[:2] / pt2_[2] - pt2[:2]
    err2 = torch.sum(tmp2_**2, dim=0)

    pt1_ = torch.matmul(torch.inverse(H), pt2)
    s1_ = torch.sign(pt1_[2])
    s1 = s1_[sidx[0]]

    if not torch.all(s1_[sidx] == s1):
        return torch.full((l,), float('inf'))

    tmp1_ = pt1_[:2] / pt1_[2] - pt1[:2]
    err1 = torch.sum(tmp1_**2, dim=0)

    err = torch.maximum(err1, err2)
    err[~torch.isfinite(err)] = float('inf')

    return err


def sampler4_par(n_par, m):
    nn = n_par.size()[0]

    n_par = n_par.repeat(m)

    sidx = (torch.rand((m * nn, 4), device=device) * torch.stack((n_par, n_par-1, n_par-2, n_par-3)).permute(1,0)).type(torch.long)  

    for k in range(1,4):
        sidx[:, 0:k] = torch.sort(sidx[:, 0:k])[0]
        for kk in range(k):
            sidx[:, k] = sidx[:, k] + (sidx[:, k] >= sidx[:, kk])

    return sidx.reshape(m, nn, 4)


def ransac_middle(pt1, pt2, dd=None, th_grid=15, th_in=7, th_out=15, max_iter=2000, min_iter=50, p=0.9, svd_th=0.05, buffers=5, ssidx=None, par_value=100000):
    n = pt1.shape[1]

    th_in = th_in ** 2
    th_out = th_out ** 2

    ptm = (pt1 + pt2) / 2

    th = torch.tensor(th_out, device=device).reshape(1, 1, 1)
    ths = torch.tensor([th_in, th_out], device=device).reshape(2, 1, 1)

    if n < 4:
        H1 = torch.tensor([], device=device)
        H2 = torch.tensor([], device=device)
        iidx = torch.zeros(n, dtype=torch.bool, device=device)
        oidx = torch.zeros(n, dtype=torch.bool, device=device)
        vidx = torch.zeros((n, 0), dtype=torch.bool, device=device)
        sidx_ = torch.zeros((4,), dtype=torch.int32, device=device)
        return H1, H2, iidx, oidx, vidx, sidx_
    
    min_iter = min(min_iter, n*(n-1)*(n-2)*(n-3) // 12)

    vidx = torch.zeros((n, buffers), dtype=torch.bool, device=device)
    midx = torch.zeros((n, buffers+1), dtype=torch.bool, device=device)

    sum_midx = 0
    Nc = float('inf')
    min_th_stats = 3

    sn = ssidx.shape[1]
    sidx_ = torch.zeros((4,), dtype=torch.long, device=device)
    
    ssidx_sz = torch.sum(ssidx, dim=0)

    par_run = par_value // n

    c = 0
    best_model = torch.zeros((2, 3, 3), device=device)
    while c < max_iter:
        c_par = torch.arange(c, min(max_iter, c + par_run), device=device)
        c_par_sz = c_par.size()[0]

        n_par = torch.full((c_par_sz,), n, dtype=torch.int, device=device)
        for i in range(c_par_sz):
            if (c_par[i] < sn):
                if ssidx_sz[c_par[i]] > 4:
                    n_par[i] = ssidx_sz[c_par[i]]
            else:
                break

        sidx = sampler4_par(n_par, min_iter)

        for i in range(c_par_sz):
            if (c_par[i] < sn):
                if ssidx_sz[c_par[i]] > 4:
                    aux = sidx[:, c_par[i]].flatten()
                    tmp = torch.nonzero(ssidx[:, c_par[i]]).squeeze()
                    aux = tmp[aux]
                    sidx[:, c_par[i]] = aux.reshape(min_iter, 4)                
            else:
                break


        dr = sidx.repeat(1, 1, 4).flatten()
        dc = sidx.repeat_interleave(4, dim=2).flatten()
        
        dd_check = (((pt1[0, dr] - pt1[0, dc])**2 + (pt1[1, dr] - pt1[1, dc])**2 > th_grid**2) &
                    ((pt2[0, dr] - pt2[0, dc])**2 + (pt2[1, dr] - pt2[1, dc])**2 > th_grid**2)).reshape(min_iter, c_par_sz, -1)

        # dd_check = dd.flatten()[sidx.repeat((1, 1, 4)).flatten() * dd.size()[0] + sidx.repeat_interleave(4, dim=2).flatten()].reshape(min_iter, c_par_sz, -1)
        dd_good = torch.sum(dd_check, dim=-1) >= 12

        # good_sample_par = torch.zeros(c_par_sz, dtype=torch.bool, device=device)
        # sidx_par = torch.zeros((c_par_sz, 4), dtype=torch.long, device=device)
        #        
        # for i in range(c_par_sz):
        #     for j in range(min_iter):
        #         if dd_good[j, i]:
        #             good_sample_par[i] = True
        #             sidx_par[i, :] = sidx[j, i, :]
        #             break
                
        good_sample_par, good_idx = dd_good.max(dim=0)
        sidx_par = sidx.reshape(min_iter * c_par_sz, 4)[good_idx * c_par_sz + torch.arange(c_par_sz, device=device)]

        sidx_par = sidx_par[good_sample_par]
        c_par = c_par[good_sample_par]             
        
        H12, sv = compute_homography_duplex(pt1, pt2, ptm, sidx_par)
        good_H = sv > svd_th

        H12 = H12[good_H.repeat(2)]            
        sidx_par = sidx_par[good_H]
        c_par = c_par[good_H]
        
        if not c_par.size()[0]:
            if (c + par_run > Nc) and (c + par_run > min_iter):
                break
            else:
                c += par_run
                continue
                
        nidx_par = get_inlier_duplex(H12, pt1, pt2, ptm, sidx_par, th)                        
        sum_nidx_par = nidx_par.sum(dim=1)
        l2 = sidx_par.size()[0]

        sum_nidx_par, sort_idx = torch.sort(sum_nidx_par, descending=True)
        nidx_par = nidx_par[sort_idx]
        sidx_par = sidx_par[sort_idx]

        for i in range(l2):
            sum_nidx = sum_nidx_par[i]

            nidx = nidx_par[i]
            
            sidx_i = sidx_par[i]

            H1 = H12[sort_idx[i]]
            H2 = H12[sort_idx[i] + l2]

            updated_model = False
    
            midx[:, -1] = nidx
    
            if sum_nidx > min_th_stats:
    
                idxs = torch.arange(buffers+1)
                q = torch.tensor(n+1)
    
                for t in range(buffers):
                    uidx = ~torch.any(midx[:, idxs[:t]], dim=1).unsqueeze(1)
    
                    tsum = uidx & midx[:, idxs[t:]]
                    ssum = torch.sum(tsum, dim=0)
                    vidx[:, t] = tsum[:, torch.argmax(ssum)]
    
                    tt = torch.argmax((ssum[-1] > ssum[:-1]).type(torch.long))
                    if ssum[-1] > ssum[tt]:
                        aux = idxs[-1].clone()
                        idxs[-1] = idxs[t+tt].clone()
                        idxs[t+tt] = aux
                        if t == 0 and tt == 0:
                            sidx_ = sidx_i
    
                    q = torch.minimum(q, torch.max(ssum))
    
                min_th_stats = torch.maximum(torch.tensor(4), q)
    
                updated_model = idxs[0] != 0
                midx = midx[:, idxs]
    
            if updated_model:
                sum_midx = torch.sum(midx[:, 0])
                best_model[0] = H1
                best_model[1] = H2
                Nc = steps(4, sum_midx / n, p)
    
        if (c + par_run > Nc) and (c + par_run > min_iter):
            break
            
        c += par_run

    vidx = vidx[:, 1:]

    if sum_midx >= 4:
        bidx = midx[:, 0]

        H12, _ = compute_homography_duplex(pt1, pt2, ptm, bidx)
        H1, H2 = H12

        iidx, oidx = get_inlier_duplex(H12, pt1, pt2, ptm, sidx_.unsqueeze(0), ths)                        

        if sum_midx > torch.sum(oidx):
            H1, H2 = best_model

            iidx, oidx = get_inlier_duplex(best_model, pt1, pt2, ptm, sidx_.unsqueeze(0), ths)
    else:
        H1 = torch.tensor([], device=device)
        H2 = torch.tensor([], device=device)

        iidx = torch.zeros(n, dtype=torch.bool, device=device)
        oidx = torch.zeros(n, dtype=torch.bool, device=device)        
        
    return H1, H2, iidx, oidx, vidx, sidx_


def rot_best(pt1, pt2, n=4):
    # start = time.time()    
    
    n = int(n)
    if n < 1:
        n = 1
    
    # current MiHo formulation is translation invariant
    pt1 = pt1[:2]
    pt2 = pt2[:2]

    d0 = dist2(pt1.t())
    d2 = dist2(pt2.t())

    a = torch.arange(n) * 2 * np.pi / n
    R = torch.stack((torch.stack((torch.cos(a), -torch.sin(a)), dim=1), torch.stack((torch.sin(a), torch.cos(a)), dim=1)), dim=1).to(device)    

    pt2_ = torch.matmul(R, pt2)
    # ptm = (pt1[None, :] + pt2_) / 2
    ptm = (pt1 + pt2_) / 2

    d1 = dist2_batch(ptm.permute(0, 2, 1))
    
    # in_middle = (torch.sign(d0[None, :, :] - d1) * torch.sign(d1 - d2[None, :, :])) > 0    
    in_middle = (torch.sign(d0 - d1) * torch.sign(d1 - d2)) > 0    

    sum_all_k = torch.sum(in_middle,(1, 2))    
    # print(sum_all_k)
    
    best_i = torch.argmax(sum_all_k)
    
    aux = torch.eye(3).to(device)
    aux[:2, :2] = R[best_i]
               
    # end = time.time()
    # print("Elapsed rot_best time = %s" % (end - start))    
    
    return aux


def rot_best_block(pt1, pt2, n=4, split_sz=2048):
    # start = time.time()    
    
    n = int(n)
    if n < 1:
        n = 1
    
    # current MiHo formulation is translation invariant
    pt1 = pt1[:2]
    pt2 = pt2[:2]

    sum_all_k = torch.zeros(4, device=device)

    a = torch.arange(n) * 2 * np.pi / n
    R = torch.stack((torch.stack((torch.cos(a), -torch.sin(a)), dim=1), torch.stack((torch.sin(a), torch.cos(a)), dim=1)), dim=1).to(device)    

    pt1_blk = torch.split(pt1, split_sz, dim=1)
    pt2_blk = torch.split(pt2, split_sz, dim=1)

    for i in np.arange(len(pt1_blk)):
        for j in np.arange(len(pt1_blk)):
            d0 = dist2p(pt1_blk[i].t(), pt1_blk[j].t())
            d2 = dist2p(pt2_blk[i].t(), pt2_blk[j].t())

            pt2_i_ = torch.matmul(R, pt2_blk[i])
            pt2_j_ = torch.matmul(R, pt2_blk[j])

            ptm_i = (pt1_blk[i] + pt2_i_) / 2
            ptm_j = (pt1_blk[j] + pt2_j_) / 2

            d1 = dist2p_batch(ptm_i.permute(0, 2, 1), ptm_j.permute(0, 2, 1))
    
            # in_middle = (torch.sign(d0[None, :, :] - d1) * torch.sign(d1 - d2[None, :, :])) > 0    
            in_middle = (torch.sign(d0 - d1) * torch.sign(d1 - d2)) > 0    

            sum_all_k = sum_all_k + torch.sum(in_middle,(1, 2))    
    # print(sum_all_k)
    
    best_i = torch.argmax(sum_all_k)
    
    aux = torch.eye(3).to(device)
    aux[:2, :2] = R[best_i]
               
    # end = time.time()
    # print("Elapsed rot_best time = %s" % (end - start))    
    
    return aux


def get_avg_hom(pt1, pt2, ransac_middle_args={}, min_plane_pts=8, min_pt_gap=4,
                max_fail_count=3, random_seed_init=123, th_grid=15,
                rot_check=4):

    # set to 123 for debugging and profiling
    if random_seed_init is not None:
        torch.manual_seed(random_seed_init)

    H1 = torch.eye(3, device=device)
    H2 = torch.eye(3, device=device)

    Hdata = []
    l = pt1.shape[0]

    midx = torch.zeros(l, dtype=torch.bool, device=device)
    tidx = torch.zeros(l, dtype=torch.bool, device=device)

    # d1 = dist2(pt1) > th_grid**2
    # d2 = dist2(pt2) > th_grid**2
    # dd = d1 & d2

    pt1 = torch.cat((pt1.t(), torch.ones(1, l, device=device)))
    pt2 = torch.cat((pt2.t(), torch.ones(1, l, device=device)))

    if rot_check > 1:
        H2 = rot_best_block(pt1, pt2, rot_check)
        # H2 = rot_best(pt1, pt2, rot_check)


    fail_count = 0
    midx_sum = 0
    ssidx = torch.zeros((l, 0), dtype=torch.bool, device=device)
    sidx = torch.arange(l, device=device)

    pt1 = torch.matmul(H1, pt1)
    pt1 = pt1 / pt1[2]

    pt2 = torch.matmul(H2, pt2)
    pt2 = pt2 / pt2[2]

    while torch.sum(midx) < l - 4:
        pt1_ = pt1[:, ~midx]
        pt2_ = pt2[:, ~midx]

        # dd_ = dd[~midx, :][:, ~midx].to(device)
        dd_ = None

        ssidx = ssidx[~midx, :]

        H1_, H2_, iidx, oidx, ssidx, sidx_ = ransac_middle(pt1_, pt2_, dd_, th_grid, ssidx=ssidx, **ransac_middle_args)

        sidx_ = sidx[~midx][sidx_]

        # print(torch.sum(ssidx, dim=0))
        good_ssidx = torch.logical_not(torch.sum(ssidx, dim=0) == 0)
        ssidx = ssidx[:, good_ssidx]
        tsidx = torch.zeros((l, ssidx.shape[1]), dtype=torch.bool, device=device)
        tsidx[~midx, :] = ssidx
        ssidx = tsidx

        idx = torch.zeros(l, dtype=torch.bool, device=device)
        idx[~midx] = oidx

        midx[~midx] = iidx
        tidx = tidx | idx

        midx_sum_old = midx_sum
        midx_sum = torch.sum(midx)
        
        H_failed = torch.sum(oidx) <= min_plane_pts
        inl_failed = midx_sum - midx_sum_old <= min_pt_gap
        if H_failed or inl_failed:
            fail_count += 1
            if fail_count > max_fail_count:
                break
            if inl_failed:
                midx = tidx
            if H_failed:
                continue
        else:
            fail_count = 0
            
        # print(f"{torch.sum(tidx)} {torch.sum(midx)} {fail_count}")

        Hdata.append([torch.matmul(H1_, H1), torch.matmul(H2_, H2), idx, sidx_])

    return Hdata, H1, H2


def dist2_batch(pt):
    pt = pt.type(torch.float32)
    d = (pt.unsqueeze(-1)[:, :, 0] - pt.unsqueeze(1)[:, :, :, 0])**2 + (pt.unsqueeze(-1)[:, :, 1] - pt.unsqueeze(1)[:, :, :, 1])**2
    # d = (pt[:, :, 0, None] - pt[:, None, :, 0])**2 + (pt[:, :, 1, None] - pt[:, None, :, 1])**2
    return d


def dist2p_batch(pt1, pt2):
    pt1 = pt1.type(torch.float32)
    pt2 = pt2.type(torch.float32)
    d = (pt1.unsqueeze(-1)[:, :, 0] - pt2.unsqueeze(1)[:, :, :, 0])**2 + (pt1.unsqueeze(-1)[:, :, 1] - pt2.unsqueeze(1)[:, :, :, 1])**2
    return d


def dist2(pt):
    pt = pt.type(torch.float32)
    d = (pt.unsqueeze(-1)[:, 0] - pt.unsqueeze(0)[:, :, 0])**2 + (pt.unsqueeze(-1)[:, 1] - pt.unsqueeze(0)[:, :, 1])**2
    # d = (pt[:, 0, None] - pt[None, :, 0])**2 + (pt[:, 1, None] - pt[None, :, 1])**2
    return d


def dist2p(pt1, pt2):
    pt1 = pt1.type(torch.float32)
    pt2 = pt2.type(torch.float32)
    d = (pt1.unsqueeze(-1)[:, 0] - pt2.unsqueeze(0)[:, :, 0])**2 + (pt1.unsqueeze(-1)[:, 1] - pt2.unsqueeze(0)[:, :, 1])**2
    return d


def cluster_assign_base(Hdata, pt1, pt2, H1_pre, H2_pre, **dummy_args):
    l = len(Hdata)
    n = pt1.shape[0]
    
    if not((l>0) and (n>0)):
        return torch.full((n, ), -1, dtype=torch.int, device=device)
    
    inl_mask = torch.zeros((n, l), dtype=torch.bool, device=device)
    for i in range(l):
        inl_mask[:, i] = Hdata[i][2]

    alone_idx = torch.sum(inl_mask, dim=1) == 0
    set_size = torch.sum(inl_mask, dim=0)

    max_size_idx = torch.argmax(
        set_size.view(1, -1).expand(inl_mask.shape[0], -1) * inl_mask, dim=1)
    max_size_idx[alone_idx] = -1

    return max_size_idx


def cluster_assign(Hdata, pt1, pt2, H1_pre, H2_pre, median_th=5, err_th=15, **dummy_args):    
    l = len(Hdata)
    n = pt1.shape[0]

    if not((l>0) and (n>0)):
        return torch.full((n, ), -1, dtype=torch.int, device=device)

    pt1 = torch.vstack((pt1.T, torch.ones((1, n), device=device)))
    pt2 = torch.vstack((pt2.T, torch.ones((1, n), device=device)))

    pt1_ = torch.matmul(H1_pre, pt1)
    pt1_ = pt1_ / pt1_[2]

    pt2_ = torch.matmul(H2_pre, pt2)
    pt2_ = pt2_ / pt2_[2]

    ptm = (pt1_ + pt2_) / 2

    # err = torch.zeros((n, l), device=device)
    # inl_mask = torch.zeros((n, l), dtype=torch.bool, device=device)
    #
    # for i in range(l):
    #     H1 = Hdata[i][0]
    #     H2 = Hdata[i][1]
    #     sidx = Hdata[i][3]
    #
    #     inl_mask[:, i] = Hdata[i][2]
    #
    #     err[:, i] = torch.maximum(get_error(pt1, ptm, H1, sidx), get_error(pt2, ptm, H2, sidx))

    H12 = torch.zeros((l*2, 3, 3), device=device)
    sidx_par = torch.zeros((l, 4), device=device, dtype=torch.long)
    inl_mask = torch.zeros((n, l), dtype=torch.bool, device=device)

    for i in range(l):
        H12[i] = Hdata[i][0]
        H12[i+l] = Hdata[i][1]
        sidx_par[i] = Hdata[i][3]

        inl_mask[:, i] = Hdata[i][2]

    err = get_error_duplex(H12, pt1, pt2, ptm, sidx_par).permute(1,0)

    # min error
    abs_err_min_val, abs_err_min_idx = torch.min(err, dim=1)

    set_size = torch.sum(inl_mask, dim=0)
    size_mask = torch.repeat_interleave(set_size.unsqueeze(0), n, dim=0) * inl_mask

    # take a cluster if its cardinality is more than the median of the top median_th ones
    ssize_mask, _ = torch.sort(size_mask, descending=True, dim=1)

    median_idx = torch.sum(ssize_mask[:, :median_th] > 0, dim=1) / 2
    median_idx[median_idx == 1] = 1.5
    median_idx = torch.maximum(torch.ceil(median_idx).to(torch.int)-1, torch.tensor(0))

    # flat_indices = torch.arange(n) * ssize_mask.shape[1] + median_idx
    # top_median = ssize_mask.view(-1)[flat_indices]
    top_median = ssize_mask.flatten()[torch.arange(n, device=device) * ssize_mask.shape[1] + median_idx]

    # take among the selected the one which gives less error
    discarded_mask = size_mask < top_median.unsqueeze(1)
    err[discarded_mask] = float('inf')
    err_min_idx = torch.argmin(err, dim=1)

    # remove match with no cluster
    alone_idx = torch.sum(inl_mask, dim=1) == 0
    really_alone_idx = alone_idx & (abs_err_min_val > err_th**2)

    err_min_idx[alone_idx] = abs_err_min_idx[alone_idx]
    err_min_idx[really_alone_idx] = -1

    return err_min_idx


def cluster_assign_other(Hdata, pt1, pt2, H1_pre, H2_pre, err_th_only=15, **dummy_args):
    l = len(Hdata)
    n = pt1.shape[0]

    if not((l>0) and (n>0)):
        return torch.full((n, ), -1, dtype=torch.int, device=device)

    pt1 = torch.vstack((pt1.T, torch.ones((1, n), device=device)))
    pt2 = torch.vstack((pt2.T, torch.ones((1, n), device=device)))

    pt1_ = torch.matmul(H1_pre, pt1)
    pt1_ = pt1_ / pt1_[2]

    pt2_ = torch.matmul(H2_pre, pt2)
    pt2_ = pt2_ / pt2_[2]

    ptm = (pt1_ + pt2_) / 2

    # err = torch.zeros((n, l), device=device)
    #
    # for i in range(l):
    #     H1 = Hdata[i][0]
    #     H2 = Hdata[i][1]
    #     sidx = Hdata[i][3]
    #
    #     err[:, i] = torch.maximum(get_error(pt1, ptm, H1, sidx), get_error(pt2, ptm, H2, sidx))

    H12 = torch.zeros((l*2, 3, 3), device=device)
    sidx_par = torch.zeros((l, 4), device=device, dtype=torch.long)

    for i in range(l):
        H12[i] = Hdata[i][0]
        H12[i+l] = Hdata[i][1]
        sidx_par[i] = Hdata[i][3]

    err = get_error_duplex(H12, pt1, pt2, ptm, sidx_par).permute(1,0)

    # err_min_idx = torch.argmin(err, dim=1)
    # err_min_val = err.flatten()[torch.arange(n, device=device) * l + err_min_idx]

    err_min_val, err_min_idx = torch.min(err, dim=1)

    err_min_idx[(err_min_val > err_th_only**2) | torch.isnan(err_min_val)] = -1

    return err_min_idx


def show_fig(im1, im2, pt1, pt2, Hidx, tosave='miho_buffered_rot_pytorch_gpu.pdf', fig_dpi=300,
             colors = ['#FF1F5B', '#00CD6C', '#009ADE', '#FFC61E', '#F28522', '#AF58BA'],
             markers = ['o','^','s','p','h'], bad_marker = 'd', bad_color = '#000000',
             plot_opt = {'markersize': 2, 'markeredgewidth': 0.5,
                         'markerfacecolor': "None", 'alpha': 0.5}):

    transform = transforms.ToPILImage()
    im1 = transform(im1.type(torch.uint8))
    im2 = transform(im2.type(torch.uint8))

    im12 = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    im12.paste(im1, (0, 0))
    im12.paste(im2, (im1.width, 0))

    plt.figure()
    plt.axis('off')
    plt.imshow(im12)

    cn = len(colors)
    mn = len(markers)

    for i, idx in enumerate(np.ndarray.tolist(np.unique(Hidx))):
        mask = Hidx == idx
        x = np.vstack((pt1[mask, 0], pt2[mask, 0]+im1.width))
        y = np.vstack((pt1[mask, 1], pt2[mask, 1]))
        if (idx == -1):
            color = bad_color
            marker = bad_marker
            plot_opt['markerfacecolor'] = color
        else:
            color = colors[((i-1)%(cn*mn))%cn]
            marker = markers[((i-1)%(cn*mn))//cn]
            plot_opt['markerfacecolor'] = color
        plt.plot(x, y, linestyle='', color=color, marker=marker, **plot_opt)

    plt.savefig(tosave, dpi = fig_dpi, bbox_inches='tight')


def go_assign(Hdata, pt1, pt2, H1_pre, H2_pre, method=cluster_assign, method_args={}):
    return method(Hdata, pt1, pt2, H1_pre, H2_pre, **method_args)


def purge_default(dict1, dict2):

    if type(dict1) != type(dict2):
        return None

    if not (isinstance(dict2, dict) or isinstance(dict2, list)):
        if dict1 == dict2:
            return None
        else:
            return dict2

    elif isinstance(dict2, dict):

        keys1 = list(dict1.keys())
        keys2 = list(dict2.keys())

        for i in keys2:
            if i not in keys1:
                dict2.pop(i, None)
            else:
                aux = purge_default(dict1[i], dict2[i])
                if not aux:
                    dict2.pop(i, None)
                else:
                    dict2[i] = aux

        return dict2

    elif isinstance(dict2, list):

        if len(dict1) != len(dict2):
            return dict2

        for i in range(len(dict2)):
            aux = purge_default(dict1[i], dict2[i])
            if aux:
               return dict2

        return None


def merge_params(dict1, dict2):

    keys1 = dict1.keys()
    keys2 = dict2.keys()

    for i in keys1:
        if (i in keys2):
            if (not isinstance(dict1[i], dict)):
                dict1[i] = dict2[i]
            else:
                dict1[i] = merge_params(dict1[i], dict2[i])

    return dict1


class miho:
    def __init__(self, params=None):
        """initiate MiHo"""
        self.set_default()

        if params is not None:
            self.update_params(params)


    def set_default(self):
        """set default MiHo parameters"""
        self.params = { 'get_avg_hom': {}, 'go_assign': {}, 'show_clustering': {}}


    def get_current(self):
        """get current MiHo parameters"""
        tmp_params = self.params.copy()

        for i in ['get_avg_hom', 'show_clustering', 'go_assign']:
            if i not in tmp_params:
                tmp_params[i] = {}

        return merge_params(self.all_params(), tmp_params)


    def update_params(self, params):
        """update current MiHo parameters"""
        all_default_params = self.all_params()
        clear_params = purge_default(all_default_params, params.copy())

        for i in ['get_avg_hom', 'show_clustering', 'go_assign']:
            if i in clear_params:
                self.params[i] = clear_params[i]


    @staticmethod
    def all_params():
        """all MiHo parameters with default values"""
        ransac_middle_params = {'th_in': 7, 'th_out': 15, 'max_iter': 2000,
                                'min_iter': 50, 'p' :0.9, 'svd_th': 0.05,
                                'buffers': 5}
        get_avg_hom_params = {'ransac_middle_args': ransac_middle_params,
                              'min_plane_pts': 8, 'min_pt_gap': 4,
                              'max_fail_count': 3, 'random_seed_init': 123,
                              'th_grid': 15, 'rot_check': 4}

        method_args_params = {'median_th': 5, 'err_th': 15, 'err_th_only': 15, 'par_value': 100000}
        go_assign_params = {'method': cluster_assign,
                            'method_args': method_args_params}

        show_clustering_params = {'tosave': 'miho_buffered_rot_pytorch_gpu.pdf', 'fig_dpi': 300,
             'colors': ['#FF1F5B', '#00CD6C', '#009ADE', '#FFC61E', '#F28522', '#AF58BA'],
             'markers': ['o','^','s','p','h'], 'bad_marker': 'd', 'bad_color': '#000000',
             'plot_opt': {'markersize': 2, 'markeredgewidth': 0.5,
                          'markerfacecolor': "None", 'alpha': 0.5}}

        return {'get_avg_hom': get_avg_hom_params,
                'go_assign': go_assign_params,
                'show_clustering': show_clustering_params}


    def planar_clustering(self, pt1, pt2):
        """run MiHo"""
        self.pt1 = pt1
        self.pt2 = pt2

        Hdata, H1_pre, H2_pre = get_avg_hom(self.pt1, self.pt2, **self.params['get_avg_hom'])
        self.Hs = Hdata
        self.H1_pre = H1_pre
        self.H2_pre = H2_pre

        self.Hidx = go_assign(Hdata, self.pt1, self.pt2, H1_pre, H2_pre, **self.params['go_assign'])

        return self.Hs, self.Hidx


    def attach_images(self, im1, im2):
        """" add image pair to MiHo and tensorify it"""

        transform_gray = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 

        transform = transforms.PILToTensor() 

#       self.img1 = im1.copy()
#       self.img2 = im2.copy()
 
        self.img1 = transform(im1).type(torch.float16).to(device)
        self.img2 = transform(im2).type(torch.float16).to(device)

        self.im1 = transform_gray(im1).type(torch.float16).to(device)
        self.im2 = transform_gray(im2).type(torch.float16).to(device)
        
        return True


    def show_clustering(self):
        """ show MiHo clutering"""
        if hasattr(self, 'Hs') and hasattr(self, 'img1'):
            show_fig(self.img1, self.img2, self.pt1.cpu(), self.pt2.cpu(), self.Hidx.cpu(), **self.params['show_clustering'])
        else:
            warnings.warn("planar_clustering must run before!!!")


class miho_module:
    def __init__(self, **args):
        self.miho = miho()
        
        for k, v in args.items():
           setattr(self, k, v)

        if hasattr(self, 'max_iter'):
            params = self.miho.get_current()
            params['get_avg_hom']['ransac_middle_args']['max_iter'] = self.max_iter
            self.miho.update_params(params)        
        
        
    def get_id(self):
        if not hasattr(self, 'max_iter'):
            return ('miho_default_duplex').lower()
        else:
            return ('miho_duplex_max_iter_' + str(self.max_iter)).lower()
            

    def run(self, **args):
        self.miho.planar_clustering(args['pt1'], args['pt2'])
        
        pt1, pt2, Hs_miho, inliers, Hs_laf = refinement_miho(None, None, args['pt1'], args['pt2'], self.miho, args['Hs'], remove_bad=True, img_patches=False, also_laf=True)        
            
        toreturn = {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_miho, 'mask': inliers}
        if not (Hs_laf is None): toreturn['Hs_prev'] = Hs_laf
        return toreturn
