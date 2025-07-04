import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def laf2homo(kps):
    c = kps[:, :, 2]
    s = torch.sqrt(torch.abs(kps[:, 0, 0] * kps[:, 1, 1] - kps[:, 0, 1] * kps[:, 1, 0]))   
    
    Hi = torch.zeros((kps.shape[0], 3, 3), device=device)
    Hi[:, :2, :] = kps / s.reshape(-1, 1, 1)
    Hi[:, 2, 2] = 1 

    H = torch.linalg.inv(Hi)
    
    return c, H, s


def refinement_laf(im1, im2, pt1=None, pt2=None, data1=None, data2=None, w=15, img_patches=True, im1_disp=None, im2_disp=None):
    if data1 is None:
        l = pt1.shape[0]
        Hs = torch.eye(3, device=device).repeat(l*2, 1).reshape(l, 2, 3, 3)
    else:
        l = data1.shape[0]
        pt1, H1, s1 = laf2homo(data1)
        pt2, H2, s2 = laf2homo(data2)

        s = torch.sqrt(s1 * s2)       
        H1[:, :2, :] = H1[:, :2, :] * (s / s1).reshape(-1, 1, 1)
        H2[:, :2, :] = H2[:, :2, :] * (s / s2).reshape(-1, 1, 1)

        Hs = torch.cat((H1.unsqueeze(1), H2.unsqueeze(1)), 1)        
    
    if img_patches:
        if im1_disp is None: im1_disp=im1
        if im2_disp is None: im2_disp=im2
        go_save_patches(im1_disp, im2_disp, pt1, pt2, Hs, w, save_prefix='laf_patch_')    
        go_save_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='laf_patch_')    
  
    return pt1, pt2, Hs


def get_inverse(pt1, pt2, Hs):
    l = Hs.size()[0] 
    Hs1, Hs2 = Hs.split(1, dim=1)
    Hs1 = Hs1.squeeze(1)
    Hs2 = Hs2.squeeze(1)
            
    pt1_ = Hs1.bmm(torch.hstack((pt1, torch.ones((pt1.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt1_ = pt1_[:, :2] / pt1_[:, 2].unsqueeze(-1)
    pt2_ = Hs2.bmm(torch.hstack((pt2, torch.ones((pt2.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt2_ = pt2_[:, :2] / pt2_[:, 2].unsqueeze(-1)
    
    Hi = torch.linalg.inv(Hs.reshape(l*2, 3, 3)).reshape(l, 2, 3, 3)    
    Hi1, Hi2 = Hi.split(1, dim=1)
    Hi1 = Hi1.squeeze(1)
    Hi2 = Hi2.squeeze(1)
    
    return pt1_, pt2_, Hi, Hi1, Hi2


def refinement_norm_corr(im1, im2, pt1, pt2, Hs, w=15, ref_image=['left', 'right'], subpix=True, img_patches=False, save_prefix='ncc_patch_', im1_disp=None, im2_disp=None):    
    l = Hs.size()[0] 
    
    if l==0:
        return pt1, pt2, Hs, torch.zeros(0, device=device), torch.zeros((0, 3, 3), device=device)
            
    pt1_, pt2_, Hi, Hi1, Hi2 = get_inverse(pt1, pt2, Hs)    
                
    patch1 = patchify(im1, pt1_.squeeze(), Hi1, w*2)
    patch2 = patchify(im2, pt2_.squeeze(), Hi2, w*2)

    uidx = torch.arange(w, 3*w + 1)
    tmp = uidx.unsqueeze(0).repeat(w*2 + 1, 1)
    vidx = tmp.permute(1, 0) * (w*4 + 1) + tmp
        
    patch_val = torch.full((2, l), -1, device=device, dtype=torch.float)
    patch_offset = torch.zeros(2, l, 2, device=device)

    if ('left' in ref_image) or ('both' in ref_image):
        patch_offset0, patch_val0 = norm_corr(patch2, patch1.reshape(l, -1)[:, vidx.flatten()].reshape(l, w*2 + 1, w*2 + 1), subpix=subpix)
        patch_offset[0] = patch_offset0
        patch_val[0] = patch_val0

    if ('right' in ref_image) or ('both' in ref_image):
        patch_offset1, patch_val1 = norm_corr(patch1, patch2.reshape(l, -1)[:, vidx.flatten()].reshape(l, w*2 + 1, w*2 + 1), subpix=subpix)
        patch_offset[1] = patch_offset1
        patch_val[1] = patch_val1
        
    val, val_idx = patch_val.max(dim=0)
    
    zidx = (torch.arange(l, device=device) * 4 + (1 - val_idx) * 2).unsqueeze(1).repeat(1,2) + torch.tensor([0, 1], device=device)
    patch_offset = patch_offset.permute(1, 0, 2).flatten()
    patch_offset[zidx.flatten()] = 0
    patch_offset = patch_offset.reshape(l, 2, 2)
    
    pt1_ = pt1_ - patch_offset[:, 0]
    pt2_ = pt2_ - patch_offset[:, 1]

    pt1 = Hi1.bmm(torch.hstack((pt1_, torch.ones((pt1_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze()
    pt1 = pt1[:, :2] / pt1[:, 2].unsqueeze(-1)
    pt2 = Hi2.bmm(torch.hstack((pt2_, torch.ones((pt2_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze()
    pt2 = pt2[:, :2] / pt2[:, 2].unsqueeze(-1)
    
    T = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l*2, 9)
    aux = patch_offset.reshape(l*2, 2)
    T[:, 2] = -aux[:, 0]
    T[:, 5] = -aux[:, 1]
    T = T.reshape(l*2, 3, 3)
    
    if img_patches:
        if im1_disp is None: im1_disp=im1
        if im2_disp is None: im2_disp=im2
        go_save_patches(im1_disp, im2_disp, pt1, pt2, Hs, w, save_prefix=save_prefix)
        go_save_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix=save_prefix)
        
    return pt1, pt2, Hs, val, T


def refinement_norm_corr_alternate(im1, im2, pt1, pt2, Hs, w=15, w_big=None, ref_image=['left', 'right'], angle=[0, ], scale=[[1, 1], ], subpix=True, img_patches=False,  save_prefix='ncc_alternate_patch_', im1_disp=None, im2_disp=None):    
    l = Hs.size()[0] 
    
    if l==0:
        return pt1, pt2, Hs, torch.zeros(0, device=device), torch.zeros((0, 3, 3), device=device)
        
    if w_big is None:
        w_big = w * 2
        
    pt1_, pt2_, Hi, Hi1, Hi2 = get_inverse(pt1, pt2, Hs)    
                        
    patch_val = torch.full((2, l), -torch.inf, device=device, dtype=torch.float)
    patch_offset = torch.zeros(2, l, 2, device=device)
    patch_t = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l, 2, 3, 3)

    a = torch.tensor(angle, device=device) * np.pi / 180
    s = torch.tensor(scale, device=device)

    Ti = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l, 2, 3, 3)
    Ti[:, 0, 0, 2] = pt1_[:, 0]
    Ti[:, 0, 1, 2] = pt1_[:, 1]
    Ti[:, 1, 0, 2] = pt2_[:, 0]
    Ti[:, 1, 1, 2] = pt2_[:, 1]

    T = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l, 2, 3, 3)
    T[:, 0, 0, 2] = -pt1_[:, 0]
    T[:, 0, 1, 2] = -pt1_[:, 1]
    T[:, 1, 0, 2] = -pt2_[:, 0]
    T[:, 1, 1, 2] = -pt2_[:, 1]

    for i in torch.arange(a.shape[0]):
        for j in torch.arange(s.shape[0]):
            R = torch.eye(3, device=device)
            R[0, 0] = torch.cos(a[i])
            R[0, 1] = -torch.sin(a[i])
            R[1, 0] = torch.sin(a[i])
            R[1, 1] = torch.cos(a[i])

            S = torch.eye(3, device=device)
            S[0, 0] = s[j, 0]
            S[1, 1] = s[j, 1]
            
            _, _, Hiu, Hi1u, Hi2u = get_inverse(pt1, pt2, Ti @ S @ R @ T @ Hs)    

            if ('left' in ref_image) or ('both' in ref_image):
                patch2 = patchify(im2, pt2_, Hi2, w_big)
                patch1_small = patchify(im1, pt1_, Hi1u, w)
        
                patch_offset0, patch_val0 = norm_corr(patch2, patch1_small, subpix=subpix)

                mask = patch_val0 > patch_val[0]                
                patch_offset[0, mask] = patch_offset0[mask]
                patch_val[0, mask] = patch_val0[mask]
                patch_t[mask, 0] = Hi1u[mask]                
        
            if ('right' in ref_image) or ('both' in ref_image):
                patch1 = patchify(im1, pt1_, Hi1, w_big)
                patch2_small = patchify(im2, pt2_, Hi2u, w)  
                
                patch_offset1, patch_val1 = norm_corr(patch1, patch2_small, subpix=subpix)
                
                mask = patch_val1 > patch_val[1]                
                patch_offset[1, mask] = patch_offset1[mask]
                patch_val[1, mask] = patch_val1[mask]
                patch_t[mask, 1] = Hi2u[mask]
        
    val, val_idx = patch_val.max(dim=0)
    
    zidx = (torch.arange(l, device=device) * 4 + (1 - val_idx) * 2).unsqueeze(1).repeat(1,2) + torch.tensor([0, 1], device=device)
    patch_offset = patch_offset.permute(1, 0, 2).flatten()
    patch_offset[zidx.flatten()] = 0
    patch_offset = patch_offset.reshape(l, 2, 2)

    pt1_ = pt1_ - patch_offset[:, 0]
    pt2_ = pt2_ - patch_offset[:, 1]

    Hiu = Hi
    valid_idx = val >= 0
    Hiu[(val_idx==0) & valid_idx, 0] = patch_t[(val_idx==0) & valid_idx, 0] 
    Hiu[(val_idx==1) & valid_idx, 1] = patch_t[(val_idx==1) & valid_idx, 1] 
    
    Hsu = torch.linalg.inv(Hiu.reshape(l*2, 3, 3)).reshape(l, 2, 3, 3)        
    
    pt1 = Hiu[:, 0].bmm(torch.hstack((pt1_, torch.ones((pt1_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt1 = pt1[:, :2] / pt1[:, 2].unsqueeze(-1)
    pt2 = Hiu[:, 1].bmm(torch.hstack((pt2_, torch.ones((pt2_.size()[0], 1), device=device))).unsqueeze(-1)).squeeze(-1)
    pt2 = pt2[:, :2] / pt2[:, 2].unsqueeze(-1)
    
    T = torch.eye(3, device=device, dtype=torch.float).reshape(1, 1, 9).repeat(l, 2, 1).reshape(l*2, 9)
    aux = patch_offset.reshape(l*2, 2)
    T[:, 2] = -aux[:, 0]
    T[:, 5] = -aux[:, 1]
    T = T.reshape(l*2, 3, 3)
    
    if img_patches:
        if im1_disp is None: im1_disp=im1
        if im2_disp is None: im2_disp=im2
        go_save_patches(im1_disp, im2_disp, pt1, pt2, Hsu, w, save_prefix=save_prefix)
        go_save_diff_patches(im1, im2, pt1, pt2, Hsu, w, save_prefix=save_prefix)
        
    return pt1, pt2, Hsu, val, T


def go_save_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_diff_', stretch=False):        
    # warning image must be grayscale and not rgb!

    pt1_, pt2_, _, Hi1, Hi2 = get_inverse(pt1, pt2, Hs) 
            
    patch1 = patchify(im1, pt1_, Hi1, w)
    patch2 = patchify(im2, pt2_, Hi2, w)
    
    for k in range(pt1.shape[0]):
        pp = patch1[k]
        pm = torch.isfinite(pp)
        m_ = pp[pm].min()
        M_ = pp[pm].max()
        pp[pm] = (pp[pm] - m_) / (M_ - m_)            
        patch1[k] = pp * 255        
    
        pp = patch2[k]
        pm = torch.isfinite(pp)
        m_ = pp[pm].min()
        M_ = pp[pm].max()
        pp[pm] = (pp[pm] - m_) / (M_ - m_)            
        patch2[k] = pp * 255       
    
    mask1 = torch.isfinite(patch1) & (~torch.isfinite(patch2))
    patch2[mask1] = 0

    mask2 = torch.isfinite(patch2) & (~torch.isfinite(patch1))
    patch1[mask2] = 0

    both_patches = torch.zeros((3, patch1.shape[0], patch1.shape[1], patch1.shape[2]), dtype=torch.float32, device=device)
    both_patches[0] = patch1
    both_patches[1] = patch2

    save_patch(both_patches, save_prefix=save_prefix, save_suffix='.png', stretch=stretch)


def go_save_list_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_list_diff_', remove_same=False, bar_idx=None, bar_width=2, stretch=False):        
    # warning image must be grayscale and not rgb!

    ww = w * 2 + 1
    l = len(pt1)
    n = pt1[0].shape[0]
    if bar_idx is None: bar_width = 0    
    patch_list = torch.zeros((n, l, ww, bar_width + ww, 3), dtype=torch.float32, device=device)                
    for i in range(l):
        pt1_, pt2_, _, Hi1, Hi2 = get_inverse(pt1[i], pt2[i], Hs[i]) 
            
        patch1 = patchify(im1, pt1_, Hi1, w)
        patch2 = patchify(im2, pt2_, Hi2, w)
        
        for k in range(n):
            pp = patch1[k]
            pm = torch.isfinite(pp)
            m_ = pp[pm].min()
            M_ = pp[pm].max()
            pp[pm] = (pp[pm] - m_) / (M_ - m_)            
            patch1[k] = pp * 255        
        
            pp = patch2[k]
            pm = torch.isfinite(pp)
            m_ = pp[pm].min()
            M_ = pp[pm].max()
            pp[pm] = (pp[pm] - m_) / (M_ - m_)            
            patch2[k] = pp * 255        
    
        mask1 = torch.isfinite(patch1) & (~torch.isfinite(patch2))
        patch2[mask1] = 0

        mask2 = torch.isfinite(patch2) & (~torch.isfinite(patch1))
        patch1[mask2] = 0

        both_patches = torch.zeros((patch1.shape[0], patch1.shape[1], patch1.shape[2] + bar_width, 3), dtype=torch.float32, device=device)
        both_patches[:, :, bar_width:, 0] = patch1
        both_patches[:, :, bar_width:, 1] = patch2        
        
        if not (bar_idx is None):
            patch_idx = torch.zeros((patch1.shape[0], patch1.shape[1], bar_width, 3), dtype=torch.float32, device=device)
            patch_idx[:, :, :, 2] = 128 # more stylish but less visible
            patch_idx[:, :, :] = 255
            for k in range(n):
                ll = torch.clamp(ww - bar_idx[i, k], 0, ww).type(torch.int16)
                # ll = torch.clamp((1 - bar_idx[i, k]) * ww, 0, ww).type(torch.int16)
                patch_idx[k, ll:ww, :, 2] = 255 # more stylish but less visible
                patch_idx[k, ll:ww, :, :2] = 0
        both_patches[:, :, :bar_width, :] = patch_idx

        patch_list[:, i, :, :, :] = both_patches                
            
    if remove_same:
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for k in range(n):
            mask[k] = True
            for i in range(l-1):
                if not torch.all(patch_list[k, i, :, :, :].type(torch.uint8) == patch_list[k, i+1, :, :, :].type(torch.uint8)):
                    mask[k] == False
                    break

        patch_list = patch_list[mask]

    patch_list = patch_list.reshape((n, l*ww, ww + bar_width, 3)).permute((-1, 0, 1, 2))

    save_patch(patch_list, grid=[50//l, 50], save_prefix=save_prefix, save_suffix='.png', stretch=stretch)


def go_save_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_', stretch=False):        
    pt1_, pt2_, _, Hi1, Hi2 = get_inverse(pt1, pt2, Hs) 
            
    patch1 = patchify(im1, pt1_, Hi1, w)
    patch2 = patchify(im2, pt2_, Hi2, w)

    save_patch(patch1, save_prefix=save_prefix, save_suffix='_a.png', stretch=stretch)
    save_patch(patch2, save_prefix=save_prefix, save_suffix='_b.png', stretch=stretch)


def refinement_miho(im1, im2, pt1, pt2, mihoo=None, Hs_laf=None, remove_bad=True, w=15, img_patches=False, also_laf=False, im1_disp=None, im2_disp=None):
    l = pt1.shape[0]
    idx = torch.ones(l, dtype=torch.bool, device=device)

    if mihoo is None:
        if Hs_laf is not None:
            if also_laf:
                return pt1, pt2, Hs_laf, idx, None
            else:
                return pt1, pt2, Hs_laf, idx
        else:
            Hs = torch.eye(3, device=device).repeat(l*2, 1).reshape(l, 2, 3, 3)
            if also_laf:
                return pt1, pt2, Hs, idx, None
            else:
                return pt1, pt2, Hs, idx

    Hs = torch.zeros((l, 2, 3, 3), device=device)
    for i in range(l):
        ii = mihoo.Hidx[i]
        if ii > -1:
            Hs[i, 0] = mihoo.Hs[ii][0]
            Hs[i, 1] = mihoo.Hs[ii][1]
        elif Hs_laf is not None:
            Hs[i, 0] = Hs_laf[i, 0]
            Hs[i, 1] = Hs_laf[i, 1]
        else:
            Hs[i, 0] = torch.eye(3, device=device)
            Hs[i, 1] = torch.eye(3, device=device)
            
    if remove_bad:
        mask = mihoo.Hidx > -1
        pt1 = pt1[mask]
        pt2 = pt2[mask]
        Hs = Hs[mask]
        idx = mask
        
        if also_laf and (Hs_laf is not None):
            Hs_laf = Hs_laf[mask]
        
    if img_patches:
        if im1_disp is None: im1_disp=im1
        if im2_disp is None: im2_disp=im2
        go_save_patches(im1_disp, im2_disp, pt1, pt2, Hs, w, save_prefix='miho_patch_')
        go_save_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='miho_patch_')
    
    if not also_laf:
        return pt1, pt2, Hs, idx
    else:
        return pt1, pt2, Hs, idx, Hs_laf


def refinement_miho_other(im1, im2, pt1, pt2, mihoo=None, Hs_laf=None, remove_bad=True, w=15, patch_ref='left', img_patches=False, also_laf=False, im1_disp=None, im2_disp=None):
    l = pt1.shape[0]
    idx = torch.ones(l, dtype=torch.bool, device=device)

    if mihoo is None:
        if Hs_laf is not None:
            if also_laf:
                return pt1, pt2, Hs_laf, idx, None
            else:
                return pt1, pt2, Hs_laf, idx
        else:
            Hs = torch.eye(3, device=device).repeat(l*2, 1).reshape(l, 2, 3, 3)
            if also_laf:
                return pt1, pt2, Hs, idx, None
            else:
                return pt1, pt2, Hs, idx

    Hs = torch.zeros((l, 2, 3, 3), device=device)
    for i in range(l):
        ii = mihoo.Hidx[i]
        if ii > -1:
            if not (patch_ref=='left'):
                Hs[i, 0] = mihoo.Hs[ii][0]
                Hs[i, 1] = torch.eye(3, device=device)
            else:
                Hs[i, 0] = torch.eye(3, device=device)
                Hs[i, 1] = mihoo.Hs[ii][0].inverse()
        elif Hs_laf is not None:
            Hs[i, 0] = Hs_laf[i, 0]
            Hs[i, 1] = Hs_laf[i, 1]
        else:
            Hs[i, 0] = torch.eye(3, device=device)
            Hs[i, 1] = torch.eye(3, device=device)
            
    if remove_bad:
        mask = mihoo.Hidx > -1
        pt1 = pt1[mask]
        pt2 = pt2[mask]
        Hs = Hs[mask]
        idx = mask
        
        if also_laf and (Hs_laf is not None):
            Hs_laf = Hs_laf[mask]
        
    if img_patches:
        if im1_disp is None: im1_disp=im1
        if im2_disp is None: im2_disp=im2
        go_save_patches(im1_disp, im2_disp, pt1, pt2, Hs, w, save_prefix='miho_patch_')
        go_save_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='miho_patch_')

    if not also_laf:
        return pt1, pt2, Hs, idx
    else:
        return pt1, pt2, Hs, idx, Hs_laf


def norm_corr(patch1, patch2, subpix=True):     
    w = patch2.size()[1]
    ww = w * w
    n = patch1.size()[0]
    
    with torch.no_grad():
        conv_ = torch.nn.Conv2d(1, 1, (w, w), padding='valid', bias=False, device=device)
        conv_.weight = torch.nn.Parameter(torch.ones((1, 1, w, w), device=device).to(torch.float))    
        m1 = conv_(patch1.unsqueeze(1)).squeeze()
        e1 = conv_((patch1**2).unsqueeze(1)).squeeze()
        s1 = ww * e1 - m1**2
    
    m2 = patch2.sum(dim=[1, 2])
    e2 = (patch2**2).sum(dim=[1, 2])    
    s2 = ww * e2 - m2**2
    
    with torch.no_grad():
        conv__ = torch.nn.Conv2d(n, n, (w, w), padding='valid', bias=False, groups=n, device=device)
        conv__.weight = torch.nn.Parameter(patch2.unsqueeze(1))    
        cc = conv__(patch1.unsqueeze(0)).squeeze()

    nc = ((ww * cc) - (m1 * m2.reshape(n, 1, 1))) / torch.sqrt(s1 * s2.reshape(n, 1, 1))   
    nc.flatten()[~torch.isfinite(nc.flatten())] = -torch.inf

    w_ = nc.shape[1]
    ww_ = w_ * w_    
    r = (w_ - 1) / 2

    idx = nc.reshape(n, ww_).max(dim=1)
    offset = (torch.vstack((idx[1] % w_, torch.div(idx[1], w_, rounding_mode='trunc')))).permute(1, 0).to(torch.float)
    
    if subpix:    
        t = ((offset > 0) & ( offset < w_ - 1)).all(dim=1).to(torch.float)
        tidx = (torch.tensor([-1, 0, 1], device=device).unsqueeze(0) * t.unsqueeze(1)).squeeze()
    
        tx = offset[:, 0].unsqueeze(1) + tidx
        v = nc.flatten()[(torch.arange(n, device=device).unsqueeze(1) * ww_ + offset[:, 1].unsqueeze(1) * w_ + tx).to(torch.long).flatten()].reshape(n, 3)
        sx = (v[:, 2] - v[:, 0]) / (2 * (2 * v[:, 1] - v[:, 0] - v[:, 2]))
        sx[~sx.isfinite()] = 0
    
        ty = offset[:, 1].unsqueeze(1) + tidx
        v = nc.flatten()[(torch.arange(n, device=device).unsqueeze(1) * ww_ + ty * w_ + offset[:, 0].unsqueeze(1)).to(torch.long).flatten()].reshape(n, 3)
        sy = (v[:, 2] - v[:, 0]) / (2 * (2 * v[:, 1] - v[:, 0] - v[:, 2]))
        sy[~sy.isfinite()] = 0
        
        offset[:, 0] = offset[:, 0] + sx
        offset[:, 1] = offset[:, 1] + sy

    offset -= r
    offset[~torch.isfinite(idx[0])] = 0

    return offset, idx[0]


def save_patch(patch, grid=[40, 50], save_prefix='patch_', save_suffix='.png', normalize=False, stretch=True):
    if patch.ndim==3:
        patch = patch.unsqueeze(0)
    cc = patch.shape[0]    

    grid_el = grid[0] * grid[1]
    l = patch.shape[1]
    n = patch.shape[2]
    m = patch.shape[3]
    transform = transforms.ToPILImage()
    for i in range(0, l, grid_el):
        j = min(i + grid_el, l)
        filename = f'{save_prefix}{i}_{j}{save_suffix}' 

        if not stretch:
            grid0 = grid[0]
            grid1 = grid[1]
        else:
            grid0 = (j - i) / grid[1]
            if grid0 >= 1:
                grid1 = grid[1]
            else:
                grid1 = (j - i) % grid[1]
            grid0 = int(np.ceil(grid0))
            
        grid_el_ = grid0 * grid1
        
        patch_ = patch[:, i:j]
        aux = torch.zeros((cc, grid_el_, n, m), dtype=torch.float32, device=device)
        aux[:, :j-i] = patch_
        
        mask = aux[0].isfinite()
        aux[:, ~mask] = 0
        
        if not normalize:
            aux = aux.type(torch.uint8)
        else:
            for ci in range(cc):
                aux[ci, ~mask] = -1        
                avg = ((mask * aux[ci]).sum(dim=(1,2)) / mask.sum(dim=(1,2))).reshape(-1, 1, 1).repeat(1, n, m)
                avg[mask] = aux[ci, mask]
                m_ = avg.reshape(grid_el_, -1).min(dim=1)[0]
                M_ = avg.reshape(grid_el_, -1).max(dim=1)[0]
                aux[ci] = (((aux[ci] - m_.reshape(-1, 1, 1)) / (M_ - m_).reshape(-1, 1, 1)) * 255).type(torch.uint8)
           
        # if not needed do not add alpha channel
        all_mask = mask.all()
        
        c_final = cc
        if (~all_mask) and (cc==3): c_final = 4
        if (~all_mask) and (cc==1): c_final = 4

        im = torch.zeros((c_final, grid0 * n, grid1 * m), dtype=torch.uint8, device=device)
        
        aux = aux.reshape(cc, grid0, grid1, n, m).permute(0, 1, 3, 2, 4).reshape(cc, grid0 * n, grid1 * m).contiguous()
        if (~all_mask) and (cc==1): aux = aux.repeat(c_final, 1, 1)
        im[:aux.shape[0]] = aux

        if not all_mask:        
            im[3, :, :] = (mask *255).type(torch.uint8).reshape(grid0, grid1, n, m).permute(0, 2, 1, 3).reshape(grid0 * n, grid1 * m).contiguous()

        transform(im).save(filename)
        

def patchify(img, pts, H, r):

    wi = torch.arange(-r,r+1, device=device)
    ws = r * 2 + 1
    n = pts.size()[0]
    cc, y_sz, x_sz = img.size()
    
    x, y = pts.split(1, dim=1)
    
    widx = torch.zeros((n, 3, ws**2), dtype=torch.float, device=device)
    
    widx[:, 0] = (wi + x).repeat(1,ws)
    widx[:, 1] = (wi + y).repeat_interleave(ws, dim=1)
    widx[:, 2] = 1

    nidx = torch.matmul(H, widx)
    xx, yy, zz = nidx.split(1, dim=1)
    zz_ = zz.squeeze()
    xx_ = xx.squeeze() / zz_
    yy_ = yy.squeeze() / zz_
    
    xf = xx_.floor().type(torch.long)
    yf = yy_.floor().type(torch.long)
    xc = xf + 1
    yc = yf + 1

    nidx_mask = ~torch.isfinite(xx_) | ~torch.isfinite(yy_) | (xf < 0) | (yf < 0) | (xc >= x_sz) | (yc >= y_sz)

    xf[nidx_mask] = 0
    yf[nidx_mask] = 0
    xc[nidx_mask] = 0
    yc[nidx_mask] = 0

    patch = torch.zeros((cc, xx.shape[0],ws,ws), device=device, dtype=torch.float32)    

    for ci in range(cc):
        # for mask
        img_ = img[ci].flatten()
        aux = img_[0].clone()
        img_[0] = float('nan')
    
        a = xx_-xf    
        b = yy_-yf
        c = xc-xx_    
        d = yc-yy_
    
        patch[ci] = (a * (b * img_[yc * x_sz + xc] + d * img_[yf * x_sz + xc]) + c * (b * img_[yc * x_sz + xf] + d * img_[yf * x_sz + xf])).reshape((-1, ws, ws))
        img_[0] = aux

    return patch.squeeze(0)


# previous version without rgb handling
# def save_patch(patch, grid=[40, 50], save_prefix='patch_', save_suffix='.png', normalize=False):
#
#     grid_el = grid[0] * grid[1]
#     l = patch.size()[0]
#     n = patch.size()[1]
#     m = patch.size()[2]
#     transform = transforms.ToPILImage()
#     for i in range(0, l, grid_el):
#         j = min(i+ grid_el, l)
#         filename = f'{save_prefix}{i}_{j}{save_suffix}' 
#       
#         patch_ = patch[i:j]
#         aux = torch.zeros((grid_el, n, m), dtype=torch.float32, device=device)
#         aux[:j-i] = patch_
#       
#         mask = aux.isfinite()
#         aux[~mask] = 0
#       
#         if not normalize:
#             aux = aux.type(torch.uint8)
#         else:
#             aux[~mask] = -1        
#             avg = ((mask * aux).sum(dim=(1,2)) / mask.sum(dim=(1,2))).reshape(-1, 1, 1).repeat(1, n, m)
#             avg[mask] = aux[mask]
#             m_ = avg.reshape(grid_el, -1).min(dim=1)[0]
#             M_ = avg.reshape(grid_el, -1).max(dim=1)[0]
#             aux = (((aux - m_.reshape(-1, 1, 1)) / (M_ - m_).reshape(-1, 1, 1)) * 255).type(torch.uint8)
#          
#         # if not needed do not add alpha channel
#         all_mask = mask.all()
#         c = 1 + (3 * ~all_mask)
#         aux = aux.reshape(grid[0], grid[1], n, m).permute(0, 2, 1, 3).reshape(grid[0] * n, grid[1] * m).contiguous().unsqueeze(0).repeat(c, 1, 1)
#         if not all_mask:        
#             aux[3, :, :] = (mask *255).type(torch.uint8).reshape(grid[0], grid[1], n, m).permute(0, 2, 1, 3).reshape(grid[0] * n, grid[1] * m).contiguous()
#         transform(aux).save(filename)
        

# def patchify(img, pts, H, r):
#
#     wi = torch.arange(-r,r+1, device=device)
#     ws = r * 2 + 1
#     n = pts.size()[0]
#     _, y_sz, x_sz = img.size()
#    
#     x, y = pts.split(1, dim=1)
#    
#     widx = torch.zeros((n, 3, ws**2), dtype=torch.float, device=device)
#    
#     widx[:, 0] = (wi + x).repeat(1,ws)
#     widx[:, 1] = (wi + y).repeat_interleave(ws, dim=1)
#     widx[:, 2] = 1
#
#     nidx = torch.matmul(H, widx)
#     xx, yy, zz = nidx.split(1, dim=1)
#     zz_ = zz.squeeze()
#     xx_ = xx.squeeze() / zz_
#     yy_ = yy.squeeze() / zz_
#    
#     xf = xx_.floor().type(torch.long)
#     yf = yy_.floor().type(torch.long)
#     xc = xf + 1
#     yc = yf + 1
#
#     nidx_mask = ~torch.isfinite(xx_) | ~torch.isfinite(yy_) | (xf < 0) | (yf < 0) | (xc >= x_sz) | (yc >= y_sz)
#
#     xf[nidx_mask] = 0
#     yf[nidx_mask] = 0
#     xc[nidx_mask] = 0
#     yc[nidx_mask] = 0
#
#     # for mask
#     img_ = img.flatten()
#     aux = img_[0].clone()
#     img_[0] = float('nan')
#
#     a = xx_-xf    
#     b = yy_-yf
#     c = xc-xx_    
#     d = yc-yy_
#
#     patch = (a * (b * img_[yc * x_sz + xc] + d * img_[yf * x_sz + xc]) + c * (b * img_[yc * x_sz + xf] + d * img_[yf * x_sz + xf])).reshape((-1, ws, ws))
#     img_[0] = aux
#
#     return patch


class ncc_module:
    def __init__(self, **args):
        self.w = 10
        self.w_big = None
        self.angle = [-30, -15, 0, 15, 30]
        self.scale = [[10/14, 1], [10/12, 1], [1, 1], [1, 12/10], [1, 14/10]]
        self.subpix = True
        self.ref_images = 'both'
        self.also_prev = False
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 
        
        for k, v in args.items():
           setattr(self, k, v)
        
        
    def get_id(self):
        return ('nnc_subpix_' + str(self.subpix) + '_w_' + str(self.w) + '_w_big_' + str(self.w_big) + '_ref_images_' + str(self.ref_images) + '_scales_' +  str(len(self.scale)) + '_angles_' + str(len(self.scale))).lower()

    
    def run(self, **args):
        im1 = Image.open(args['im1'])
        im2 = Image.open(args['im2'])

        im1 = self.transform(im1).type(torch.float16).to(device)
        im2 = self.transform(im2).type(torch.float16).to(device)        
        
        pt1, pt2, Hs_ncc, val, T = refinement_norm_corr_alternate(im1, im2, args['pt1'], args['pt2'], args['Hs'], w=self.w, w_big=self.w_big, ref_image=[self.ref_images], angle=self.angle, scale=self.scale, subpix=self.subpix, img_patches=False)   

        laf_is_better = np.nan
        if self.also_prev and ('Hs_prev' in args.keys()) and (args['Hs'].size()[0] > 0):
            pt1_, pt2_, Hs_ncc_, val_, T_ = refinement_norm_corr_alternate(im1, im2, args['pt1'], args['pt2'], args['Hs_prev'], w=self.w, w_big=self.w_big, ref_image=[self.ref_images], angle=[0, ], scale=[[1, 1], ], subpix=self.subpix, img_patches=False)   
            replace_idx = torch.argwhere((torch.cat((val.unsqueeze(0),val_.unsqueeze(0)), dim=0)).max(dim=0)[1] == 1)
            pt1[replace_idx] = pt1_[replace_idx]
            pt2[replace_idx] = pt2_[replace_idx]
            Hs_ncc[replace_idx] = Hs_ncc_[replace_idx]
            val[replace_idx] = val_[replace_idx]
            T[replace_idx] = T_[replace_idx]
            laf_is_better = replace_idx.shape[0] / pt1_.shape[0] 
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_ncc, 'val': val, 'T': T, 'laf_is_better': laf_is_better}
