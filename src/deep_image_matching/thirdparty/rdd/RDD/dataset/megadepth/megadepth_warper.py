import torch
from kornia.utils import create_meshgrid
import matplotlib.pyplot as plt
import pdb
from .utils import warp

@torch.no_grad()
def spvs_coarse(data, scale = 8):
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    device = data['image0'].device
    corrs = []
    for idx in range(N):
        warp01_params = {}
        for k, v in data['warp01_params'].items():
            if isinstance(v[idx], torch.Tensor):
                warp01_params[k] = v[idx].to(device)
            else:
                warp01_params[k] = v[idx]
        warp10_params = {}
        for k, v in data['warp10_params'].items():
            if isinstance(v[idx], torch.Tensor):
                warp10_params[k] = v[idx].to(device)
            else:
                warp10_params[k] = v[idx]
            
        # create kpts
        h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
        grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(h1*w1, 2)    # [N, hw, 2]
        
        # normalize kpts
        grid_pt1_c = grid_pt1_c * scale

        

        try:
            grid_pt1_c_valid, grid_pt10_c, ids1, ids1_out = warp(grid_pt1_c, warp10_params)
            grid_pt10_c_valid, grid_pt01_c, ids0, ids0_out = warp(grid_pt10_c, warp01_params)
            
            # check reproj error
            grid_pt1_c_valid = grid_pt1_c_valid[ids0]
            dist = torch.linalg.norm(grid_pt1_c_valid - grid_pt01_c, dim=-1)
            
            mask_mutual = (dist < 1.5) 
            
            #get correspondences
            pts = torch.cat([grid_pt10_c_valid[mask_mutual] / scale,
                                grid_pt01_c[mask_mutual] / scale], dim=-1)
            #remove repeated correspondences
            lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = torch.float32) * -1
            lut_mat21 = torch.clone(lut_mat12)
            src_pts = pts[:, :2]
            tgt_pts = pts[:, 2:]
        
            lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
            points = lut_mat12[mask_valid12]

            #Target-src check
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            lut_mat21[tgt_pts[:,1].long(), tgt_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
            points = lut_mat21[mask_valid21]

            corrs.append(points)
        except:
            corrs.append(torch.zeros((0, 4), device = device))
            #pdb.set_trace()
            #print('..')

    #Plot for debug purposes    
    # for i in range(len(corrs)):
    #     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2]*8, corrs[i][:, 2:]*8)

    return corrs