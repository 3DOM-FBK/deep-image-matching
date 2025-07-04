from PIL import Image
import torch
import kornia as K
import src.ncc as ncc
import src.base_modules as base_pipe
import src.miho as miho_duplex 
import src.bench_utils as bench
import numpy as np
import os
import shutil
import src.DeDoDe2.dedode2_custom as dedode2
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__': 
    pipe_bases = [
        base_pipe.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99),
        base_pipe.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True),
        base_pipe.lightglue_module(num_features=8000, upright=True, what='superpoint'),
        base_pipe.lightglue_module(num_features=8000, upright=True, what='aliked'),
        base_pipe.lightglue_module(num_features=8000, upright=True, what='disk'),  
        base_pipe.loftr_module(num_features=8000, upright=True),        
        dedode2.dedode2_module(num_features=8000, upright=True), 
    ]
    pipe_miho =  miho_duplex.miho_module()
    pipe_ncc = ncc.ncc_module(also_prev=True)

    bench_path = '../bench_data' # results will be in the subfolder "patches" 
    bench_im='imgs'  
    save_to = 'res'
    w = 10 # patch radius to display

    how_many = -1 # number of (random chosen) patches to show, set to -1 to all patches
    stretch = False # if True removes black unneeded background in patch images
    flat_folder = False # single folder with long path as filename if True

    # # paper table 
    # how_many = 15
    # stretch = True
    # flat_folder = True

    tg = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 

    t = transforms.Compose([
            transforms.PILToTensor() 
            ]) 

    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False, 'index': [1489, 281]},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False, 'index': [0]},
            'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False, 'index': [86]},
          # 'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True, 'index': [0]},
        }

    for pipe_base in pipe_bases:
        pipe_name = pipe_base.get_id()
        
        for b in benchmark_data:    
            b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)

            # select image pair subset
            b_index = benchmark_data[b]['index']
            if not (b_index is None):
                for bname in b_data.keys():
                    if isinstance(b_data[bname], list): b_data[bname] = [b_data[bname][bb] for bb in b_index]
                    if isinstance(b_data[bname], np.ndarray): b_data[bname] = b_data[bname][b_index]
        
            n = len(b_data['im1'])
            ext = benchmark_data[b]['ext']
            im_path = os.path.join(bench_im, benchmark_data[b]['name'])        

            for i in range(n):   
                if b_index is None: 
                    ii = i
                else:
                    ii = b_index[i]
                                        
                if flat_folder == False:
                    base_prefix = os.path.join(bench_path, 'patches', benchmark_data[b]['name'], pipe_name, str(ii))
                    img_base_prefix = os.path.join(bench_path, 'patches', benchmark_data[b]['name'], 'image_pairs', str(ii))
                    name_prefix = ''
                    img_name_prefix = ''                
                else:
                    base_prefix = os.path.join(bench_path, 'patches')                
                    img_base_prefix = base_prefix
                    name_prefix = benchmark_data[b]['name'] + '_' + str(ii) + '_' + pipe_name + '_'                
                    img_name_prefix = benchmark_data[b]['name'] + '_' + str(ii) + '_'                
                
                os.makedirs(base_prefix, exist_ok=True)
                os.makedirs(img_base_prefix, exist_ok=True)

                im1 = os.path.join(bench_path, im_path, os.path.splitext(b_data['im1'][i])[0]) + ext
                im2 = os.path.join(bench_path, im_path, os.path.splitext(b_data['im2'][i])[0]) + ext
                
                shutil.copyfile(im1, os.path.join(img_base_prefix, img_name_prefix + 'im1' + ext))
                shutil.copyfile(im2, os.path.join(img_base_prefix, img_name_prefix + 'im2' + ext))
                  
                if benchmark_data[b]['is_not_planar'] == True:                
                    K1 = b_data['K1']
                    K2 = b_data['K2']
                    R_gt = b_data['R']
                    t_gt = b_data['T']            
        
                    F_gt = torch.tensor(K2[i].T, device=device, dtype=torch.float64).inverse() @ \
                           torch.tensor([[0, -t_gt[i][2], t_gt[i][1]],
                                        [t_gt[i][2], 0, -t_gt[i][0]],
                                        [-t_gt[i][1], t_gt[i][0], 0]], device=device) @ \
                           torch.tensor(R_gt[i], device=device) @ \
                           torch.tensor(K1[i], device=device, dtype=torch.float64).inverse()
                    F_gt = F_gt / F_gt.sum()
                    F_gt = F_gt
                else:
                    rad = 15                    
                    H_gt = torch.tensor(b_data['H'][i], device=device)
                    H_inv_gt = torch.tensor(b_data['H_inv'][i], device=device)

                pipe_data_im = {'im1': im1, 'im2': im2}
                pipe_data_base = pipe_base.run(**pipe_data_im)
                            
                for k, v in pipe_data_im.items():
                    pipe_data_base[k] = v
    
                pipe_data_base_miho = pipe_miho.run(**pipe_data_base)
    
                for k, v in pipe_data_im.items():
                    pipe_data_base_miho[k] = v
        
                mask = pipe_data_base_miho['mask']
                for k in ['Hs','kp1','kp2','pt1','pt2','val']:
                    if k in pipe_data_base.keys(): pipe_data_base[k] = pipe_data_base[k][mask]
    
                pipe_data_base_ncc = pipe_ncc.run(**pipe_data_base)
                pipe_data_base_miho_ncc = pipe_ncc.run(**pipe_data_base_miho)            
    
                im1 = Image.open(pipe_data_base['im1'])
                im2 = Image.open(pipe_data_base['im2'])
    
                im1c = t(im1).type(torch.float16).to(device)
                im2c = t(im2).type(torch.float16).to(device)     
    
                im1g = tg(im1).type(torch.float16).to(device)
                im2g = tg(im2).type(torch.float16).to(device)
    
                # split patches according to
                mask = {
                    # 'all': torch.full((pipe_data_base['pt1'].shape[0], ), 1 , device=device, dtype=torch.bool),
                    'best_miho': pipe_data_base_miho_ncc['val'] > pipe_data_base_ncc['val'],
                    'best_base': pipe_data_base_ncc['val'] > pipe_data_base_miho_ncc['val'],
                    'equal': pipe_data_base_ncc['val'] == pipe_data_base_miho_ncc['val'],
                    }
    
                for mk in mask.keys():
                    mm = mask[mk]
                                                    
                    pp_list = {'base': pipe_data_base, 'base_ncc': pipe_data_base_ncc, 'base_miho': pipe_data_base_miho, 'base_miho_ncc': pipe_data_base_miho_ncc}
                    err_list = []
                    pt1_list = []
                    pt2_list = []
                    Hs_list = []
                    
                    for pk in pp_list.keys():
                        d = pp_list[pk]                
                        pt1 = d['pt1'][mm]
                        pt2 = d['pt2'][mm]
                        Hs = d['Hs'][mm]
                        
                        if benchmark_data[b]['use_scale']:
                            scales = b_data['im_pair_scale'][i]
                        else:
                            scales = np.asarray([[1.0, 1.0], [1.0, 1.0]])                        
        
                        nn = pt1.shape[0]
                        
                        spt1 = pt1 * torch.tensor(scales[0], device=device)
                        spt2 = pt2 * torch.tensor(scales[1], device=device)

                        pt1_ = torch.vstack((torch.clone(spt1.T), torch.ones((1, nn), device=device))).type(torch.float64)
                        pt2_ = torch.vstack((torch.clone(spt2.T), torch.ones((1, nn), device=device))).type(torch.float64)

                        if benchmark_data[b]['is_not_planar'] == True:  
                            
                            l1_ = F_gt @ pt1_
                            d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()
                            
                            l2_ = F_gt.T @ pt2_
                            d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()                                                                                                                    
                        else:                           
                            pt1_reproj = H_gt @ pt1_
                            pt1_reproj = pt1_reproj[:2] / pt1_reproj[2].unsqueeze(0)
                            d1 = ((pt2_[:2] - pt1_reproj)**2).sum(0).sqrt()
                            
                            pt2_reproj = H_inv_gt @ pt2_
                            pt2_reproj = pt2_reproj[:2] / pt2_reproj[2].unsqueeze(0)
                            d2 = ((pt1_[:2] - pt2_reproj)**2).sum(0).sqrt()
                            
                            valid_matches = torch.ones(nn, device=device, dtype=torch.bool)
                            
                            if b_data['im1_use_mask'][i]:
                                valid_matches = valid_matches & ~bench.invalid_matches(b_data['im1_mask'][i], b_data['im2_full_mask'][i], spt1, spt2, rad)
        
                            if b_data['im2_use_mask'][i]:
                                valid_matches = valid_matches & ~bench.invalid_matches(b_data['im2_mask'][i], b_data['im1_full_mask'][i], spt2, spt1, rad)

                            d1[~valid_matches] = np.nan
                            d2[~valid_matches] = np.nan

                        pt1_list.append(pt1) 
                        pt2_list.append(pt2)                           
                        Hs_list.append(Hs)
                        err_list.append(torch.maximum(d1, d2))

                    # remove outliers according to the GT
                    err_min = torch.min(torch.cat([err_list[pi].unsqueeze(0) for pi in range(len(pp_list))]), dim=0)[0]
                    inl_mask = (err_min < 10) & torch.isfinite(err_min)
                        
                    for pi in range(len(pp_list.keys())):
                        pt1 = pt1_list[pi][inl_mask]
                        pt2 = pt2_list[pi][inl_mask]
                        Hs = Hs_list[pi][inl_mask]
                        
                        # # uncomment to save patch pair differences
                        # ncc.go_save_diff_patches(im1g, im2g, pt1, pt2, Hs, w, save_prefix=os.path.join(base_prefix, name_prefix + mk + '_patch_' + pk + '_'))
    
                        pt1_list[pi] = pt1 
                        pt2_list[pi] = pt2                           
                        Hs_list[pi] = Hs
                        err_list[pi] = err_list[pi][inl_mask]  
                        
                    # 1 px epi error will count 2 px in the blue bar
                    err_idx = torch.cat([err_list[pi].unsqueeze(0) for pi in range(len(pp_list))]) * 2
                    
                    if how_many != -1:
                        pidx = torch.randperm(pt1_list[0].shape[0])[:how_many]
                        
                        err_idx = err_idx[:, pidx]
                        pt1_list = [pt1_list[pi][pidx] for pi in range(len(pp_list))]
                        pt2_list = [pt2_list[pi][pidx] for pi in range(len(pp_list))]
                        Hs_list = [Hs_list[pi][pidx] for pi in range(len(pp_list))]
                                            
                    ncc.go_save_list_diff_patches(im1g, im2g, pt1_list, pt2_list, Hs_list, w, save_prefix=os.path.join(base_prefix, name_prefix + mk + '_patch_list_'), bar_idx=err_idx, stretch=stretch)

                
