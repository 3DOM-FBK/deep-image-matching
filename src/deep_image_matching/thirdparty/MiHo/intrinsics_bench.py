import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import _pickle as cPickle
import bz2
import csv
import scipy.stats as ss
import src.bench_utils as bench

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data, add_ext=False):
    if add_ext:
        ext = '.pbz2'
    else:
        ext = ''
        
    with bz2.BZ2File(title + ext, 'w') as f: 
        cPickle.dump(data, f)
        

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


if __name__ == '__main__':  
    force = False
    bench_path = '../bench_data'
    fig_dpi = 300
    
    # data_<scene>
    # [focal_length, cx, cy, width, heigth]
    # cx ~=  width / 2
    # cy ~= height / 2

    # # not needed if launched run_bench.py before     
    # benchmark_data = {
    #         'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False},
    #         'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False},
    #         'planar': {'name': 'planar', 'Name': 'Planar', 'setup': bench.planar_bench_setup, 'is_outdoor': True, 'is_not_planar': False, 'ext': '.png', 'use_scale': False, 'also_metric': False},
    #         'imc_phototourism': {'name': 'imc_phototourism', 'Name': 'IMC PhotoTourism', 'setup': bench.imc_phototourism_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.jpg', 'use_scale': False, 'also_metric': True},
    #     }

    for b in benchmark_data.keys():
        b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)
    
    if os.path.isfile(os.path.join(bench_path, 'intrinsics_stats.pbz2')) and not force:
        data_megadepth, data_scannet, data_imc, data_megadepth_a, data_scannet_a, data_imc_a, data_megadepth_b, data_scannet_b, data_imc_b = decompress_pickle(os.path.join(bench_path,'intrinsics_stats.pbz2'))
    else:    
        # megadepth
        ppath= os.path.join(bench_path, 'gt_data', 'megadepth')
        npz_list = [i for i in os.listdir(ppath) if (os.path.splitext(i)[1] == '.npz')]
    
        data_megadepth = {}
        data_megadepth_a = []
        data_megadepth_b = []
        for name in npz_list:
            scene_info = np.load(os.path.join(ppath, name), allow_pickle=True)
        
            # collect pairs
            for pair_info in scene_info['pair_infos']:
                (id1, id2), overlap, _ = pair_info
    
                im1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
                K1 = scene_info['intrinsics'][id1].astype(np.float32)
                ori_im1= os.path.join(bench_path, 'megadepth_test_1500/Undistorted_SfM', im1)
                img1 = cv2.imread(ori_im1)
                sz1_ori = np.array(img1.shape)[:2][::-1]
                data_megadepth[im1] = [K1[1, 1], K1[0, 2], K1[1, 2], sz1_ori[0], sz1_ori[1]]
                data_megadepth_a.append(data_megadepth[im1])
    
                im2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')                        
                K2 = scene_info['intrinsics'][id2].astype(np.float32)
                ori_im2= os.path.join(bench_path, 'megadepth_test_1500/Undistorted_SfM', im2)
                img2 = cv2.imread(ori_im2)
                sz2_ori = np.array(img2.shape)[:2][::-1]
                data_megadepth[im2] = [K2[1, 1], K2[0, 2], K2[1, 2], sz2_ori[0], sz2_ori[1]]
                data_megadepth_b.append(data_megadepth[im2])           
    
        data_megadepth = np.asarray([data_megadepth[im] for im in data_megadepth.keys()])
        data_megadepth_a = np.asarray(data_megadepth_a)
        data_megadepth_b = np.asarray(data_megadepth_b)
        
        # scannet
        ppath= os.path.join(bench_path, 'gt_data', 'scannet')
        intrinsic_path = 'intrinsics.npz'
        npz_path = 'test.npz'
    
        data = np.load(os.path.join(ppath, npz_path))
        data_names = data['name']
        intrinsics = dict(np.load(os.path.join(ppath, intrinsic_path)))
        rel_pose = data['rel_pose']
        
        data_scannet = {}
        data_scannet_a = []
        data_scannet_b = []    
        for idx in range(data_names.shape[0]):
            scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_names[idx]
            scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
        
            im1 = os.path.join(scene_name, 'color', f'{stem_name_0}.jpg')
            K1 = intrinsics[scene_name]
            data_scannet[im1] = [K1[1, 1], K1[0, 2], K1[1, 2], 640, 480]
            data_scannet_a.append(data_scannet[im1])
    
            im2 = os.path.join(scene_name, 'color', f'{stem_name_1}.jpg')
            K2 = intrinsics[scene_name]
            data_scannet[im2] = [K2[1, 1], K2[0, 2], K2[1, 2], 640, 480]
            data_scannet_b.append(data_scannet[im2])
            
        data_scannet = np.asarray([data_scannet[im] for im in data_scannet.keys()])
        data_scannet_a = np.asarray(data_scannet_a)
        data_scannet_b = np.asarray(data_scannet_b)
     
        # phototourism
        data_imc = {}
        data_imc_a = []
        data_imc_b = []
        
        out_dir = os.path.join(bench_path, 'imc_phototourism')
        scenes = [scene for scene in os.listdir(os.path.join(out_dir, 'train')) if os.path.isdir(os.path.join(out_dir, 'train', scene))]
        
        sampled_idx = decompress_pickle(os.path.join(bench_path, 'downloads', 'imc_sampled_idx.pbz2'))
        
        for sn in range(len(scenes)):    
            scene = scenes[sn]
            
            work_path = os.path.join(out_dir, 'train', scene)
            pose_file  = os.path.join(work_path, 'calibration.csv')
            covis_file  = os.path.join(work_path, 'pair_covisibility.csv')
    
            if (not os.path.isfile(pose_file)) or (not os.path.isfile(covis_file)):
                continue
            
            im1_ = []
            im2_ = []
            with open(covis_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    pp = row['pair'].split('-')
                    im1_.append(os.path.join(scene, pp[0]))
                    im2_.append(os.path.join(scene, pp[1]))
    
            full_idx = sampled_idx[scene]
                        
            im1_ = [im1_[i] for i in full_idx]
            im2_ = [im2_[i] for i in full_idx]
            
            Kv = {}        
            calib_file = os.path.join(work_path, 'calibration.csv')
            with open(calib_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    cam = os.path.join(scene, row['image_id'])
                    Kv[cam] = np.asarray([float(i) for i in row['camera_intrinsics'].split(' ')]).reshape((3, 3))
    
            for i in range(len(im1_)):
                im1s = os.path.join(bench_path, 'imc_phototourism', 'train', scene, 'images', os.path.split(im1_[i])[1] + '.jpg')
                img1 = cv2.imread(im1s)
                K1 = Kv[im1_[i]]
                sz1_ori = np.array(img1.shape)[:2][::-1]
                data_imc[im1_[i]] = [K1[1, 1], K1[0, 2], K1[1, 2], sz1_ori[0], sz1_ori[1]]
                data_imc_a.append(data_imc[im1_[i]])
    
                im2s = os.path.join(bench_path, 'imc_phototourism', 'train', scene, 'images', os.path.split(im2_[i])[1] + '.jpg')
                img2 = cv2.imread(im2s)
                K2 = Kv[im2_[i]]
                sz2_ori = np.array(img2.shape)[:2][::-1]
                data_imc[im2_[i]] = [K2[1, 1], K2[0, 2], K2[1, 2], sz2_ori[0], sz2_ori[1]]
                data_imc_b.append(data_imc[im2_[i]])
    
        data_imc = np.asarray([data_imc[im] for im in data_imc.keys()])
        data_imc_a = np.asarray(data_imc_a)
        data_imc_b = np.asarray(data_imc_b)
    
        compressed_pickle(os.path.join(bench_path,'intrinsics_stats.pbz2'),
                          (data_megadepth, data_scannet, data_imc,
                           data_megadepth_a, data_scannet_a, data_imc_a,
                           data_megadepth_b, data_scannet_b, data_imc_b)
                          )

    scale_fun = lambda vdata: np.max(vdata[:, -2:], axis=1)
    # scale_fun = lambda vdata: np.sqrt(vdata[:, -2] * vdata[:, -1])

    vdata = [data_megadepth, data_imc, data_scannet]
    for i in range(len(vdata)):
        scale_to = scale_fun(vdata[i])
        vdata[i] = vdata[i][:, 0] / scale_to
    
    vdata_a = [data_megadepth_a, data_imc_a, data_scannet_a]
    for i in range(len(vdata_a)):
        scale_to = scale_fun(vdata_a[i])
        vdata_a[i] = vdata_a[i][:, 0] / scale_to

    vdata_b = [data_megadepth_b, data_imc_b, data_scannet_b]
    for i in range(len(vdata_b)):
        scale_to = scale_fun(vdata_b[i])
        vdata_b[i] = vdata_b[i][:, 0] / scale_to

    v_min = np.min(np.hstack(vdata))
    v_max = np.max(np.hstack(vdata))
    
    nbins = 100    
    h = [np.histogram2d(vdata_a[i], vdata_b[i], bins=nbins, range=[[v_min, v_max], [v_min, v_max]]) for i in range(len(vdata))]
    
    ppath = os.path.join(bench_path, 'res', 'latex')
    os.makedirs(ppath, exist_ok=True)
    
    labels = ['megadepth', 'phototourism', 'scannet']
    
    for i in range(3):
        fig = plt.figure(i)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Times",
            })
        imm = np.rot90((1 - h[i][0] / np.max(h[i][0]))**2)
        plt.imshow(imm, cmap='gray', extent=[v_min, v_max, v_min, v_max])
        plt.xlabel("$f$ / $\max(w, h)$ for the $1^{st}$ image")
        plt.ylabel("$f$ / $\max(w, h)$ for the $2^{nd}$ image")
        plt.xticks(range(9))
        plt.yticks(range(9))
        fig_name = os.path.join(ppath, '2d_distribution_' + labels[i] + '.pdf')
        plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')
        plt.close(fig)
    
    cf = [0.3, 0.3, 3]        
    imm = [np.rot90(((h[i][0] / np.max(h[i][0])))**cf[i]) for i in [0, 1, 2]]

    imm = np.stack(imm, axis=-1)
    # force single point blue for ScanNet in order to improve visualization
    imm[100-7,6,:] = [0, 0, 1]

    fig = plt.figure()
    ax = plt.gca()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": "Times",
        })
    plt.imshow(imm, cmap='gray', extent=[v_min, v_max, v_min, v_max])
    plt.xlabel("$f$ / $\max(w, h)$ in the $1^{st}$ image")
    plt.ylabel("$f$ / $\max(w, h)$ in the $2^{nd}$ image")
    r_patch = mpatches.Patch(color='red', label='MegaDepth')
    g_patch = mpatches.Patch(color='green', label='IMC PhotoTourism')
    b_patch = mpatches.Patch(color='blue', label='ScanNet')
    ax.legend(handles=[r_patch, g_patch, b_patch])
    plt.xticks(range(9))
    plt.yticks(range(9))
    fig_name = os.path.join(ppath, '2d_distribution_as_rgb.pdf')
    plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')
    plt.close(fig)
    
    
    fig, ax = plt.subplots()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": "Times",
        })    
    nnbins = 150
    h = [np.histogram(vdata[i], bins=nnbins, range=[v_min, v_max]) for i in range(len(vdata))]
    for i in range(len(vdata)):    
        ax.stairs(h[i][0] / np.sum(h[i][0]), h[i][1])
    ax.legend(['MegaDepth', 'IMC PhotoTourism', 'ScanNet'])
    ax.set_yscale('log')    
    plt.xlabel("$f$ / $\max(w, h)$")
    plt.ylabel("probability density")
    fig_name = os.path.join(ppath, 'intrinsics_distribution.pdf')
    plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": "Times",
        })    
    nnbins = 150
    h = [np.histogram(vdata[i], bins=nnbins, range=[v_min, v_max]) for i in range(len(vdata))]
    for i in range(2):    
        ax.stairs(h[i][0] / np.sum(h[i][0]), h[i][1])
    ax.legend(['MegaDepth', 'IMC PhotoTourism']) 
    plt.xlabel('$f$ / $\max(w, h)$')
    plt.ylabel("probability density")
    fig_name = os.path.join(ppath, 'intrinsics_distribution_outdoor.pdf')
    plt.savefig(fig_name, dpi = fig_dpi, bbox_inches='tight')
    plt.close(fig)    
    
    l = np.min(vdata[1])
    r = np.max(vdata[1])
    # MegaDepth vdata in the range of ScanNet vdata: 0.0%
    n1 = np.sum((vdata[0]>= l) & (vdata[0] <= r))/vdata[0].shape[0]
    # PhotoTourism vdata in the range of ScanNet vdata: 0.002% (only 10 images) 
    n2 = np.sum((vdata[2]>= l) & (vdata[2] <= r))/vdata[2].shape[0]
