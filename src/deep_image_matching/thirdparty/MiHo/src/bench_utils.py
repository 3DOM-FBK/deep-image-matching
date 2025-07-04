from PIL import Image
import cv2
import numpy as np
import torch
import gdown
import tarfile
import zipfile
import os
import warnings
import _pickle as cPickle
import bz2
import shutil
import src.ncc as ncc
import csv
import time

from matplotlib import colormaps
import matplotlib.pyplot as plt
import src.plot.viz2d as viz
import src.plot.utils as viz_utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe_color = ['red', 'blue', 'lime', 'fuchsia', 'yellow']


from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


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


def progress_bar(text=''):
    return Progress(
        TextColumn(text + " [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def megadepth_1500_list(ppath='bench_data/gt_data/megadepth'):
    npz_list = [i for i in os.listdir(ppath) if (os.path.splitext(i)[1] == '.npz')]

    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    # Sort to avoid os.listdir issues 
    for name in sorted(npz_list):
        scene_info = np.load(os.path.join(ppath, name), allow_pickle=True)
    
        # Sort to avoid pickle issues 
        pidx = sorted([[pair_info[0][0], pair_info[0][1]] for pair_info in scene_info['pair_infos']])
    
        # Collect pairs
        for idx in pidx:
            id1, id2 = idx
            im1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
            im2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')                        
            K1 = scene_info['intrinsics'][id1].astype(np.float32)
            K2 = scene_info['intrinsics'][id2].astype(np.float32)
    
            # Compute relative pose
            T1 = scene_info['poses'][id1]
            T2 = scene_info['poses'][id2]
            T12 = np.matmul(T2, np.linalg.inv(T1))
    
            data['im1'].append(im1)
            data['im2'].append(im2)
            data['K1'].append(K1)
            data['K2'].append(K2)
            data['T'].append(T12[:3, 3])
            data['R'].append(T12[:3, :3])   
    return data


def scannet_1500_list(ppath='bench_data/gt_data/scannet'):
    intrinsic_path = 'intrinsics.npz'
    npz_path = 'test.npz'

    data = np.load(os.path.join(ppath, npz_path))
    data_names = data['name']
    intrinsics = dict(np.load(os.path.join(ppath, intrinsic_path)))
    rel_pose = data['rel_pose']
    
    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    
    for idx in range(data_names.shape[0]):
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_names[idx]
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
    
        # read the grayscale image which will be resized to (1, 480, 640)
        im1 = os.path.join(scene_name, 'color', f'{stem_name_0}.jpg')
        im2 = os.path.join(scene_name, 'color', f'{stem_name_1}.jpg')
        
        # read the intrinsic of depthmap
        K1 = intrinsics[scene_name]
        K2 = intrinsics[scene_name]
    
        # pose    
        T12 = np.concatenate((rel_pose[idx],np.asarray([0, 0, 0, 1.0]))).reshape(4,4)
        
        data['im1'].append(im1)
        data['im2'].append(im2)
        data['K1'].append(K1)
        data['K2'].append(K2)  
        data['T'].append(T12[:3, 3])
        data['R'].append(T12[:3, :3])     
    return data


def megadepth_scannet_bench_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', save_to='megadepth_scannet.pbz2', debug=False, **dummy_args):
    megadepth_data, scannet_data, data_file = bench_init(bench_path=bench_path, bench_gt=bench_gt, save_to=save_to, debug=debug)
    megadepth_data, scannet_data = setup_images(megadepth_data, scannet_data, data_file=data_file, bench_path=bench_path, bench_imgs=bench_imgs)
    
    return megadepth_data, scannet_data, data_file


def megadepth_bench_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', save_to='megadepth_scannet.pbz2', debug=False, **dummy_args):
    megadepth_data, scannet_data, data_file = bench_init(bench_path=bench_path, bench_gt=bench_gt, save_to=save_to, debug=debug)
    megadepth_data, scannet_data = setup_images(megadepth_data, scannet_data, data_file=data_file, bench_path=bench_path, bench_imgs=bench_imgs)
    
    return megadepth_data, data_file


def scannet_bench_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', save_to='megadepth_scannet.pbz2', debug=False, **dummy_args):
    megadepth_data, scannet_data, data_file = bench_init(bench_path=bench_path, bench_gt=bench_gt, save_to=save_to, debug=debug)
    megadepth_data, scannet_data = setup_images(megadepth_data, scannet_data, data_file=data_file, bench_path=bench_path, bench_imgs=bench_imgs)
    
    return scannet_data, data_file


def bench_init(bench_path='bench_data', bench_gt='gt_data', save_to='megadepth_scannet.pbz2', debug=False, debug_pairs=10):
    download_megadepth_scannet_data(bench_path)
        
    data_file = os.path.join(bench_path, save_to)
    if not os.path.isfile(data_file):      
        megadepth_data = megadepth_1500_list(os.path.join(bench_path, bench_gt, 'megadepth'))
        scannet_data = scannet_1500_list(os.path.join(bench_path, bench_gt, 'scannet'))

        # for debugging, use only first debug_pairs image pairs
        if debug:
            for what in megadepth_data.keys():
                megadepth_data[what] = [megadepth_data[what][i] for i in range(debug_pairs)]
            for what in scannet_data.keys():
                scannet_data[what] = [scannet_data[what][i] for i in range(debug_pairs)]

        compressed_pickle(data_file, (megadepth_data, scannet_data))
    else:
        megadepth_data, scannet_data = decompress_pickle(data_file)
    
    return megadepth_data, scannet_data, data_file


def resize_megadepth(im, res_path='imgs/megadepth', bench_path='bench_data', force=False):
    mod_im = os.path.join(bench_path, res_path, os.path.splitext(im)[0] + '.png')
    ori_im= os.path.join(bench_path, 'megadepth_test_1500/Undistorted_SfM', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size) 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1]

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]
    sz_max = float(max(sz_ori))

    if sz_max > 1200:
        cf = 1200 / sz_max                    
        sz_new = np.ceil(sz_ori * cf).astype(int) 
        img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
        sc = sz_ori/sz_new
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return sc
    else:
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return np.array([1., 1.])


def resize_scannet(im, res_path='imgs/scannet', bench_path='bench_data', force=False):
    mod_im = os.path.join(bench_path, res_path, os.path.splitext(im)[0] + '.png')
    ori_im= os.path.join(bench_path, 'scannet_test_1500', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size) 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1]

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]

    sz_new = np.array([640, 480])
    img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
    sc = sz_ori/sz_new
    os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
    cv2.imwrite(mod_im, img)
    return sc


def setup_images(megadepth_data, scannet_data, data_file='bench_data/megadepth_scannet.pbz2', bench_path='bench_data', bench_imgs='imgs'):
    if not ('im_pair_scale' in megadepth_data.keys()):        
        n = len(megadepth_data['im1'])
        im_pair_scale = np.zeros((n, 2, 2))
        res_path = os.path.join(bench_imgs, 'megadepth')
        with progress_bar('MegaDepth - image setup completion') as p:
            for i in p.track(range(n)):
                im_pair_scale[i, 0] = resize_megadepth(megadepth_data['im1'][i], res_path, bench_path)
                im_pair_scale[i, 1] = resize_megadepth(megadepth_data['im2'][i], res_path, bench_path)
        megadepth_data['im_pair_scale'] = im_pair_scale

        n = len(scannet_data['im1'])
        im_pair_scale = np.zeros((n, 2, 2))
        res_path = os.path.join(bench_imgs, 'scannet')
        with progress_bar('ScanNet - image setup completion') as p:
            for i in p.track(range(n)):
                im_pair_scale[i, 0] = resize_scannet(scannet_data['im1'][i], res_path, bench_path)
                im_pair_scale[i, 1] = resize_scannet(scannet_data['im2'][i], res_path, bench_path)
        scannet_data['im_pair_scale'] = im_pair_scale
        
        compressed_pickle(data_file, (megadepth_data, scannet_data))
 
    return megadepth_data, scannet_data


def relative_pose_error_angular(R_gt, t_gt, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    # t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    # R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def relative_pose_error_metric(R_gt, t_gt, R, t, scale_cf=1.0, use_gt_norm=True, t_ambiguity=True):
    t_gt = t_gt * scale_cf
    t = t * scale_cf
    if use_gt_norm: 
        n_gt = np.linalg.norm(t_gt)
        n = np.linalg.norm(t)
        t = t / n * n_gt

    if t_ambiguity:
        t_err = np.minimum(np.linalg.norm(t_gt - t), np.linalg.norm(t_gt + t))
    else:
        t_err = np.linalg.norm(t_gt - t)

    if not isinstance(R, list):
        R = [R]
        
    R_err = []
    for R_ in R:        
        cos = (np.trace(np.dot(R_.T, R_gt)) - 1) / 2
        cos = np.clip(cos, -1., 1.)  # handle numercial errors
        R_err.append(np.rad2deg(np.abs(np.arccos(cos))))
    
    R_err = np.min(R_err)

    return t_err, R_err


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, max_iters=10000):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC, maxIters=max_iters)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def error_auc(errors, thr):
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))

    last_index = np.searchsorted(errors, thr)
    y = recall[:last_index] + [recall[last_index-1]]
    x = errors[:last_index] + [thr]
    return np.trapz(y, x) / thr    


def run_pipe(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', bench_res='res', force=False, ext='.png', running_time=False):

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)        
    with progress_bar(bar_name + ' - pipeline completion') as p:
        for i in p.track(range(n)):
            im1 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im1'][i])[0]) + ext
            im2 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im2'][i])[0]) + ext

            pipe_data = {'im1': im1, 'im2': im2}
            pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
            for pipe_module in pipe:
                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')
            
                if os.path.isfile(pipe_f) and not force:
                    out_data = decompress_pickle(pipe_f)
                else:                    
                    # # start CC crash skip
                    # out_data_cc = out_data
                    # out_data_cc['mask'] = []
                    # if running_time:
                    #    out_data_cc['running_time'] = [np.nan]
                    # os.makedirs(os.path.dirname(pipe_f), exist_ok=True)                 
                    # compressed_pickle(pipe_f, out_data)
                    # # end CC crash skip

                    if running_time: start_time = time.time()

                    out_data = pipe_module.run(**pipe_data)

                    if running_time:
                        stop_time = time.time()
                        out_data['running_time'] = [stop_time - start_time]

                    os.makedirs(os.path.dirname(pipe_f), exist_ok=True)                 
                    compressed_pickle(pipe_f, out_data)
                    
                for k, v in out_data.items(): pipe_data[k] = v

                                                
# original benchmark metric
def eval_pipe_essential(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data', bench_res='res', essential_th_list=[0.5, 1, 1.5], save_to='res_essential.pbz2', force=False, use_scale=False, also_metric=False):
    warnings.filterwarnings("ignore", category=UserWarning)


    angular_thresholds = [5, 10, 20]
    metric_thresholds = [0.5, 1, 2]
    # warning: current metric error requires that angular_thresholds[i] / metric_thresholds[i] = am_scaling
    am_scaling = 10
 
    K1 = dataset_data['K1']
    K2 = dataset_data['K2']
    R_gt = dataset_data['R']
    t_gt = dataset_data['T']

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    for essential_th in essential_th_list:            
        n = len(dataset_data['im1'])
        
        pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
        pipe_name_base_small = ''
        for pipe_module in pipe:
            pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
            pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())

            print(bar_name + ' evaluation with RANSAC essential matrix threshold ' + str(essential_th) + ' px')
            print('Pipeline: ' + pipe_name_base_small)

            if ((pipe_name_base + '_essential_th_list_' + str(essential_th)) in eval_data.keys()) and not force:
                eval_data_ = eval_data[pipe_name_base + '_essential_th_list_' + str(essential_th)]                
                for a in angular_thresholds:
                    print(f"AUC@{str(a).ljust(2,' ')} (E) : {eval_data_['pose_error_e_auc_' + str(a)]}")   
                
                if also_metric:
                    for a, m in zip(angular_thresholds, metric_thresholds):    
                        print(f"AUC@{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} (E) : {eval_data_['pose_error_em_auc_' + str(a) + '_' + str(m)]}")

                continue
                    
            eval_data_ = {}
            eval_data_['R_errs_e'] = []
            eval_data_['t_errs_e'] = []
            eval_data_['inliers_e'] = []
            
            if also_metric:
                eval_data_['R_errm_e'] = []
                eval_data_['t_errm_e'] = []
                                
            with progress_bar('Completion') as p:
                for i in p.track(range(n)):            
                    pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')
                                        
                    if os.path.isfile(pipe_f):
                        out_data = decompress_pickle(pipe_f)
    
                        pts1 = out_data['pt1']
                        pts2 = out_data['pt2']
                                                
                        if torch.is_tensor(pts1):
                            pts1 = pts1.detach().cpu().numpy()
                            pts2 = pts2.detach().cpu().numpy()
                        
                        if use_scale:
                            scales = dataset_data['im_pair_scale'][i]
                        
                            pts1 = pts1 * scales[0]
                            pts2 = pts2 * scales[1]
                            
                        nn = pts1.shape[0]
                                                                            
                        if nn < 5:
                            Rt = None
                        else:                            
                            Rt = estimate_pose(pts1, pts2, K1[i], K2[i], essential_th)                                                        
                    else:
                        Rt = None
        
                    if Rt is None:
                        eval_data_['R_errs_e'].append(np.inf)
                        eval_data_['t_errs_e'].append(np.inf)
                        eval_data_['inliers_e'].append(np.array([]).astype('bool'))
                        
                        if also_metric:
                            eval_data_['R_errm_e'].append(np.inf)
                            eval_data_['t_errm_e'].append(np.inf)                            
                    else:
                        R, t, inliers = Rt
                        t_err, R_err = relative_pose_error_angular(R_gt[i], t_gt[i], R, t)
                        eval_data_['R_errs_e'].append(R_err)
                        eval_data_['t_errs_e'].append(t_err)
                        eval_data_['inliers_e'].append(inliers)

                        if also_metric:
                            t_err, R_err = relative_pose_error_metric(R_gt[i], t_gt[i], R, t, scale_cf=dataset_data['scene_scales'][i])
                            eval_data_['R_errm_e'].append(R_err)
                            eval_data_['t_errm_e'].append(t_err)
        
                aux = np.asarray([eval_data_['R_errs_e'], eval_data_['t_errs_e']]).T
                max_Rt_err = np.max(aux, axis=1)        
                tmp = np.concatenate((aux, np.expand_dims(max_Rt_err, axis=1)), axis=1)
        
                for a in angular_thresholds:       
                    auc_R = error_auc(np.squeeze(eval_data_['R_errs_e']), a)
                    auc_t = error_auc(np.squeeze(eval_data_['t_errs_e']), a)
                    auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
                    eval_data_['pose_error_e_auc_' + str(a)] = np.asarray([auc_R, auc_t, auc_max_Rt])
                    eval_data_['pose_error_e_acc_' + str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

                    print(f"AUC@{str(a).ljust(2,' ')} (E) : {eval_data_['pose_error_e_auc_' + str(a)]}")

                if also_metric:
                    aux = np.asarray([eval_data_['R_errm_e'], eval_data_['t_errm_e']]).T
                    aux[:, 1] = aux[:, 1] * am_scaling
                    max_Rt_err = np.max(aux, axis=1)
        
                    for a, m in zip(angular_thresholds, metric_thresholds):       
                        auc_R = error_auc(np.squeeze(eval_data_['R_errm_e']), a)
                        auc_t = error_auc(np.squeeze(eval_data_['t_errm_e']), m)
                        auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
                        eval_data_['pose_error_em_auc_' + str(a) + '_' + str(m)] = np.asarray([auc_R, auc_t, auc_max_Rt])

                        aa = (aux[:, 0] < a)[:, np.newaxis]
                        mm = (aux[:, 1] < m)[:, np.newaxis]
                        tmp = np.concatenate((aa, mm, aa & mm), axis=1)
                        eval_data_['pose_error_em_acc_' + str(a) + '_' + str(m)] = np.sum(tmp, axis=0)/np.shape(tmp)[0]
    
                        print(f"AUC@{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} (E) : {eval_data_['pose_error_em_auc_' + str(a) + '_' + str(m)]}")

                eval_data[pipe_name_base + '_essential_th_list_' + str(essential_th)] = eval_data_
                compressed_pickle(save_to, eval_data)


def eval_pipe_fundamental(pipe, dataset_data,  dataset_name, bar_name, bench_path='bench_data', bench_res='res', save_to='res_fundamental.pbz2', force=False, use_scale=False, err_th_list=list(range(1,16)), also_metric=False):
    warnings.filterwarnings("ignore", category=UserWarning)

    angular_thresholds = [5, 10, 20]
    metric_thresholds = [0.5, 1, 2]
    # warning: current metric error requires that angular_thresholds[i] / metric_thresholds[i] = am_scaling
    am_scaling = 10

    K1 = dataset_data['K1']
    K2 = dataset_data['K2']
    R_gt = dataset_data['R']
    t_gt = dataset_data['T']

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    n = len(dataset_data['im1'])
    
    pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
    pipe_name_base_small = ''
    pipe_name_root = os.path.join(pipe_name_base, pipe[0].get_id())

    for pipe_module in pipe:
        pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
        pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())

        print('Pipeline: ' + pipe_name_base_small)

        if (pipe_name_base in eval_data.keys()) and not force:
            eval_data_ = eval_data[pipe_name_base]                
            for a in angular_thresholds:
                print(f"AUC@{str(a).ljust(2,' ')} (F) : {eval_data_['pose_error_f_auc_' + str(a)]}")

            if also_metric:
                for a, m in zip(angular_thresholds, metric_thresholds):    
                    print(f"AUC@{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} (F) : {eval_data_['pose_error_fm_auc_' + str(a) + '_' + str(m)]}")

            print(f"precision (F) : {eval_data_['epi_global_prec_f']}")
            print(f"   recall (F) : {eval_data_['epi_global_recall_f']}")
                                                
            continue
                
        eval_data_ = {}
        eval_data_['R_errs_f'] = []
        eval_data_['t_errs_f'] = []
        eval_data_['epi_max_error_f'] = []
        eval_data_['epi_inliers_f'] = []
        eval_data_['epi_prec_f'] = []
        eval_data_['epi_recall_f'] = []
        
        if also_metric:
            eval_data_['R_errm_f'] = []
            eval_data_['t_errm_f'] = []        
            
        with progress_bar('Completion') as p:
            for i in p.track(range(n)):            
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')

                epi_max_err =[]
                inl_sum = []
                avg_prec = 0
                avg_recall = 0

                if os.path.isfile(pipe_f):
                    out_data = decompress_pickle(pipe_f)

                    pts1 = out_data['pt1']
                    pts2 = out_data['pt2']
                                            
                    if torch.is_tensor(pts1):
                        pts1 = pts1.detach().cpu().numpy()
                        pts2 = pts2.detach().cpu().numpy()

                    if use_scale:
                        scales = dataset_data['im_pair_scale'][i]
                    else:
                        scales = np.asarray([[1.0, 1.0], [1.0, 1.0]])
                    
                    pts1 = pts1 * scales[0]
                    pts2 = pts2 * scales[1]
                        
                    nn = pts1.shape[0]
                                                
                    if nn < 8:
                        Rt_ = None
                    else:
                        F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
                        if F is None:
                            Rt_ = None
                        else:
                            E = K2[i].T @ F @ K1[i]
                            Rt_ = cv2.decomposeEssentialMat(E)

                    if nn > 0:
                        F_gt = torch.tensor(K2[i].T, device=device, dtype=torch.float64).inverse() @ \
                               torch.tensor([[0, -t_gt[i][2], t_gt[i][1]],
                                            [t_gt[i][2], 0, -t_gt[i][0]],
                                            [-t_gt[i][1], t_gt[i][0], 0]], device=device) @ \
                               torch.tensor(R_gt[i], device=device) @ \
                               torch.tensor(K1[i], device=device, dtype=torch.float64).inverse()
                        F_gt = F_gt / F_gt.sum()

                        pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
                        pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))
                        
                        l1_ = F_gt @ pt1_
                        d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()
                        
                        l2_ = F_gt.T @ pt2_
                        d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()
                        
                        epi_max_err = torch.maximum(d1, d2)                               
                        inl_sum = (epi_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)
                        avg_prec = inl_sum.type(torch.double).mean()/nn
                                                
                        if pipe_name_base==pipe_name_root:
                            recall_normalizer = torch.tensor(inl_sum, device=device)
                        else:
                            recall_normalizer = torch.tensor(eval_data[pipe_name_root]['epi_inliers_f'][i], device=device)
                        avg_recall = inl_sum.type(torch.double) / recall_normalizer
                        avg_recall[~avg_recall.isfinite()] = 0
                        avg_recall = avg_recall.mean()
                        
                        epi_max_err = epi_max_err.detach().cpu().numpy()
                        inl_sum = inl_sum.detach().cpu().numpy()
                        avg_prec = avg_prec.item()
                        avg_recall = avg_recall.item()
                else:
                    Rt_ = None
                    
                    
                if Rt_ is None:
                    eval_data_['R_errs_f'].append(np.inf)
                    eval_data_['t_errs_f'].append(np.inf)
                    
                    if also_metric:
                        eval_data_['R_errm_f'].append(np.inf)
                        eval_data_['t_errm_f'].append(np.inf)                        
                    
                else:
                    R_a, t_a, = Rt_[0], Rt_[2].squeeze()
                    t_err_a, R_err_a = relative_pose_error_angular(R_gt[i], t_gt[i], R_a, t_a)

                    R_b, t_b, = Rt_[1], Rt_[2].squeeze()
                    t_err_b, R_err_b = relative_pose_error_angular(R_gt[i], t_gt[i], R_b, t_b)

                    if max(R_err_a, t_err_a) < max(R_err_b, t_err_b):
                        R_err, t_err = R_err_a, t_err_b
                    else:
                        R_err, t_err = R_err_b, t_err_b

                    eval_data_['R_errs_f'].append(R_err)
                    eval_data_['t_errs_f'].append(t_err)
                                        
                    if also_metric:                           
                        t_err, R_err = relative_pose_error_metric(R_gt[i], t_gt[i], [Rt_[0], Rt_[1]], Rt_[2].squeeze(), scale_cf=dataset_data['scene_scales'][i])
                        eval_data_['R_errm_f'].append(R_err)
                        eval_data_['t_errm_f'].append(t_err)                    
                    
                eval_data_['epi_max_error_f'].append(epi_max_err)  
                eval_data_['epi_inliers_f'].append(inl_sum)
                eval_data_['epi_prec_f'].append(avg_prec)                           
                eval_data_['epi_recall_f'].append(avg_recall)
                    
            aux = np.asarray([eval_data_['R_errs_f'], eval_data_['t_errs_f']]).T
            max_Rt_err = np.max(aux, axis=1)        
            tmp = np.concatenate((aux, np.expand_dims(max_Rt_err, axis=1)), axis=1)
    
            for a in angular_thresholds:       
                auc_R = error_auc(np.squeeze(eval_data_['R_errs_f']), a)
                auc_t = error_auc(np.squeeze(eval_data_['t_errs_f']), a)
                auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
                eval_data_['pose_error_f_auc_' + str(a)] = np.asarray([auc_R, auc_t, auc_max_Rt])
                eval_data_['pose_error_f_acc_' + str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

                print(f"AUC@{str(a).ljust(2,' ')} (F) : {eval_data_['pose_error_f_auc_' + str(a)]}")
            
            if also_metric:
                aux = np.asarray([eval_data_['R_errm_f'], eval_data_['t_errm_f']]).T
                aux[:, 1] = aux[:, 1] * am_scaling
                max_Rt_err = np.max(aux, axis=1)
    
                for a, m in zip(angular_thresholds, metric_thresholds):       
                    auc_R = error_auc(np.squeeze(eval_data_['R_errm_f']), a)
                    auc_t = error_auc(np.squeeze(eval_data_['t_errm_f']), m)
                    auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
                    eval_data_['pose_error_fm_auc_' + str(a) + '_' + str(m)] = np.asarray([auc_R, auc_t, auc_max_Rt])

                    aa = (aux[:, 0] < a)[:, np.newaxis]
                    mm = (aux[:, 1] < m)[:, np.newaxis]
                    tmp = np.concatenate((aa, mm, aa & mm), axis=1)
                    eval_data_['pose_error_fm_acc_' + str(a) + '_' + str(m)] = np.sum(tmp, axis=0)/np.shape(tmp)[0]

                    print(f"@AUC{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} (F) : {eval_data_['pose_error_fm_auc_' + str(a) + '_' + str(m)]}")
            
            eval_data_['epi_global_prec_f'] = torch.tensor(eval_data_['epi_prec_f'], device=device).mean().item()
            eval_data_['epi_global_recall_f'] = torch.tensor(eval_data_['epi_recall_f'], device=device).mean().item()
        
            print(f"precision (F) : {eval_data_['epi_global_prec_f']}")
            print(f"   recall (F) : {eval_data_['epi_global_recall_f']}")

            eval_data[pipe_name_base] = eval_data_
            compressed_pickle(save_to, eval_data)


def download_megadepth_scannet_data(bench_path ='bench_data'):   
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)   

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_scannet_gt_data.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1GtpHBN6RLcgSW5RPPyqYLyfbn7ex360G/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'gt_data')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(bench_path)    

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1Vwk_htrvWmw5AtJRmHw10ldK57ckgZ3r/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)
    
    out_dir = os.path.join(bench_path, 'megadepth_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    file_to_download = os.path.join(bench_path, 'downloads', 'scannet_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/13KCCdC1k3IIZ4I3e4xJoVMvDA84Wo-AG/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'scannet_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    return


def show_pipe(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', bench_res='res', bench_plot='plot', force=False, ext='.png', save_ext='.jpg', fig_min_size=960, fig_max_size=1280):

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)    
    fig = plt.figure()    
    
    with progress_bar(bar_name + ' - pipeline completion') as p:
        for i in p.track(range(n)):
            im1 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im1'][i])[0]) + ext
            im2 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im2'][i])[0]) + ext
                        
            pair_data = []            
            pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
            pipe_img_save = os.path.join(bench_path, bench_plot, dataset_name)
            for pipe_module in pipe:
                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
                pipe_img_save = os.path.join(pipe_img_save, pipe_module.get_id())

                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')            
                pair_data.append(decompress_pickle(pipe_f))
            
            os.makedirs(pipe_img_save, exist_ok=True)
            pipe_img_save = os.path.join(pipe_img_save, str(i) + save_ext)
            if os.path.isfile(pipe_img_save) and not force:
                continue
            
            img1 = viz_utils.load_image(im1)
            img2 = viz_utils.load_image(im2)
            fig, axes = viz.plot_images([img1, img2], fig_num=fig.number)              
            
            pt1 = pair_data[0]['pt1']
            pt2 = pair_data[0]['pt2']
            l = pt1.shape[0]
            
            idx = torch.arange(l, device=device)                            
            clr = 0
            for j in range(1, len(pair_data)):
                if 'mask' in pair_data[j].keys():
                    mask = pair_data[j]['mask']
                    if isinstance(mask, list): mask = np.asarray(mask, dtype=bool)
                    if mask.shape[0] > 0:
                        mpt1 = pt1[idx[~mask]]
                        mpt2 = pt2[idx[~mask]]
                        viz.plot_matches(mpt1, mpt2, color=pipe_color[clr], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
                        idx = idx[mask]
                    clr = clr + 1
            mpt1 = pt1[idx]
            mpt2 = pt2[idx]
            viz.plot_matches(mpt1, mpt2, color=pipe_color[clr], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)

            fig_dpi = fig.get_dpi()
            fig_sz = [fig.get_figwidth() * fig_dpi, fig.get_figheight() * fig_dpi]

            fig_cz = min(fig_sz)
            if fig_cz < fig_min_size:
                fig_sz[0] = fig_sz[0] / fig_cz * fig_min_size
                fig_sz[1] = fig_sz[1] / fig_cz * fig_min_size

            fig_cz = max(fig_sz)
            if fig_cz > fig_min_size:
                fig_sz[0] = fig_sz[0] / fig_cz * fig_max_size
                fig_sz[1] = fig_sz[1] / fig_cz * fig_max_size
                
            fig.set_size_inches(fig_sz[0] / fig_dpi, fig_sz[1]  / fig_dpi)

            viz.save_plot(pipe_img_save, fig)
            viz.clear_plot(fig)
    plt.close(fig)


def planar_bench_setup(to_exclude =['graf'], max_imgs=6, bench_path='bench_data', bench_imgs='imgs', bench_plot='plot', save_to='planar.pbz2', upright=True, force=False, save_ext='.jpg', **dummy_args):        

    save_to_full = os.path.join(bench_path, save_to)
    if os.path.isfile(save_to_full) and (not force):
        return decompress_pickle(save_to_full), save_to_full  

    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)

    file_to_download = os.path.join(bench_path, 'downloads', 'planar_data.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1XkP4RR9KKbCV94heI5JWlue2l32H0TNs/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'planar')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(out_dir)        

    planar_scenes = sorted([scene[:-5] for scene in os.listdir(out_dir) if scene[-5:]=='1.png' and scene[:5]!='mask_'])
    for i in to_exclude: planar_scenes.remove(i)

    in_path = out_dir
    out_path = os.path.join(bench_path, bench_imgs, 'planar')
    check_path = os.path.join(bench_path, bench_plot, 'planar_check')

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(check_path, exist_ok=True)

    im1 = []
    im2 = []
    sz1 = []
    sz2 = []
    H = []
    H_inv = []
    im1_mask = []
    im2_mask = []
    im1_full_mask = []
    im2_full_mask = []

    im1_use_mask = []
    im2_use_mask = []
    im_pair_scale = []
    
    with progress_bar('Completion') as p:
        for sn in p.track(range(len(planar_scenes))):     
            scene = planar_scenes[sn]
    
            img1 = scene + '1.png'
            img1_mask = 'mask_' + scene + '1.png'
            img1_mask_bad = 'mask_bad_' + scene + '1.png'
    
            for i in range(2, max_imgs+1):
                img2 = scene + str(i) + '.png'
                img2_mask = 'mask_' + scene  + str(i) + '.png'
                img2_mask_bad = 'mask_bad_' + scene + str(i) + '.png'
    
                H12 = scene + '_H1' + str(i) + '.txt'
                            
                im1s = os.path.join(in_path, img1)
                im1s_mask = os.path.join(in_path, img1_mask)
                im1s_mask_bad = os.path.join(in_path, img1_mask_bad)
                im2s = os.path.join(in_path, img2)
                im2s_mask = os.path.join(in_path, img2_mask)
                im2s_mask_bad = os.path.join(in_path, img2_mask_bad)
                H12s = os.path.join(in_path, H12)
                
                if not os.path.isfile(H12s):
                    continue
                
                im1d = os.path.join(out_path, img1)
                im2d = os.path.join(out_path, img2)
     
                shutil.copyfile(im1s, im1d)
                shutil.copyfile(im2s, im2d)
                
                H_ = np.loadtxt(H12s)
                H_inv_ = np.linalg.inv(H_)
                
                im1.append(img1)
                im2.append(img2)
                H.append(H_)
                H_inv.append(H_inv_)
                
                im1i=cv2.imread(im1s)
                im2i=cv2.imread(im2s)
                
                sz1.append(np.array(im1i.shape)[:2][::-1])
                sz2.append(np.array(im2i.shape)[:2][::-1])
                            
                im2i_ = cv2.warpPerspective(im1i,H_,(im2i.shape[1],im2i.shape[0]), flags=cv2.INTER_LANCZOS4)
                im1i_ = cv2.warpPerspective(im2i,H_inv_,(im1i.shape[1],im1i.shape[0]), flags=cv2.INTER_LANCZOS4)
                
                im_pair_scale.append(np.ones((2, 2)))
                
                im1_mask_ = torch.ones((sz1[-1][1],sz1[-1][0]), device=device, dtype=torch.bool)
                im2_mask_ = torch.ones((sz2[-1][1],sz2[-1][0]), device=device, dtype=torch.bool)
                im1_use_mask_ = False
                im2_use_mask_ = False
                
                if os.path.isfile(im1s_mask):
                    im1_mask_ = torch.tensor((cv2.imread(im1s_mask, cv2.IMREAD_GRAYSCALE)==0), device=device)
                    im1_use_mask_ = True
     
                if os.path.isfile(im2s_mask):
                    im2_mask_ = torch.tensor((cv2.imread(im2s_mask, cv2.IMREAD_GRAYSCALE)==0), device=device)
                    im2_use_mask_ = True
    
                if os.path.isfile(im1s_mask_bad):
                    im1_mask_ = torch.tensor((cv2.imread(im1s_mask_bad, cv2.IMREAD_GRAYSCALE)==0), device=device)
    
                if os.path.isfile(im2s_mask_bad):
                    im2_mask_ = torch.tensor((cv2.imread(im1s_mask_bad, cv2.IMREAD_GRAYSCALE)==0), device=device)
    
                im1_mask.append(im1_mask_.detach().cpu().numpy())
                im2_mask.append(im2_mask_.detach().cpu().numpy())
    
                im1_use_mask.append(im1_use_mask_)
                im2_use_mask.append(im2_use_mask_)
    
                im1_full_mask_ = refine_mask(im1_mask_, im2_mask_, sz1[-1], sz2[-1], H_)
                im2_full_mask_ = refine_mask(im2_mask_, im1_full_mask_, sz2[-1], sz1[-1], H_inv_)
                
                im1_full_mask.append(im1_full_mask_.detach().cpu().numpy())
                im2_full_mask.append(im2_full_mask_.detach().cpu().numpy())
                
                iname = os.path.splitext(img1)[0] + '_' + os.path.splitext(img2)[0]
                            
                cv2.imwrite(os.path.join(check_path, iname + '_1a' + save_ext), im1i)
                cv2.imwrite(os.path.join(check_path, iname + '_1b' + save_ext), im1i_)
                cv2.imwrite(os.path.join(check_path, iname + '_2a' + save_ext), im2i)
                cv2.imwrite(os.path.join(check_path, iname + '_2b' + save_ext), im2i_)
        
        if upright:
            for scene in planar_scenes:
                is_upright = scene[-3:] == 'rot'
                if is_upright:
                    img1_unrot = scene[:-3] + '1.png'
                    img1_rot = scene + '1.png'
    
                    for i in range(2, max_imgs+1):
                        img2_unrot = scene[:-3] + str(i) + '.png'
                        img2_rot = scene + str(i) + '.png'
    
                        rot_idx = [ii for ii, (im1i, im2i) in enumerate(zip(im1, im2)) if (im1i==img1_rot) and (im2i==img2_rot)]
    
                        if len(rot_idx)>0:
                            unrot_idx = [ii for ii, (im1i, im2i) in enumerate(zip(im1, im2)) if (im1i==img1_unrot) and (im2i==img2_unrot)][0]
                            
                            im2d = os.path.join(out_path, img2_unrot)    
                            os.remove(im2d)
                            
                            iname = os.path.splitext(img1_unrot)[0] + '_' + os.path.splitext(img2_unrot)[0]
                                        
                            os.remove(os.path.join(check_path, iname + '_1a'  + save_ext))
                            os.remove(os.path.join(check_path, iname + '_1b'  + save_ext))
                            os.remove(os.path.join(check_path, iname + '_2a'  + save_ext))
                            os.remove(os.path.join(check_path, iname + '_2b'  + save_ext))
                                                    
                            del im1[unrot_idx]
                            del im2[unrot_idx]
                            del H[unrot_idx]
                            del H_inv[unrot_idx]
                            del im1_mask[unrot_idx]
                            del im2_mask[unrot_idx]
                            del sz1[unrot_idx]
                            del sz2[unrot_idx]
                            del im1_use_mask[unrot_idx]
                            del im2_use_mask[unrot_idx]
                            del im1_full_mask[unrot_idx]
                            del im2_full_mask[unrot_idx]

    H = np.asarray(H)
    H_inv = np.asarray(H_inv)

    sz1 = np.asarray(sz1)
    sz2 = np.asarray(sz2)

    im1_use_mask = np.asarray(im1_use_mask)
    im2_use_mask = np.asarray(im2_use_mask)

    im_pair_scale = np.asarray(im_pair_scale)
    
    data = {'im1': im1, 'im2': im2, 'H': H, 'H_inv': H_inv,
            'im1_mask': im1_mask, 'im2_mask': im2_mask, 'sz1': sz1, 'sz2': sz2,
            'im1_use_mask': im1_use_mask, 'im2_use_mask': im2_use_mask,
            'im1_full_mask': im1_full_mask, 'im2_full_mask': im2_full_mask}

    compressed_pickle(save_to_full, data)
    return data, save_to_full


def  refine_mask(im1_mask, im2_mask, sz1, sz2, H):
                
    x = torch.arange(sz1[0], device=device).unsqueeze(0).repeat(sz1[1],1).unsqueeze(-1)
    y = torch.arange(sz1[1], device=device).unsqueeze(1).repeat(1,sz1[0]).unsqueeze(-1)
    z = torch.ones((sz1[1],sz1[0]), device=device).unsqueeze(-1)
    pt1 = torch.cat((x, y, z), dim=-1).reshape((-1, 3))
    pt2_ = torch.tensor(H, device=device, dtype=torch.float) @ pt1.permute(1,0)
    pt2_ = (pt2_[:2] / pt2_[-1].unsqueeze(0)).reshape(2, sz1[1], -1).round()
    mask1_reproj = torch.isfinite(pt2_).all(dim=0) & (pt2_ >= 0).all(dim=0) & (pt2_[0] < sz2[0]) & (pt2_[1] < sz2[1])
    mask1_reproj = mask1_reproj & im1_mask
    masked_pt2 = pt2_[:, mask1_reproj]
    idx = masked_pt2[1] * sz2[0] + masked_pt2[0]
    mask1_reproj[mask1_reproj.clone()] = im2_mask.flatten()[idx.type(torch.long)]
    
    return mask1_reproj


def csv_summary_non_planar(essential_th_list=[0.5, 1, 1.5], essential_load_from='res_essential.pbz2', fundamental_load_from='res_fundamental.pbz2', match_count_load_from=None, runtime_load_from=None, save_to='res_non_planar.csv', also_metric=False, to_remove_prefix=''):
    warnings.filterwarnings("ignore", category=UserWarning)
    lines = []

    angular_thresholds = [5, 10, 20]
    metric_thresholds = [0.5, 1, 2]
    
    # warning: current metric error requires that angular_thresholds[i] / metric_thresholds[i] = am_scaling
 
    e_eval_data = decompress_pickle(essential_load_from)
    f_eval_data = decompress_pickle(fundamental_load_from)

    match_count_header = ''   
    if not (match_count_load_from is None):
        c_eval_data = decompress_pickle(match_count_load_from)
        
        tot_match = None
        for pname in c_eval_data.keys():
            if pname == pname[:pname.rfind(to_remove_prefix)] + to_remove_prefix:
                tot_match = c_eval_data[pname]['matches_avg']
                break
                
        if not (tot_match is None):
            match_count_header = ';filtered_of_' + str(tot_match)

    runtime_header = ''   
    if not (runtime_load_from is None):
        r_eval_data = decompress_pickle(runtime_load_from)
        
        base_runtime = None
        for pname in r_eval_data.keys():
            if pname == pname[:pname.rfind(to_remove_prefix)] + to_remove_prefix:
                base_runtime = r_eval_data[pname]['running_time_avg'][-1]
                break
                
        if not (tot_match is None):
            runtime_header = ';runtime_increment_from_' + str(round(base_runtime*1000)/1000) + '_s'

    l = 0
    for pname in f_eval_data.keys(): l = max(l, len(pname[pname.rfind(to_remove_prefix):].split(os.path.sep)))

    header = ';'.join(['pipe_module_' + str(li) for li in range(l)]) + match_count_header + runtime_header + ';F_precision;F_recall'
    if len(angular_thresholds) > 0:
        header = header + ';' + ';'.join(['F_AUC@' + str(a) for a in angular_thresholds])

        if also_metric:
            header = header + ';' + ';'.join(['F_AUC@(' + str(a) + ',' + str(m) + ')' for a, m in zip(angular_thresholds, metric_thresholds)])
    
    for essential_th in essential_th_list: 
        if len(essential_th_list) == 1:
            lname = ''
        else:
            lname = '(' + str(essential_th) + ')'
        
        if len(angular_thresholds) > 0:        
            header = header + ';' + ';'.join(['E' + lname + '_AUC@' + str(a) for a in angular_thresholds])
        
            if also_metric:
                header = header + ';' + ';'.join(['E' + lname + '_AUC@(' + str(a) + ',' + str(m) + ')' for a, m in zip(angular_thresholds, metric_thresholds)])

    lines.append(header + '\n')
        
    for pname in f_eval_data.keys():    
        lp = len(pname[pname.rfind(to_remove_prefix):].split(os.path.sep))
        row = pname[pname.rfind(to_remove_prefix):].replace(os.path.sep, ';') + (';' * (l - lp))

        match_count_row = ''   
        if not (match_count_load_from is None):
            match_count_row = ';' + str(c_eval_data[pname]['filtered_avg'])

        runtime_row = ''   
        if not (runtime_load_from is None):
            runtime_row = ';' + str(r_eval_data[pname]['running_time_pct_avg'][-1]-1)

        row = row + match_count_row + runtime_row + ';' + str(f_eval_data[pname]['epi_global_prec_f']) + ';' + str(f_eval_data[pname]['epi_global_recall_f']) 
        if len(angular_thresholds) > 0:
            row = row + ';' + ';'.join([str(f_eval_data[pname]['pose_error_f_auc_' + str(a)][-1]) for a in angular_thresholds])        

            if also_metric:
                row = row + ';' + ';'.join([str(f_eval_data[pname]['pose_error_fm_auc_' + str(a) + '_' + str(m)][-1]) for a, m in zip(angular_thresholds, metric_thresholds)])

        for essential_th in essential_th_list:  
            ppname = e_eval_data[pname + '_essential_th_list_' + str(essential_th)]

            if len(angular_thresholds) > 0:
                row = row + ';' + ';'.join([str(ppname['pose_error_e_auc_' + str(a)][-1]) for a in angular_thresholds])         
                
                if also_metric:                    
                    row = row + ';' + ';'.join([str(ppname['pose_error_em_auc_' + str(a) + '_' + str(m)][-1]) for a, m in zip(angular_thresholds, metric_thresholds)] )

        lines.append(row + '\n')
        
    with open(save_to, 'w') as f:
        for l in lines:
            f.write(l)    


def csv_summary_planar(load_from='res_homography.pbz2', save_to='res_planar.csv', match_count_load_from=None, runtime_load_from=None, to_remove_prefix=''):
    warnings.filterwarnings("ignore", category=UserWarning)
    lines = []

    angular_thresholds = [5, 10, 15]
 
    eval_data = decompress_pickle(load_from)
        
    match_count_header = ''   
    if not (match_count_load_from is None):
        c_eval_data = decompress_pickle(match_count_load_from)
        
        tot_match = None
        for pname in c_eval_data.keys():
            if pname == pname[:pname.rfind(to_remove_prefix)] + to_remove_prefix:
                tot_match = c_eval_data[pname]['matches_avg']
                break
                
        if not (tot_match is None):
            match_count_header = ';filtered_of_' + str(tot_match)
    
    runtime_header = ''   
    if not (runtime_load_from is None):
        r_eval_data = decompress_pickle(runtime_load_from)
        
        base_runtime = None
        for pname in r_eval_data.keys():
            if pname == pname[:pname.rfind(to_remove_prefix)] + to_remove_prefix:
                base_runtime = r_eval_data[pname]['running_time_avg'][-1]
                break
                
        if not (tot_match is None):
            runtime_header = ';runtime_increment_from_' + str(round(base_runtime*1000)/1000) + '_s'
        
    l = 0
    for pname in eval_data.keys(): l = max(l, len(pname[pname.rfind(to_remove_prefix):].split(os.path.sep)))
        
    header = ';'.join(['pipe_module_' + str(li) for li in range(l)]) + match_count_header + runtime_header + ';H_precision;H_recall'
    if len(angular_thresholds) > 0: header = header + ';' + ';'.join(['H_AUC@' + str(a) for a in angular_thresholds])
    lines.append(header + '\n')
        
    for pname in eval_data.keys():    
        lp = len(pname[pname.rfind(to_remove_prefix):].split(os.path.sep))
        row = pname[pname.rfind(to_remove_prefix):].replace(os.path.sep, ';') + (';' * (l - lp))
        
        match_count_row = ''   
        if not (match_count_load_from is None):
            match_count_row = ';' + str(c_eval_data[pname]['filtered_avg'])
            
        runtime_row = ''   
        if not (runtime_load_from is None):
            runtime_row = ';' + str(r_eval_data[pname]['running_time_pct_avg'][-1]-1)
                    
        row = row + match_count_row + runtime_row + ';' + str(eval_data[pname]['reproj_global_prec_h']) + ';' + str(eval_data[pname]['reproj_global_recall_h']) 
        
        if len(angular_thresholds) > 0: row = row + ';' + ';'.join([str(eval_data[pname]['pose_error_h_auc_' + str(a)][-1]) for a in angular_thresholds])    
        lines.append(row + '\n')
        
    with open(save_to, 'w') as f:
        for l in lines:
            f.write(l)    


def eval_pipe_homography(pipe, dataset_data,  dataset_name, bar_name, bench_path='bench_data', bench_res='res', save_to='res_homography.pbz2', force=False, use_scale=False, rad=15, err_th_list=list(range(1,16)), bench_plot='plot', save_acc_images=True, save_ext='.jpg', **dummy_args):
    warnings.filterwarnings("ignore", category=UserWarning)

    # these are actually pixel errors
    angular_thresholds = [5, 10, 15]

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    n = len(dataset_data['im1'])
    
    pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
    pipe_name_base_small = ''
    pipe_name_root = os.path.join(pipe_name_base, pipe[0].get_id())
    pipe_img_save = os.path.join(bench_path, bench_plot, 'planar_accuracy')

    for pipe_module in pipe:
        pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
        pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())
        pipe_img_save = os.path.join(pipe_img_save, pipe_module.get_id())

        print('Pipeline: ' + pipe_name_base_small)

        if (pipe_name_base in eval_data.keys()) and not force:
            eval_data_ = eval_data[pipe_name_base]                
            for a in angular_thresholds:
                print(f"AUC@{str(a).ljust(2,' ')} (H) : {eval_data_['pose_error_h_auc_' + str(a)]}")

            print(f"precision(H) : {eval_data_['reproj_global_prec_h']}")
            print(f"recall (H) : {eval_data_['reproj_global_recall_h']}")
                                                
            continue
                
        eval_data_ = {}

        # eval_data_['err_plane_1_h'] = []
        # eval_data_['err_plane_2_h'] = []        

        eval_data_['acc_1_h'] = []
        eval_data_['acc_2_h'] = []        
        
        eval_data_['reproj_max_error_h'] = []
        eval_data_['reproj_inliers_h'] = []
        eval_data_['reproj_valid_h'] = []
        eval_data_['reproj_prec_h'] = []
        eval_data_['reproj_recall_h'] = []
            
        with progress_bar('Completion') as p:
            for i in p.track(range(n)):            
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')

                reproj_max_err =[]
                inl_sum = []
                avg_prec = 0
                avg_recall = 0
                valid_matches = None

                if os.path.isfile(pipe_f):
                    out_data = decompress_pickle(pipe_f)

                    pts1 = out_data['pt1']
                    pts2 = out_data['pt2']
                                            
                    if torch.is_tensor(pts1):
                        pts1 = pts1.detach().cpu().numpy()
                        pts2 = pts2.detach().cpu().numpy()

                    if use_scale:
                        scales = dataset_data['im_pair_scale'][i]
                    else:
                        scales = np.asarray([[1.0, 1.0], [1.0, 1.0]])
                    
                    pts1 = pts1 * scales[0]
                    pts2 = pts2 * scales[1]
                        
                    nn = pts1.shape[0]
                                                
                    if nn < 4:
                        H = None
                    else:
                        H = torch.tensor(cv2.findHomography(pts1, pts2, 0)[0], device=device)

                    if nn > 0:
                        H_gt = torch.tensor(dataset_data['H'][i], device=device)
                        H_inv_gt = torch.tensor(dataset_data['H_inv'][i], device=device)
                        
                        pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
                        pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))
                        
                        pt1_reproj = H_gt @ pt1_
                        pt1_reproj = pt1_reproj[:2] / pt1_reproj[2].unsqueeze(0)
                        d1 = ((pt2_[:2] - pt1_reproj)**2).sum(0).sqrt()
                        
                        pt2_reproj = H_inv_gt @ pt2_
                        pt2_reproj = pt2_reproj[:2] / pt2_reproj[2].unsqueeze(0)
                        d2 = ((pt1_[:2] - pt2_reproj)**2).sum(0).sqrt()
                        
                        valid_matches = torch.ones(nn, device=device, dtype=torch.bool)
                        
                        if dataset_data['im1_use_mask'][i]:
                            valid_matches = valid_matches & ~invalid_matches(dataset_data['im1_mask'][i], dataset_data['im2_full_mask'][i], pts1, pts2, rad)

                        if dataset_data['im2_use_mask'][i]:
                            valid_matches = valid_matches & ~invalid_matches(dataset_data['im2_mask'][i], dataset_data['im1_full_mask'][i], pts2, pts1, rad)
                                                
                        reproj_max_err_ = torch.maximum(d1, d2)                                
                        reproj_max_err = reproj_max_err_[valid_matches]
                        inl_sum = (reproj_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)
                        avg_prec = inl_sum.type(torch.double).mean()/nn
                                                
                        if pipe_name_base==pipe_name_root:
                            recall_normalizer = torch.tensor(inl_sum, device=device)
                        else:
                            recall_normalizer = torch.tensor(eval_data[pipe_name_root]['reproj_inliers_h'][i], device=device)
                        avg_recall = inl_sum.type(torch.double) / recall_normalizer
                        avg_recall[~avg_recall.isfinite()] = 0
                        avg_recall = avg_recall.mean()
                        
                        reproj_max_err = reproj_max_err_.detach().cpu().numpy()
                        valid_matches = valid_matches.detach().cpu().numpy()
                        inl_sum = inl_sum.detach().cpu().numpy()
                        avg_prec = avg_prec.item()
                        avg_recall = avg_recall.item()
                else:
                    H = None
                    valid_matches = torch.zeros(nn, device=device, dtype=torch.bool)
                                        
                if H is None:
                    # eval_data_['err_plane_1_h'].append([])
                    # eval_data_['err_plane_2_h'].append([])

                    eval_data_['acc_1_h'].append(np.inf) 
                    eval_data_['acc_2_h'].append(np.inf)        
                else:
                    heat1 = homography_error_heat_map(H_gt, H, torch.tensor(dataset_data['im1_full_mask'][i], device=device))
                    heat2 = homography_error_heat_map(H_inv_gt, H.inverse(), torch.tensor(dataset_data['im2_full_mask'][i], device=device))
                    
                    eval_data_['acc_1_h'].append(heat1[heat1 != -1].mean().detach().cpu().numpy()) 
                    eval_data_['acc_2_h'].append(heat2[heat2 != -1].mean().detach().cpu().numpy())       

                    # eval_data_['err_plane_1_h'].append(heat1.type(torch.half).detach().cpu().numpy())
                    # eval_data_['err_plane_2_h'].append(heat2.type(torch.half).detach().cpu().numpy())

                    if save_acc_images:
                        pipe_img_save_base = os.path.join(pipe_img_save, 'base')
                        os.makedirs(pipe_img_save_base, exist_ok=True)
                        iname = os.path.splitext(dataset_data['im1'][i])[0] + '_' + os.path.splitext(dataset_data['im2'][i])[0]
    
                        pipe_img_save1 = os.path.join(pipe_img_save_base, iname + '_1' + save_ext)
                        if not (os.path.isfile(pipe_img_save1) and not force):
                            im1s = os.path.join(bench_path,'planar',dataset_data['im1'][i])
                            colorize_plane(im1s, heat1, cmap_name='viridis', max_val=45, cf=0.7, save_to=pipe_img_save1)
    
                        pipe_img_save2 = os.path.join(pipe_img_save_base, iname + '_2' + save_ext)
                        if not (os.path.isfile(pipe_img_save2) and not force):
                            im2s = os.path.join(bench_path,'planar',dataset_data['im2'][i])
                            colorize_plane(im2s, heat2, cmap_name='viridis', max_val=45, cf=0.7, save_to=pipe_img_save2)
   
                eval_data_['reproj_max_error_h'].append(reproj_max_err)  
                eval_data_['reproj_inliers_h'].append(inl_sum)
                eval_data_['reproj_valid_h'].append(valid_matches)
                eval_data_['reproj_prec_h'].append(avg_prec)                           
                eval_data_['reproj_recall_h'].append(avg_recall)
                    
            aux = np.stack(([eval_data_['acc_1_h'], eval_data_['acc_2_h']]), axis=1)
            max_acc_err = np.max(aux, axis=1)        
            tmp = np.concatenate((aux, np.expand_dims(max_acc_err, axis=1)), axis=1)
    
            for a in angular_thresholds:       
                auc_1 = error_auc(np.squeeze(eval_data_['acc_1_h']), a)
                auc_2 = error_auc(np.squeeze(eval_data_['acc_2_h']), a)
                auc_max_12 = error_auc(np.squeeze(max_acc_err), a)
                eval_data_['pose_error_h_auc_' + str(a)] = np.asarray([auc_1, auc_2, auc_max_12])
                eval_data_['pose_error_h_acc_' + str(a)] = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]

                # accuracy using 1st, 2nd image as reference, and the maximum accuracy
                print(f"AUC@{str(a).ljust(2,' ')} (H) : {eval_data_['pose_error_h_auc_' + str(a)]}")
            
            eval_data_['reproj_global_prec_h'] = torch.tensor(eval_data_['reproj_prec_h'], device=device).mean().item()
            eval_data_['reproj_global_recall_h'] = torch.tensor(eval_data_['reproj_recall_h'], device=device).mean().item()
        
            print(f"precision (H) : {eval_data_['reproj_global_prec_h']}")
            print(f"recall (H) : {eval_data_['reproj_global_recall_h']}")

            eval_data[pipe_name_base] = eval_data_    
            compressed_pickle(save_to, eval_data)


def colorize_plane(ims, heat, cmap_name='viridis', max_val=45, cf=0.7, save_to='plane_acc.png'):
    im_gray = cv2.imread(ims, cv2.IMREAD_GRAYSCALE)
    im_gray = torch.tensor(im_gray, device=device).unsqueeze(0).repeat(3,1,1).permute(1,2,0)
    heat_mask = heat != -1
    heat_ = heat.clone()
    cmap = (colormaps[cmap_name](np.arange(0,(max_val + 1)) / max_val))[:,[2, 1, 0]]
    heat_[heat_ > max_val - 1] = max_val - 1
    heat_[heat_ == -1] = max_val
    cmap = torch.tensor(cmap, device=device)
    heat_im = cmap[heat_.type(torch.long)]
    heat_im = heat_im.type(torch.float) * 255
    blend_mask = heat_mask.unsqueeze(-1).type(torch.float) * cf
    imm = heat_im * blend_mask + im_gray.type(torch.float) * (1-blend_mask)                    
    cv2.imwrite(save_to, imm.type(torch.uint8).detach().cpu().numpy())   
 

def invalid_matches(mask1, mask2, pts1, pts2, rad):
    dmask2 = cv2.dilate(mask2.astype(np.ubyte),np.ones((rad*2+1, rad*2+1)))
    
    pt1 = torch.tensor(pts1, device=device).round().permute(1, 0)
    pt2 = torch.tensor(pts2, device=device).round().permute(1, 0)

    invalid_ = torch.zeros(pt1.shape[1], device=device, dtype=torch.bool)

    to_exclude = (pt1[0] < 0) | (pt2[0] < 0) | (pt1[0] >= mask1.shape[1]) | (pt2[0] >= mask2.shape[1]) | (pt1[1] < 0) | (pt2[1] < 0) | (pt1[1] >= mask1.shape[0]) | (pt2[1] >= mask2.shape[0])

    pt1 = pt1[:, ~to_exclude]
    pt2 = pt2[:, ~to_exclude]
    
    l1 = (pt1[1, :] * mask1.shape[1] + pt1[0,:]).type(torch.long)
    l2 = (pt2[1, :] * mask2.shape[1] + pt2[0,:]).type(torch.long)

    invalid_check = ~(torch.tensor(mask1, device=device).flatten()[l1]) & ~(torch.tensor(dmask2, device=device, dtype=torch.bool).flatten()[l2])
    invalid_[~to_exclude] = invalid_check 

    return invalid_


def homography_error_heat_map(H12_gt, H12, mask1):
    pt1 = mask1.argwhere()
    
    pt1 = torch.cat((pt1, torch.ones(pt1.shape[0], 1, device=device)), dim=1).permute(1,0)   

    pt2_gt_ = H12_gt.type(torch.float) @ pt1
    pt2_gt_ = pt2_gt_[:2] / pt2_gt_[2].unsqueeze(0)

    pt2_ = H12.type(torch.float) @ pt1
    pt2_ = pt2_[:2] / pt2_[2].unsqueeze(0)

    d1 = ((pt2_gt_ - pt2_)**2).sum(dim=0).sqrt()
    d1[~d1.isfinite()] = np.inf

    heat_map = torch.full(mask1.shape, -1, device=device, dtype=torch.float)
    heat_map[mask1] = d1
    
    return heat_map


def imc_phototourism_bench_setup(bench_path='bench_data', bench_imgs='imgs', save_to='imc_phototourism.pbz2', sample_size=800, seed=42, covisibility_range=[0.1, 0.7], new_sample=False, force=False, **dummy_args):
    
    save_to_full = os.path.join(bench_path, save_to)
    if os.path.isfile(save_to_full) and (not force):
        return decompress_pickle(save_to_full), save_to_full  

    rng = np.random.default_rng(seed=seed)    
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)

    file_to_download = os.path.join(bench_path, 'downloads', 'image-matching-challenge-2022.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1RyqsKr_X0Itkf34KUv2C7XP35drKSXht/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'imc_phototourism')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(out_dir)        
        
    scenes = sorted([scene for scene in os.listdir(os.path.join(out_dir, 'train')) if os.path.isdir(os.path.join(out_dir, 'train', scene))])

    scale_file = os.path.join(out_dir, 'train', 'scaling_factors.csv')
    scale_dict = {}
    with open(scale_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            scale_dict[row['scene']] = float(row['scaling_factor'])
        
    im1 = []
    im2 = []
    K1 = []
    K2 = []
    R = []
    T = []
    scene_scales = []
    covisibility = []
    
    if new_sample:
        sampled_idx = {}
    else:
        file_to_download = os.path.join(bench_path, 'downloads', 'imc_sampled_idx.pbz2')    
        if not os.path.isfile(file_to_download):    
            url = "https://drive.google.com/file/d/13AE6pbkJ8bNfVYjkxYvpVN6mkok98NuM/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)
        
        sampled_idx = decompress_pickle(file_to_download)
    
    with progress_bar('Completion') as p:
        for sn in p.track(range(len(scenes))):    
            scene = scenes[sn]
            
            work_path = os.path.join(out_dir, 'train', scene)
            pose_file  = os.path.join(work_path, 'calibration.csv')
            covis_file  = os.path.join(work_path, 'pair_covisibility.csv')
    
            if (not os.path.isfile(pose_file)) or (not os.path.isfile(covis_file)):
                continue
            
            im1_ = []
            im2_ = []
            covis_val = []
            with open(covis_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    pp = row['pair'].split('-')
                    im1_.append(os.path.join(scene, pp[0]))
                    im2_.append(os.path.join(scene, pp[1]))
                    covis_val.append(float(row['covisibility']))
    
            covis_val = np.asarray(covis_val)
            
            if new_sample:
                mask_val = (covis_val >= covisibility_range[0]) & (covis_val <= covisibility_range[1])

                n = covis_val.shape[0]
                
                full_idx = np.arange(n)  
                full_idx = full_idx[mask_val]
    
                m = full_idx.shape[0]
                
                idx = rng.permutation(m)[:sample_size]
                full_idx = np.sort(full_idx[idx])

                sampled_idx[scene] = full_idx
            else:
                full_idx = sampled_idx[scene]
                        
            covis_val = covis_val[full_idx]
            im1_ = [im1_[i] for i in full_idx]
            im2_ = [im2_[i] for i in full_idx]
            
            img_path = os.path.join(bench_path, bench_imgs, 'imc_phototourism')
            os.makedirs(os.path.join(img_path, scene), exist_ok=True)
            for im in im1_: shutil.copyfile(os.path.join(bench_path, 'imc_phototourism', 'train', scene, 'images', os.path.split(im)[1]) + '.jpg', os.path.join(img_path, im) + '.jpg')
            for im in im2_: shutil.copyfile(os.path.join(bench_path, 'imc_phototourism', 'train', scene, 'images', os.path.split(im)[1]) + '.jpg', os.path.join(img_path, im) + '.jpg')
    
            Kv = {}
            Tv = {}
            calib_file = os.path.join(work_path, 'calibration.csv')
            with open(calib_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    cam = os.path.join(scene, row['image_id'])
                    Kv[cam] = np.asarray([float(i) for i in row['camera_intrinsics'].split(' ')]).reshape((3, 3))
                    tmp = np.eye(4)
                    tmp[:3, :3] = np.asarray([float(i) for i in row['rotation_matrix'].split(' ')]).reshape((3, 3))
                    tmp[:3, 3] = np.asarray([float(i) for i in row['translation_vector'].split(' ')])
                    Tv[cam] = tmp
    
            K1_ = []
            K2_ = []
            T_ = []
            R_ = []
            scales_ = []
            for i in range(len(im1_)):
                K1_.append(Kv[im1_[i]])
                K2_.append(Kv[im2_[i]])
                T1 = Tv[im1_[i]]
                T2 = Tv[im2_[i]]
                T12 = np.matmul(T2, np.linalg.inv(T1))
                T_.append(T12[:3, 3])
                R_.append(T12[:3, :3])
                scales_.append(scale_dict[scene])
                
                
            im1 = im1 + im1_
            im2 = im2 + im2_
            K1 = K1 + K1_
            K2 = K2 + K2_
            T = T + T_
            R = R + R_
            scene_scales = scene_scales + scales_
            covisibility = covisibility + covis_val.tolist()  
        
    imc_data = {}
    imc_data['im1'] = im1
    imc_data['im2'] = im2
    imc_data['K1'] = np.asarray(K1)
    imc_data['K2'] = np.asarray(K2)
    imc_data['T'] = np.asarray(T)
    imc_data['R'] = np.asarray(R)
    imc_data['scene_scales'] = np.asarray(scene_scales)
    imc_data['covisibility'] = np.asarray(covisibility)
    imc_data['im_pair_scale'] = np.zeros((len(im1), 2, 2))
    
    compressed_pickle(os.path.join(bench_path, save_to_full), imc_data)
    if new_sample: compressed_pickle(os.path.join(bench_path, 'downloads', 'imc_sampled_idx.pbz2'), sampled_idx)
    
    return imc_data, save_to_full


def count_pipe_match(pipe, dataset_data,  dataset_name, bench_path='bench_data', bench_res='res', save_to='res_count.pbz2', force=False):
    warnings.filterwarnings("ignore", category=UserWarning)

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    n = len(dataset_data['im1'])
    
    pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
    pipe_name_base_small = ''
    pipe_name_root = os.path.join(pipe_name_base, pipe[0].get_id())

    for pipe_module in pipe:
        pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
        pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())

        print('Pipeline: ' + pipe_name_base_small)

        if (pipe_name_base in eval_data.keys()) and not force:
            eval_data_ = eval_data[pipe_name_base]
            print(f"filtered {round(eval_data_['filtered_avg']*100*100)/100}%, {eval_data_['matches_avg']} matches on average")
            continue
                
        eval_data_ = {}
        eval_data_['matches'] = []
        eval_data_['filtered'] = []

        with progress_bar('Counting completion') as p:
            for i in p.track(range(n)):            
                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')

                if os.path.isfile(pipe_f):
                    out_data = decompress_pickle(pipe_f)

                    pts1 = out_data['pt1']
                                            
                    if torch.is_tensor(pts1):
                        pts1 = pts1.detach().cpu().numpy()
                        
                    nn = pts1.shape[0]
                    eval_data_['matches'].append(nn)
                    
                    if pipe_name_base==pipe_name_root:
                        filtered_normalizer = nn
                    else:
                        filtered_normalizer = eval_data[pipe_name_root]['matches'][i]
                        
                    if filtered_normalizer == 0: filtered = 0
                    else: filtered = (filtered_normalizer - nn) / filtered_normalizer
                    if not np.isfinite(filtered): filtered = 0                   
                    eval_data_['filtered'].append(filtered)
                else:
                    eval_data_['matches'].append(np.nan)
                    eval_data_['filtered'].append(np.nan)
                    
            eval_data_['matches'] = np.asarray(eval_data_['matches'])
            eval_data_['filtered'] = np.asarray(eval_data_['filtered'])                    
            
            valid = np.isfinite(eval_data_['matches'].astype(float))
            
            eval_data_['matches_avg'] = int(np.mean(eval_data_['matches'][valid]))
            eval_data_['filtered_avg'] = np.mean(eval_data_['filtered'][valid])
        
            print(f"filtered {round(eval_data_['filtered_avg']*100*100)/100}%, {eval_data_['matches_avg']} matches on average")

            eval_data[pipe_name_base] = eval_data_
            compressed_pickle(save_to, eval_data)


def show_pipe_other(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', bench_res='res', bench_plot='showcase', force=False, ext='.png', save_ext='.jpg', fig_min_size=960, fig_max_size=1280, pipe_select=[-2, -1], save_mode='as_bench', b_index=None, bench_mode='fundamental_matrix', use_scale=False):

    err_bound = [[0, 1], [1, 3], [3, 7], [7, 15], [15, np.Inf]]
    
    clr = np.asarray([
        [0  , 255,   0],
        [255, 128,   0],
        [255,   0,   0],
        [255,   0, 255],
        [  0,   0, 255],
        [ 96,  96,  96], # outside planes
        ]) / 255.0

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)    
    fig = plt.figure()    
        
    with progress_bar(bar_name + ' - pipeline completion') as p:
        for i in p.track(range(n)):
            im1 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im1'][i])[0]) + ext
            im2 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im2'][i])[0]) + ext

            pair_data = []            
            pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
            pipe_img_save_list = []
            for pipe_module in pipe:
                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
                pipe_img_save_list.append(pipe_module.get_id())

                pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')            
                pair_data.append(decompress_pickle(pipe_f))
            
            if pipe_select is None: pipe_select = np.arange(len(pair_data))

            for pp in pipe_select:
                ppn = pp
                if ppn < 0:
                    ppn = len(pipe_img_save_list) + 1 + ppn
                img_folder = [bench_path, bench_plot]
                if save_mode == 'as_bench':
                    img_folder = img_folder + [dataset_name] + [pipe_img_save_list[ii] for ii in np.arange(ppn)] + ['base']
                
                pipe_img_save = os.path.join(*img_folder)                                
                os.makedirs(pipe_img_save, exist_ok=True)
                
                ni = i
                if not (b_index is None): ni = b_index[i]
                
                if save_mode == 'as_bench':               
                    pipe_img_save = os.path.join(pipe_img_save, str(ni) + save_ext)
                else:
                    pipe_img_save = os.path.join(*img_folder, dataset_name + '_' + str(ni) + '_' +  '_'.join([pipe_img_save_list[ii] for ii in np.arange(ppn)]) + save_ext)
                    
                if os.path.isfile(pipe_img_save) and not force:
                    continue
                
                img1 = viz_utils.load_image(im1)
                img2 = viz_utils.load_image(im2)
                fig, axes = viz.plot_images([img1, img2], fig_num=fig.number)              
                   
                pt1 = pair_data[pp]['pt1']
                pt2 = pair_data[pp]['pt2']
                
                nn = pt1.shape[0]
                
                if use_scale == True:
                    scales = dataset_data['im_pair_scale'][i]
                else:
                    scales = np.asarray([[1.0, 1.0], [1.0, 1.0]])                        
                                
                spt1 = pt1 * torch.tensor(scales[0], device=device)
                spt2 = pt2 * torch.tensor(scales[1], device=device)

                pt1_ = torch.vstack((torch.clone(spt1.T), torch.ones((1, nn), device=device))).type(torch.float64)
                pt2_ = torch.vstack((torch.clone(spt2.T), torch.ones((1, nn), device=device))).type(torch.float64)
                                                
                if bench_mode == 'fundamental_matrix':
                
                    K1 = dataset_data['K1'][i]
                    K2 = dataset_data['K2'][i]
                    R_gt = dataset_data['R'][i]
                    t_gt = dataset_data['T'][i]            
        
                    F_gt = torch.tensor(K2.T, device=device, dtype=torch.float64).inverse() @ \
                           torch.tensor([[0, -t_gt[2], t_gt[1]],
                                        [t_gt[2], 0, -t_gt[0]],
                                        [-t_gt[1], t_gt[0], 0]], device=device) @ \
                           torch.tensor(R_gt, device=device) @ \
                           torch.tensor(K1, device=device, dtype=torch.float64).inverse()
                    F_gt = F_gt / F_gt.sum()
                    F_gt = F_gt
                        
                    l1_ = F_gt @ pt1_
                    d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()
                    
                    l2_ = F_gt.T @ pt2_
                    d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()

                    valid_matches = None
                else:
                    rad = 15
                    
                    H_gt = torch.tensor(dataset_data['H'][i], device=device)
                    H_inv_gt = torch.tensor(dataset_data['H_inv'][i], device=device)
                                        
                    pt1_reproj = H_gt @ pt1_
                    pt1_reproj = pt1_reproj[:2] / pt1_reproj[2].unsqueeze(0)
                    d1 = ((pt2_[:2] - pt1_reproj)**2).sum(0).sqrt()
                    
                    pt2_reproj = H_inv_gt @ pt2_
                    pt2_reproj = pt2_reproj[:2] / pt2_reproj[2].unsqueeze(0)
                    d2 = ((pt1_[:2] - pt2_reproj)**2).sum(0).sqrt()
                    
                    valid_matches = torch.ones(nn, device=device, dtype=torch.bool)
                    
                    if dataset_data['im1_use_mask'][i]:
                        valid_matches = valid_matches & ~invalid_matches(dataset_data['im1_mask'][i], dataset_data['im2_full_mask'][i], spt1, spt2, rad)

                    if dataset_data['im2_use_mask'][i]:
                        valid_matches = valid_matches & ~invalid_matches(dataset_data['im2_mask'][i], dataset_data['im1_full_mask'][i], spt2, spt1, rad)
                                            
                err = torch.maximum(d1, d2) 
                err[~torch.isfinite(err)] = np.Inf
                if not (valid_matches is None):
                    err[~valid_matches] = np.nan
                
                mask = torch.isnan(err)
                if torch.any(mask): 
                    viz.plot_matches(pt1[mask], pt2[mask], color=clr[-1], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
                                
                for j in reversed(range(len(err_bound))):
                    mask = (err >= err_bound[j][0]) & (err < err_bound[j][1])
                    if torch.any(mask): 
                        viz.plot_matches(pt1[mask], pt2[mask], color=clr[j], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
    
                fig_dpi = fig.get_dpi()
                fig_sz = [fig.get_figwidth() * fig_dpi, fig.get_figheight() * fig_dpi]
    
                fig_cz = min(fig_sz)
                if fig_cz < fig_min_size:
                    fig_sz[0] = fig_sz[0] / fig_cz * fig_min_size
                    fig_sz[1] = fig_sz[1] / fig_cz * fig_min_size
    
                fig_cz = max(fig_sz)
                if fig_cz > fig_min_size:
                    fig_sz[0] = fig_sz[0] / fig_cz * fig_max_size
                    fig_sz[1] = fig_sz[1] / fig_cz * fig_max_size
                    
                fig.set_size_inches(fig_sz[0] / fig_dpi, fig_sz[1]  / fig_dpi)
    
                viz.save_plot(pipe_img_save, fig)
                viz.clear_plot(fig)
    plt.close(fig)


def collect_pipe_time(pipe, dataset_data,  dataset_name, bench_path='bench_data', bench_res='res', save_to='res_time.pbz2', force=False):
    warnings.filterwarnings("ignore", category=UserWarning)

    if os.path.isfile(save_to):
        eval_data = decompress_pickle(save_to)
    else:
        eval_data = {}
        
    n = len(dataset_data['im1'])    

    for l in range(len(pipe)):            
        pipe_name_final = os.path.join(bench_path, bench_res, dataset_name)
        pipe_name_final_small = ''

        for pipe_module in pipe[:l+1]:
            pipe_name_final = os.path.join(pipe_name_final, pipe_module.get_id())
            pipe_name_final_small = os.path.join(pipe_name_final_small, pipe_module.get_id())
        
        pipe_name_base = os.path.join(bench_path, bench_res, dataset_name)
        pipe_name_base_small = ''
    
        pipe_mode = []
        ttable = [[] for i in range(n)]

        print('Pipeline: ' + pipe_name_final_small)

        if (pipe_name_final in eval_data.keys()) and not force:
            eval_data_ = eval_data[pipe_name_final]
            print(f"running time: {np.round(eval_data_['running_time_avg']*100)/100} s")
            print(f"running time percentage: {np.round(eval_data_['running_time_pct_avg']*100*100)/100} %")
            continue

        eval_data_ = {}
    
        for pipe_module in pipe[:l+1]:
            pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())
            pipe_name_base_small = os.path.join(pipe_name_base_small, pipe_module.get_id())
        
            if hasattr(pipe_module, 'placeholder'):
                if pipe_module.placeholder == 'head': pipe_mode.append(0)
                elif pipe_module.placeholder == 'ransac': pipe_mode.append(2)
            else: pipe_mode.append(1)
                                
            with progress_bar('Running times collection completion') as p:
                for i in p.track(range(n)):
                    pipe_f = os.path.join(pipe_name_base, 'base', str(i) + '.pbz2')
    
                    if os.path.isfile(pipe_f):
                        out_data = decompress_pickle(pipe_f)
    
                        if 'running_time' in out_data.keys():
                            r = out_data['running_time'][0]
                        else:
                            r = np.nan
    
                        ttable[i].append(r)
    
        pipe_mode = np.asarray(pipe_mode)
    
        ttable = np.asarray(ttable)
        mask = np.all(np.isfinite(ttable), axis=1)
        
        gtable = ttable[mask]
        btable = gtable[:, np.squeeze(np.argwhere(pipe_mode==0))]
        ftable = gtable[:, np.squeeze(np.argwhere(pipe_mode==1))]
        rtable = gtable[:, np.squeeze(np.argwhere(pipe_mode==2))]
    
        if len(btable.shape) < 2: btable = np.expand_dims(btable, -1)
        if len(ftable.shape) < 2: ftable = np.expand_dims(ftable, -1)
        if len(rtable.shape) < 2: rtable = np.expand_dims(rtable, -1)
    
        btable = np.sum(btable, axis=-1)
        ftable = np.sum(ftable, axis=-1)
        rtable = np.sum(rtable, axis=-1)
        
        qtable = np.stack((btable, ftable, rtable, btable + ftable + rtable), axis=1)
        ptable = np.stack((btable/btable, ftable/btable, rtable/btable,  (btable + ftable + rtable) / btable ), axis=1)
    
        ave_qtable = np.mean(qtable, axis=0)
        ave_ptable = np.mean(ptable, axis=0)
    
        eval_data_['running_time'] = qtable
        eval_data_['running_time_pct'] = ptable
    
        eval_data_['running_time_avg'] = ave_qtable
        eval_data_['running_time_pct_avg'] = ave_ptable

        print(f"running time: {np.round(eval_data_['running_time_avg']*100)/100} s")
        print(f"running time percentage: {np.round(eval_data_['running_time_pct_avg']*100*100)/100} %")
    
        eval_data[pipe_name_base] = eval_data_
        compressed_pickle(save_to, eval_data)
