#! /usr/bin/env python3
import argparse
import random
import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from datasets import collate_fn, CorrespondencesDataset
from utils import (compute_pose_error, pose_auc, estimate_pose_norm_kpts, estimate_pose_from_E)

from model import CLNet
from config import get_config, print_usage

torch.set_grad_enabled(False)
torch.manual_seed(0)

def test(opt, thr=1e-4, use_ransac=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    test_dataset = CorrespondencesDataset(opt.data_te, opt)

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True, collate_fn=collate_fn)

    model = CLNet(opt)
    checkpoint = torch.load(opt.model_path, map_location=torch.device('cpu'))

    state_dict = {}

    '''Load a parallelly trained model'''
    for key in checkpoint['state_dict'].keys():
        key_new = key.split('module')[1][1:]
        state_dict[key_new] = checkpoint['state_dict'][key]

    '''Load a model trained on a single GPU'''
#    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    err_ts, err_Rs, precisions, matching_scores, num_corrects = [], [], [], [], []
    for idx, test_data in enumerate(tqdm(test_loader)):
        xs = test_data['xs'].to(device)
        ys = test_data['ys'].to(device)

        _, _, e_hat, y_hat = model(xs, ys)

        mkpts0 = xs.squeeze()[:, :2].cpu().detach().numpy()
        mkpts1 = xs.squeeze()[:, 2:].cpu().detach().numpy()

        mask = y_hat.squeeze().cpu().detach().numpy() < thr
        mask_kp0 = mkpts0[mask]
        mask_kp1 = mkpts1[mask]

        if use_ransac == True:
            file_name = '/aucs.txt'
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1)
        else:
            file_name = '/aucs_DLT.txt'
            e_hat = e_hat[-1].view(3, 3).cpu().detach().numpy()

            ret = estimate_pose_from_E(mkpts0, mkpts1, mask, e_hat)

        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            R_gt, t_gt = test_data['Rs'], test_data['ts']
            T_0to1 = torch.cat([R_gt.squeeze(), t_gt.squeeze().unsqueeze(-1)], dim=-1).numpy()
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        err_ts.append(err_t)
        err_Rs.append(err_R)

    # Write the evaluation results to disk.
    out_eval = {'error_t': err_ts,
                'error_R': err_Rs
                }

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)


    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]

    print('Evaluation Results (mean over {} pairs):'.format(len(test_loader)))
    print('AUC@5\t AUC@10\t AUC@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))

    np.savetxt(opt.output_dir + file_name, np.asarray(aucs))
    return np.asarray(aucs)

if __name__ == '__main__':
    opt, unparsed = get_config()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    aucs = test(opt, thr=opt.thr, use_ransac=opt.use_ransac)
