# Adapted from OANet repo:  https://github.com/zjhthu/OANet.git

from __future__ import print_function
import sys
import os
import pickle
import h5py
import numpy as np
import cv2
import torch
import torch.utils.data as data

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

# from ACNe code
def compute_T_with_imagesize(w, h, f=None, ratio=1.0):
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5
    mean = np.array([cx, cy])
    if f is not None:
        f = f
    else:
        f = max(w - 1.0, h - 1.0) * ratio

    scale = 1.0 / f

    T = np.zeros((3, 3,))
    T[0, 0], T[1, 1], T[2, 2] = scale, scale, 1
    T[0, 2], T[1, 2] = -scale * mean[0], -scale * mean[1]

    return T.copy()


def norm_points_with_T(x, T):
    x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
    return x


def collate_fn(batch):
    batch_size = len(batch)
    # print("batch_size: {}".format(batch_size))
    numkps = np.array([sample['xs'].shape[1] for sample in batch])
    cur_num_kp = int(numkps.min())
    data = {}
    data['K1s'], data['K2s'], data['Rs'], \
        data['ts'], data['xs'], data['ys'], data['T1s'], data['T2s'] = [], [], [], [], [], [], [], []
    for sample in batch:
        data['K1s'].append(sample['K1'])
        data['K2s'].append(sample['K2'])
        data['T1s'].append(sample['T1'])
        data['T2s'].append(sample['T2'])
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp)
            data['xs'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])

    for key in ['K1s', 'K2s', 'Rs', 'ts', 'xs', 'ys', 'T1s', 'T2s']:
        data[key] = np.stack(data[key])
    if data["ys"].shape[-1] == 1:
        data["ys"] = np.repeat(data["ys"], 2, axis=-1)
    return data



class CorrespondencesDataset(data.Dataset):
    def __init__(self, filename, config, mode="train"):
        self.config = config
        self.data = None
        data_dir = "data_dump_oan"
        if filename == "oan_outdoor":
            fn = "yfcc-sift-2000-train.hdf5"
        elif filename == "oan_indoor":
            fn = "sun3d-sift-2000-train.hdf5"
        else:
            raise NotImplementedError
        self.filename = os.path.join(data_dir, fn)

    def norm_input(self, x):
        x_mean = np.mean(x, axis=0)
        dist = x - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3,3])
        T[0,0], T[1,1], T[2,2] = scale, scale, 1
        T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
        x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
        return x, T
    
    def __getitem__(self, index_in):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')
        index = str(index_in)

        
        xs = np.asarray(self.data['xs'][index])
        # print(xs.shape)
        ys = np.asarray(self.data['ys'][index])
        R = np.asarray(self.data['Rs'][index])
        t = np.asarray(self.data['ts'][index])
        side = []

        if self.config.prefiltering == "B":
            mask = np.asarray(self.data['mutuals'][index]).reshape(-1).astype(bool)
            xs = xs[:,mask,:]
            ys = ys[mask]
        elif self.config.prefiltering == "RB":
            mask_B = np.asarray(self.data['mutuals'][index]).reshape(-1).astype(bool)
            mask_R = np.asarray(self.data['ratios'][index]).reshape(-1)  < 0.8
            mask = np.all([mask_B, mask_R], axis=0)
            xs = xs[:,mask,:]
            ys = ys[mask]
            if len(side) > 0:
                side = side[mask]
        elif self.config.prefiltering == "":
            pass
        else:
            raise NotImplementedError

        if self.config.use_fundamental>0:
            cx1 = np.asarray(self.data['cx1s'][index]).squeeze()
            cy1 = np.asarray(self.data['cy1s'][index]).squeeze()
            cx2 = np.asarray(self.data['cx2s'][index]).squeeze()
            cy2 = np.asarray(self.data['cy2s'][index]).squeeze()
            f1 = np.asarray(self.data['f1s'][index]).squeeze()
            f2 = np.asarray(self.data['f2s'][index]).squeeze()
            # in case single f
            if f1.size == 2:
                f1i = f1[0]
                f1j = f1[1]
            else:
                f1i = f1
                f1j = f1

            if f2.size == 2:
                f2i = f2[0]
                f2j = f2[1]
            else:
                f2i = f2
                f2j = f2

                
            K1 = np.asarray([
                [f1i, 0, cx1],
                [0, f1j, cy1],
                [0, 0, 1]
                ])
            K2 = np.asarray([
                [f2i, 0, cx2],
                [0, f2j, cy2],
                [0, 0, 1]
                ])
            x1, x2 = xs[0,:,:2], xs[0,:,2:4]
            x1 = x1 * np.array([K1[0,0], K1[1,1]]) + np.array([K1[0,2], K1[1,2]])
            x2 = x2 * np.array([K2[0,0], K2[1,1]]) + np.array([K2[0,2], K2[1,2]])
            # norm input
            if self.config.use_fundamental == 1:
                # normal norm
                x1, T1 = self.norm_input(x1)
                x2, T2 = self.norm_input(x2)
            elif self.config.use_fundamental == 2:
                # img_size norm
                w1 = cx1 * 2 + 1.0
                h1 = cy1 * 2 + 1.0 
                T1 = compute_T_with_imagesize(w1, h1)
                w2 = cx2 * 2 + 1.0
                h2 = cy2 * 2 + 1.0 
                T2 = compute_T_with_imagesize(w2, h2)
                x1 = norm_points_with_T(x1, T1)
                x2 = norm_points_with_T(x2, T2)
            else:
                raise ValueError("worng norm tyep")
            
            xs = np.concatenate([x1,x2],axis=-1).reshape(1,-1,4)
        else:
            K1, K2 = np.zeros(1), np.zeros(1)
            T1, T2 = np.zeros(1), np.zeros(1)

        return {'K1':K1, 'K2':K2, 'R':R, 't':t, \
        'xs':xs, 'ys':ys, 'T1':T1, 'T2':T2, 'side':side}
        
    def reset(self):
        if self.data is not None:
            self.data.close()
        self.data = None

    def __len__(self):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')
            _len = len(self.data['xs'])
            self.data.close()
            self.data = None
        else:
            _len = len(self.data['xs'])
        return _len

    def __del__(self):
        if self.data is not None and not self.bool_acne_format:
            self.data.close()

