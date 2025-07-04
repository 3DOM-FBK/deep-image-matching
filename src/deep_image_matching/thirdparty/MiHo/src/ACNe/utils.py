# Filename: utils.py
# License: LICENSES/LICENSE_UVIC_EPFL

import gzip
import pickle
import numpy as np
import h5py

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def denorm_points(x, T):
    x = (x - np.array([T[0,2], T[1,2]])) / np.asarray([T[0,0], T[1,1]])
    return x


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


def norm_points(x):
    x_mean = np.mean(x, axis=0)
    dist = x - x_mean
    meandist = np.sqrt((dist**2).sum(axis=1)).mean()
    scale = np.sqrt(2) / meandist
    T = np.zeros([3,3])
    T[0,0], T[1,1], T[2,2] = scale, scale, 1
    T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
    x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
    return x, T


def norm_points_with_T(x, T):
    x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
    return x


def savepklz(data_to_dump, dump_file_full_name, force_run=False):
    ''' Saves a pickle object and gzip it '''

    if not force_run:
        raise RuntimeError("This function should no longer be used!")

    with gzip.open(dump_file_full_name, 'wb') as out_file:
        pickle.dump(data_to_dump, out_file)


def loadpklz(dump_file_full_name, force_run=False):
    ''' Loads a gziped pickle object '''

    if not force_run:
        raise RuntimeError("This function should no longer be used!")

    with gzip.open(dump_file_full_name, 'rb') as in_file:
        dump_data = pickle.load(in_file)

    return dump_data


def saveh5(dict_to_dump, dump_file_full_name):
    ''' Saves a dictionary as h5 file '''

    with h5py.File(dump_file_full_name, 'w') as h5file:
        if isinstance(dict_to_dump, list):
            for i, d in enumerate(dict_to_dump):
                newdict = {'dict' + str(i): d}
                writeh5(newdict, h5file)
        else:
            writeh5(dict_to_dump, h5file)


def writeh5(dict_to_dump, h5node):
    ''' Recursive function to write dictionary to h5 nodes '''

    for _key in dict_to_dump.keys():
        if isinstance(dict_to_dump[_key], dict):
            h5node.create_group(_key)
            cur_grp = h5node[_key]
            writeh5(dict_to_dump[_key], cur_grp)
        else:
            h5node[_key] = dict_to_dump[_key]


def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    try:
        with h5py.File(dump_file_full_name, 'r') as h5file:
            dict_from_file = readh5(h5file)
    except Exception as e:
        print("Error while loading {}".format(dump_file_full_name))
        raise e

    return dict_from_file


def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key].value

    return dict_from_file


#
# utils.py ends here
