# Filename: test.py
# License: LICENSES/LICENSE_UVIC_EPFL

import datetime
import os
import sys
import time

import numpy as np
from parse import parse
import getpass

import cv2
from six.moves import xrange
from .transformations import quaternion_from_matrix
from .utils import loadh5, saveh5

# import *SAC here
username = getpass.getuser()
try:
    sys.path.insert(0, "/home/{}/data/gitdata/magsac-wrapper/python".format(username))
    from wrapper_magsac import magsacFindEssentialMat
    print("Successfully import magsac")
except Exception:
#   print("Warning: failed to import magsac")
    pass
    
try:
    sys.path.insert(
        0, "/home/{}/data/gitdata/usac-wrapper/python".format(username))
    from wrapper_usac import usacFindEssentialMat
    print("Successfully import usac")
except Exception:
#   print("Warning: failed to import usac")
    pass
    
try:
    import pyransac
    print("Successfully import pyransac")
except Exception: 
#   print("Warning: failed to import pyransac")
    pass
    
try:
    import pygcransac
    print("Successfully import pygcransac")
except Exception:
#  print("Warning: failed to import pygcransac")
   pass

from multiprocessing import Pool as ThreadPool 
import multiprocessing as mp


def get_pool_result(num_processor, fun, args):
    pool = ThreadPool(num_processor)
    pool_res = pool.map(fun, args)
    pool.close()
    pool.join()
    return pool_res


def denorm_points(x, T):
    x = (x - np.array([T[0,2], T[1,2]])) / np.asarray([T[0,0], T[1,1]])
    return x


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):

    # from Utils.transformations import quaternion_from_matrix

    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t


def eval_nondecompose(p1s, p2s, E_hat, dR, dt, scores):

    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(p1s.dtype)
    mask_new = None
    num_inlier = 0
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E_hat, p1s_good, p2s_good)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            print("Failed in evaluation")
            print(R)
            print(t)
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    # Change mask type
    mask = mask.flatten().astype(bool)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated


def eval_decompose_F(p1s, p2s, dR, dt, K1, K2, mask=None, method=cv2.LMEDS, probs=None,
                   weighted=False, use_prob=True, idx=None):

    # import wrappers
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    # cancel it because GCRANSAC
    # assert method.endswith("_F")
    if p1s_good.shape[0] >= 8:
        if method == "USAC_F":
            # using usac wrapper
            F, mask_new = usacFindEssentialMat(
                p1s_good, p2s_good, method=method, threshold=2.0,
                probs=probs_good, weighted=weighted, use_prob=use_prob)
            E = None
        elif method == "PYRANSAC_F":
            threshold = 1.0
            F, mask_new = pyransac.findFundamentalMatrix(
                p1s_good, p2s_good, threshold, 0.999, 100000, 0, 'sampson', True)
            mask_new = mask_new.reshape(-1, 1).astype(np.uint8)
            E = None
        elif method == "MAGSAC_F":
            # using magsac wrapper: 
            # Funda Mat: method="MAGSAC_F"; Essen Mat: method="MAGSAC_E" 
            F, mask_new = magsacFindEssentialMat(
                p1s_good, p2s_good, method=method, threshold=2.0,
                probs=probs_good, weighted=weighted, use_prob=use_prob, idx=idx) 
            E = None
        elif method == "RANSAC_F":
            # using pyransac by disabling degeneracy check
            # Better performance than opencv's ransac 
            threshold = 1.0
            F, mask_new = pyransac.findFundamentalMatrix(
                p1s_good, p2s_good, threshold, 0.999, 100000, 0, 'sampson',
                True, enable_degeneracy_check=False)
            mask_new = mask_new.reshape(-1, 1).astype(np.uint8)
            E = None
            # using opencv
            # F, mask_new = cv2.findFundamentalMat(
            #     p1s_good, p2s_good, cv2.FM_RANSAC, 3.0, 0.999)
            # E = None
        elif method == "GCRANSAC_F":
            w1 = int(K1[0, 2] * 2 + 1.0)
            h1 = int(K1[1, 2] * 2 + 1.0)
            w2 = int(K2[0, 2] * 2 + 1.0)
            h2 = int(K2[1, 2] * 2 + 1.0)
            F, mask_new = pygcransac.findFundamentalMatrix(
                p1s_good, p2s_good, h1, w1, h2, w2, threshold=0.5)
            mask_new = mask_new.reshape(-1, 1).astype(np.uint8)
            E = None
        else:
            raise ValueError("wrong method!")

        # convert to E if there is a F
        if F is not None:
            if F.shape[0] != 3:
                F = np.split(F, len(F) / 3)[0]
            E = np.matmul(np.matmul(K2.T,F), K1)
            E = E.astype(np.float64)
            # go back calibrated
            p1s_good = (p1s_good - np.array([K1[0, 2], K1[1, 2]])) / K1[0,0]
            p2s_good = (p2s_good - np.array([K2[0, 2], K2[1, 2]])) / K2[0,0]

        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good, p2s_good, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    # print("err_q: {} err_t: {}".format(err_q, err_t))

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated


def eval_decompose(p1s, p2s, dR, dt, threshold=0.001, mask=None, method=cv2.LMEDS, probs=None,
                   weighted=False, use_prob=True):

    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    if p1s_good.shape[0] >= 5:
        if method in ["RANSAC", cv2.RANSAC]:
            E, mask_new = cv2.findEssentialMat(
                p1s_good, p2s_good, method=method, threshold=threshold) # threshold=0.001--roughly 1 / f
        elif method == "USAC":
            E, mask_new = usacFindEssentialMat(
                p1s_good, p2s_good, method=method + "_E", threshold=threshold,
                probs=probs_good, weighted=weighted, use_prob=use_prob)
        elif method == "MAGSAC":
            # using magsac wrapper 
            E, mask_new = magsacFindEssentialMat(
                p1s_good, p2s_good, method=method + "_E", threshold=threshold,
                probs=probs_good, weighted=weighted, use_prob=use_prob)
        else:
            raise ValueError("Wrong method")

        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good, p2s_good, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated


def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = len(x1)
    if len(x2) != n:
        raise ValueError("Number of points don't match.")

    # make homogeneous
    ones = np.ones((n, 1))
    x1 = np.concatenate([x1, ones], axis=1)
    x2 = np.concatenate([x2, ones], axis=1)

    # build matrix for equations
    A = np.matmul(x2.reshape(n, 3, 1), x1.reshape(n, 1, 3)).reshape(n, 9)

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F / F[2, 2]


def eval_decompose_8points(p1s, p2s, dR, dt, mask=None, method=None):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    num_inlier = 0
    mask_new = None
    if p1s_good.shape[0] >= 8:
        E = compute_fundamental(p1s_good, p2s_good)
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E, p1s_good, p2s_good)
        err_q, err_t = evaluate_R_t(dR, dt, R, t)
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated


def test_sample(args):
    _x1, _x2, _dR, _dt, e_hat_out, y_hat_out, y_g_hat_out, y_w_hat_out, config, K1, K2, cur_val_idx, dump_test_cache_dir, test_list = args
    mode = config.run_mode
    # current validity from network
    _valid = y_hat_out.flatten()
    # choose top ones (get validity threshold)
    if config.weight_opt == "sigmoid_softmax":
        # use local attention to get inliers
        # _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
        # _mask_before = _valid >= max(0, _valid_th)
        th = 0.0000001 # Best in Essential case
        _mask_before = y_w_hat_out > th
    else:
        _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
        _mask_before = _valid >= max(0, _valid_th)


    # For every things to test
    _use_prob = True
    res_dict = {}
    for _test in test_list:
        dump_test_cache_fn = "{}_{}.txt".format(_test, cur_val_idx )
        dump_test_cache_fn = os.path.join(dump_test_cache_dir, dump_test_cache_fn)
        if os.path.exists(dump_test_cache_fn) and mode == "test":
            with open(dump_test_cache_fn, "r") as ifp:
                dump_res = ifp.read()
            dump_res = parse(
                "{err_q:e}, {err_t:e}, {num_inlier:d}\n", dump_res)
            _err_q = dump_res["err_q"]
            _err_t = dump_res["err_t"]
            _num_inlier = dump_res["num_inlier"]
        else:
            if _test == "ours":
                _eval_func = "non-decompose"
                _method = None
                _probs = None
                _weighted = False
            elif _test == "ours_ransac":
                _eval_func = "decompose"
                _method = cv2.RANSAC
                _probs = None
                _weighted = False
            elif _test == "ours_magsac":
                _eval_func = "decompose"
                _method = "MAGSAC" 
                _probs = None
                _weighted = False
            elif _test == "ours_ransac_F":
                _eval_func = "decompose_F"
                _method = "RANSAC_F"
                _probs = None
                _weighted = False
            elif _test == "ours_magsac_F":
                _eval_func = "decompose_F"
                _method = "MAGSAC_F"
                _probs = None
                _weighted = False
            else:
                raise ValueError("Not implemented")
            
            if config.weight_opt == "sigmoid_softmax":
                if _test.startswith("ours_magsac_F"):
                    th = 1e-6
                    _mask_before = y_w_hat_out > th
                elif _test.startswith("ours_ransac_F"):
                    th = 1e-5
                    _mask_before = y_w_hat_out > th

            if _eval_func == "non-decompose":
                _err_q, _err_t, _, _, _num_inlier, \
                    _ = eval_nondecompose(
                        _x1, _x2, e_hat_out, _dR, _dt, y_hat_out)
                _mask_after = _mask_before
            elif _eval_func == "decompose":
                # print("RANSAC loop with ours")
                time_start = datetime.datetime.now()
                _err_q, _err_t, _, _, _num_inlier, \
                    _mask_after = eval_decompose(
                        _x1, _x2, _dR, _dt, mask=_mask_before,
                        method=_method, probs=_probs,
                        weighted=_weighted, use_prob=_use_prob)
                time_end = datetime.datetime.now()
                time_diff = time_end - time_start
            elif _eval_func == "decompose_F":
                # print("RANSAC_F loop with ours")
                _x1_uncalib = _x1 * np.array([K1[0,0], K1[1,1]]) + np.array([K1[0,2], K1[1,2]])
                _x2_uncalib = _x2 * np.array([K2[0,0], K2[1,1]]) + np.array([K2[0,2], K2[1,2]])
                
                # _x1_uncalib = _x1 * f1 + np.array([cx1, cy1])
                # _x2_uncalib = _x2 * f2 + np.array([cx2, cy2])

                _err_q, _err_t, _, _, _num_inlier, \
                    _mask_after = eval_decompose_F(
                        _x1_uncalib, _x2_uncalib, _dR, _dt, K1, K2, mask=_mask_before,
                        method=_method, probs=_probs,
                        weighted=_weighted, use_prob=_use_prob, idx=cur_val_idx)
            # if mode == "test" and _method == "MAGSAC_F":
            # if mode == "test": # don't cache anymore
            #     # only cache for test
            #     with open(dump_test_cache_fn, "w") as ofp:
            #         ofp.write("{:e}, {:e}, {:d}\n".format(
            #             _err_q, _err_t, _num_inlier))

        res_dict[_test] = [cur_val_idx, _err_q, _err_t, _num_inlier]
    return res_dict


def test_process(mode, sess,
                 cur_global_step, merged_summary_op, summary_writer,
                 test_process_ins, 
                 img1, img2, r,
                 logits_mean, e_hat, loss, precision, recall,
                 last_e_hat, last_logit, last_x_in,
                 data,
                 res_dir, config, va_res_only=False):

    import tensorflow as tf
    txt_save_dir = config.save_test_dir
    if txt_save_dir == "":
        txt_save_dir = os.path.join(res_dir, mode)
    dump_test_cache_dir = os.path.join(txt_save_dir, "dump")
    if not os.path.exists(dump_test_cache_dir):
        os.makedirs(dump_test_cache_dir)

    if config.use_fundamental > 0:
        x, y, R, t, is_training, T1_in, T2_in, K1_in, K2_in = test_process_ins
    else:
        x, y, R, t, is_training = test_process_ins

    time_us = []
    time_ransac_us = []
    time_ransac = []

    inlier_us = []
    inlier_ransac = []
    inlier_ransac_us = []

    if mode == "test":
        print("[{}] {}: Start testing".format(config.data_tr, time.asctime()))

    # Unpack some references
    xs = data["xs"]
    ys = data["ys"]
    Rs = data["Rs"]
    ts = data["ts"]
    img1s = data["img1s"]
    cx1s = data["cx1s"]
    cy1s = data["cy1s"]
    f1s = data["f1s"]
    img2s = data["img2s"]
    cx2s = data["cx2s"]
    cy2s = data["cy2s"]
    f2s = data["f2s"]
    T1s = data["T1s"]
    T2s = data["T2s"]
    K1s = data["K1s"]
    K2s = data["K2s"]
    # ratios = data["ratios"]
    # mutuals = data["mutuals"]

    # Validation
    num_sample = len(xs)

    test_list = []
    F_suffix = "_F" if config.use_fundamental>0 else ""
    if va_res_only:
        test_list += [
            "ours",
            # "ours_ransac{}".format(F_suffix),
        ]
    else:
        test_list += [
            "ours",
            "ours_ransac{}".format(F_suffix),
            # "ours_magsac{}".format(F_suffix),
        ]

    eval_res = {}
    measure_list = ["err_q", "err_t", "num"]
    for measure in measure_list:
        eval_res[measure] = {}
        for _test in test_list:
            eval_res[measure][_test] = np.zeros(num_sample)

    e_hats = []
    y_hats = []
    precisions = []
    recalls = []
    losses = []
    last_e_hats = [] 
    last_y_hats = []
    last_x_ins = []
    softmax_logits = []
    final_weights = []
    if config.weight_opt == "sigmoid_softmax":
        final_weight = last_logit[2]
        softmax_logit = last_logit[1]
        last_logit = last_logit[0]
    else:
        softmax_logit = last_logit
        final_weight = last_logit
    # Run every test independently. might have different number of keypoints
    for idx_cur in xrange(num_sample):
        # Use minimum kp in batch to construct the batch
        _xs = np.array(
            xs[idx_cur][:, :, :]
        ).reshape(1, 1, -1, 4)
        _ys = np.array(
            ys[idx_cur][:, :]
        ).reshape(1, -1, 2)
        _dR = np.array(Rs[idx_cur]).reshape(1, 9)
        _dt = np.array(ts[idx_cur]).reshape(1, 3)
        # Create random permutation indices
        feed_dict = {
            x: _xs,
            y: _ys,
            R: _dR,
            t: _dt,
            is_training:  config.net_bn_test_is_training,
        }
        if config.use_fundamental > 0:
            T1s_b = np.array(
                [T1s[idx_cur]]
            )
            T2s_b = np.array(
                [T2s[idx_cur]]
            )
            K1s_b = np.array(
                [K1s[idx_cur]]
            )
            K2s_b = np.array(
                [K2s[idx_cur]]
            )
            feed_dict[T1_in] = T1s_b
            feed_dict[T2_in] = T2s_b
            feed_dict[K1_in] = K1s_b
            feed_dict[K2_in] = K2s_b

        fetch = {
            "last_e_hat": last_e_hat,
            "last_y_hat": last_logit,
            "last_x_in": last_x_in,
            "loss": loss,
            "precision": precision,
            "recall": recall, 
            "softmax_logit": softmax_logit,
            "final_weight": final_weight,
            # "summary": merged_summary_op,
            # "global_step": global_step,
        }
        # print("Running network for {} correspondences".format(
        #     _xs.shape[2]
        # ))
        time_start = datetime.datetime.now()
        res = sess.run(fetch, feed_dict=feed_dict)
        time_end = datetime.datetime.now()
        time_diff = time_end - time_start
        # print("Runtime in milliseconds: {}".format(
        #     float(time_diff.total_seconds() * 1000.0)
        # ))
        time_us += [time_diff.total_seconds() * 1000.0]
        # print("valid loss: {}".format(res["loss"]))
        # print("valid precision: {}".format(res["precision"]))
        # print("valid recall: {}".format(res["recall"]))

        last_e_hats.append(res["last_e_hat"])
        last_y_hats.append(res["last_y_hat"])
        last_x_ins.append(res["last_x_in"])
        softmax_logits.append(res["softmax_logit"])
        losses += [res["loss"]]
        precisions += [res["precision"]]
        recalls += [res["recall"]]
        final_weights += [res["final_weight"]]
        if config.vis_dir != "":
            dump_vis_file = os.path.join(
                config.vis_dir, "precision.npy")
            np.save(dump_vis_file, np.array(precisions))
            dump_vis_file = os.path.join(
                config.vis_dir, "recall.npy")
            np.save(dump_vis_file, np.array(recalls))

    results, pool_arg = [], []

    num_processor = int(mp.cpu_count() * 0.9)
    # num_processor = 12

    eval_step, eval_step_i = num_sample, 0 

    for cur_val_idx in xrange(num_sample):
        # _xs = xs[cur_val_idx][:, :, :].reshape(1, 1, -1, 4)
        # _ys = ys[cur_val_idx][:, :].reshape(1, -1, 2)
        _xs = np.array(last_x_ins[cur_val_idx], np.float64) # TO be compatible with iterative topk framework
        _dR = Rs[cur_val_idx]
        _dt = ts[cur_val_idx]
        e_hat_out = last_e_hats[cur_val_idx].flatten()
        y_hat_out = last_y_hats[cur_val_idx].flatten() # logit for local attention
        y_g_hat_out = softmax_logits[cur_val_idx].flatten() # logit for global attention
        y_w_hat_out = final_weights[cur_val_idx].flatten() # blended attention
        if len(y_hat_out) != _xs.shape[2]:
            y_hat_out = np.ones(_xs.shape[2])
        # Eval decompose for all pairs
        _xs = _xs.reshape(-1, 4)
        # x coordinates
        _x1 = _xs[:, :2]
        _x2 = _xs[:, 2:]
        # Convert x1,x2 if use fundamental
        # Make x1, x2 calibrated in any case.
        if config.use_fundamental > 0:
            _T1 = T1s[cur_val_idx]
            _T2 = T2s[cur_val_idx]
            _K1 = K1s[cur_val_idx]
            _K2 = K2s[cur_val_idx]
            # convert calibrated format
            _x1, _x2 = denorm_points(_x1, _T1), denorm_points(_x2, _T2)
            _x1, _x2 = denorm_points(_x1, _K1), denorm_points(_x2, _K2)

        # To get K1 and K2
        # Don't use _K1 and _K2 because K1s and K2s is null in case of EssentialMat
        cx1 = np.asarray(cx1s[cur_val_idx]).squeeze()
        cy1 = np.asarray(cy1s[cur_val_idx]).squeeze()
        cx2 = np.asarray(cx2s[cur_val_idx]).squeeze()
        cy2 = np.asarray(cy2s[cur_val_idx]).squeeze()
        f1 = np.asarray(f1s[cur_val_idx]).squeeze()
        f2 = np.asarray(f2s[cur_val_idx]).squeeze()
        # In case single f
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

        K1 = np.array([
            [np.asscalar(f1i), 0, np.asscalar(cx1)],
            [0, np.asscalar(f1j), np.asscalar(cy1)],
            [0, 0, 1],
        ])
        K2 = np.array([
            [np.asscalar(f2i), 0, np.asscalar(cx2)],
            [0, np.asscalar(f2j), np.asscalar(cy2)],
            [0, 0, 1],
        ])

        pool_arg += [
            (_x1, _x2, _dR, _dt, e_hat_out, y_hat_out, y_g_hat_out, y_w_hat_out, config, K1, K2, cur_val_idx, dump_test_cache_dir, test_list)
        ]
        eval_step_i += 1

        if eval_step_i % eval_step == 0:
            results += get_pool_result(num_processor, test_sample, pool_arg)
            pool_arg = []
    if len(pool_arg) > 0:
        results += get_pool_result(num_processor, test_sample, pool_arg)

    for result in results:
        for key, value in result.items():
            # key is _test, value is [cur_val_idx, _err, _err_t, num]
            # Load them in list
            eval_res["err_q"][key][value[0]] = value[1] 
            eval_res["err_t"][key][value[0]] = value[2] 
            eval_res["num"][key][value[0]] = value[3] 

    summaries = []
    ret_val = 0
    ret_val_ours_ransac = 0
    func_dict = {}
    func_dict["mean"] = np.mean
    func_dict["median"] = np.median
    criterions_dict = {}
    criterions_dict["precision"] = precisions
    criterions_dict["recall"] = recalls 
    criterions_dict["loss"] = losses
    
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)

    for key_cri in ["precision", "recall", "loss"]:
        for key_func in ["mean", "median"]:
            summaries.append(
                tf.Summary.Value(
                    tag="ErrorComputation/{}_{}".format(key_func, key_cri),
                    simple_value=func_dict[key_func](criterions_dict[key_cri])
                )
            )
            ofn = os.path.join(
                txt_save_dir, "{}_{}.txt".format(key_func, key_cri)
            )
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(
                    func_dict[key_func](criterions_dict[key_cri])))

    for _tag in test_list:
        for _sub_tag in measure_list:
            summaries.append(
                tf.Summary.Value(
                    tag="ErrorComputation/" + _tag,
                    simple_value=np.median(eval_res[_sub_tag][_tag])
                )
            )

            # For median error
            ofn = os.path.join(
                txt_save_dir, "median_{}_{}.txt".format(_sub_tag, _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(
                    np.median(eval_res[_sub_tag][_tag])))

        ths = np.arange(7) * 5
        cur_err_q = np.array(eval_res["err_q"][_tag]) * 180.0 / np.pi
        cur_err_t = np.array(eval_res["err_t"][_tag]) * 180.0 / np.pi
        # Get histogram
        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        t_acc_hist, _ = np.histogram(cur_err_t, ths)
        qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
        num_pair = float(len(cur_err_q))
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        t_acc_hist = t_acc_hist.astype(float) / num_pair
        qt_acc_hist = qt_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)
        t_acc = np.cumsum(t_acc_hist)
        qt_acc = np.cumsum(qt_acc_hist)
        # Store return val
        if _tag == "ours":
            ret_val = np.mean(qt_acc[:4])  # 1 == 5
        if _tag == "ours_ransac":
            ret_val_ours_ransac = np.mean(qt_acc[:4])  # 1 == 5
        for _idx_th in xrange(1, len(ths)):
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_q_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(q_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_t_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(t_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_qt_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(qt_acc[:_idx_th]),
                )
            ]
            # for q_auc
            ofn = os.path.join(
                txt_save_dir,
                "acc_q_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                txt_save_dir,
                "acc_t_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                txt_save_dir,
                "acc_qt_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    summary_writer.add_summary(
        tf.Summary(value=summaries), global_step=cur_global_step)

    if mode == "test":
        print("[{}] {}: End testing".format(
            config.data_tr, time.asctime()))

    # Return qt_auc20 of ours
    return ret_val, ret_val_ours_ransac

#
# test.py ends here
