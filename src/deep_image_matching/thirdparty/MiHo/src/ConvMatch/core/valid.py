import torch
import numpy as np
import os
import cv2
from six.moves import xrange
from loss import MatchLoss
from evaluation import eval_nondecompose, eval_decompose
from utils import tocuda, get_pool_result


def valid_sample(args):
    _xs, _dR, _dt, _e_hat, _y_hat, _y_gt, config, = args
    _xs = _xs.reshape(-1, 4).astype('float64')
    _dR, _dt = _dR.astype('float64').reshape(3,3), _dt.astype('float64')
    _y_hat_out = _y_hat.flatten().astype('float64')
    e_hat_out = _e_hat.flatten().astype('float64')

    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = _y_hat_out
    # choose top ones (get validity threshold)
    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
    _mask_before = _valid >= max(0, _valid_th)

    if not config.use_ransac:
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_nondecompose(_x1, _x2, e_hat_out, _dR, _dt, _y_hat_out)
    else:
        # actually not use prob here since probs is None
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_decompose(_x1, _x2, _dR, _dt, mask=_mask_before, method=cv2.RANSAC, \
            probs=None, weighted=False, use_prob=True)
    if _R_hat is None:
        _R_hat = np.random.randn(3,3)
        _t_hat = np.random.randn(3,1)
    return [float(_err_q), float(_err_t), float(_num_inlier), _R_hat.reshape(1,-1), _t_hat.reshape(1,-1)]

def dump_res(measure_list, res_path, eval_res, tag):
    # dump test results
    for sub_tag in measure_list:
        # For median error
        ofn = os.path.join(res_path, "median_{}_{}.txt".format(sub_tag, tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(eval_res[sub_tag])))

    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    cur_err_t = np.array(eval_res["err_t"]) * 180.0 / np.pi
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
    for _idx_th in xrange(1, len(ths)):
        ofn = os.path.join(res_path, "acc_q_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
        ofn = os.path.join(res_path, "acc_t_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
        ofn = os.path.join(res_path, "acc_qt_auc{}_{}.txt".format(ths[_idx_th], tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    ofn = os.path.join(res_path, "all_acc_qt_auc20_{}.txt".format(tag))
    np.savetxt(ofn, np.maximum(cur_err_q, cur_err_t))
    ofn = os.path.join(res_path, "all_acc_q_auc20_{}.txt".format(tag))
    np.savetxt(ofn, cur_err_q)
    ofn = os.path.join(res_path, "all_acc_t_auc20_{}.txt".format(tag))
    np.savetxt(ofn, cur_err_t)

    # Return qt_auc20 
    ret_val = np.mean(qt_acc[:4])  # 1 == 5
    return ret_val

def denorm(x, T):
    x = (x - np.array([T[0,2], T[1,2]])) / np.asarray([T[0,0], T[1,1]])
    return x

def valid_process(mode, model, cur_global_step, data_loader, config):
    model.eval()
    match_loss = MatchLoss(config)
    loader_iter = iter(data_loader)

    # save info given by the network
    network_infor_list = ["geo_losses", "cla_losses", "l2_losses", 'precisions', 'recalls', 'f_scores']
    network_info = {info:[] for info in network_infor_list}

    results, pool_arg = [], []
    eval_step, eval_step_i, num_processor = 100, 0, 8
    with torch.no_grad(): 
        for test_data in loader_iter:
            test_data = tocuda(test_data)
            res_logits, res_e_hat = model(test_data)
            y_hat, e_hat = res_logits[-1], res_e_hat[-1]
            loss, geo_loss, cla_loss, l2_loss, prec, rec = match_loss.run(cur_global_step, test_data, y_hat, e_hat)
            info = [geo_loss, cla_loss, l2_loss, prec, rec, 2*prec*rec/(prec+rec+1e-15)]
            for info_idx, value in enumerate(info):
                network_info[network_infor_list[info_idx]].append(value)

            if config.use_fundamental:
                # unnorm F
                e_hat = torch.matmul(torch.matmul(test_data['T2s'].transpose(1,2), e_hat.reshape(-1,3,3)),test_data['T1s'])
                # get essential matrix from fundamental matrix
                e_hat = torch.matmul(torch.matmul(test_data['K2s'].transpose(1,2), e_hat.reshape(-1,3,3)),test_data['K1s']).reshape(-1,9)
                e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)

            for batch_idx in range(e_hat.shape[0]):
                test_xs = test_data['xs'][batch_idx].detach().cpu().numpy()
                if config.use_fundamental: # back to original
                    x1, x2 = test_xs[0,:,:2], test_xs[0,:,2:4]
                    T1, T2 = test_data['T1s'][batch_idx].cpu().numpy(), test_data['T2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, T1), denorm(x2, T2) # denormalize coordinate
                    K1, K2 = test_data['K1s'][batch_idx].cpu().numpy(), test_data['K2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, K1), denorm(x2, K2) # normalize coordiante with intrinsic
                    test_xs = np.concatenate([x1,x2],axis=-1).reshape(1,-1,4)
                
                pool_arg += [(test_xs, test_data['Rs'][batch_idx].detach().cpu().numpy(), \
                              test_data['ts'][batch_idx].detach().cpu().numpy(), e_hat[batch_idx].detach().cpu().numpy(), \
                              y_hat[batch_idx].detach().cpu().numpy(),  \
                              test_data['ys'][batch_idx,:,0].detach().cpu().numpy(), config)]

                eval_step_i += 1
                if eval_step_i % eval_step == 0:
                    results += get_pool_result(num_processor, valid_sample, pool_arg)
                    pool_arg = []
        if len(pool_arg) > 0:
            results += get_pool_result(num_processor, valid_sample, pool_arg)

    measure_list = ["err_q", "err_t", "num", 'R_hat', 't_hat']
    eval_res = {}
    for measure_idx, measure in enumerate(measure_list):
        eval_res[measure] =  np.asarray([result[measure_idx] for result in results])

    if config.res_path == '':
        config.res_path = os.path.join(config.log_path, mode)
    tag = "ours" if not config.use_ransac else "ours_ransac"
    ret_val = dump_res(measure_list, config.res_path, eval_res, tag)
    return [ret_val, np.mean(np.asarray(network_info['geo_losses'])), np.mean(np.asarray(network_info['cla_losses'])), \
        np.mean(np.asarray(network_info['l2_losses'])), np.mean(np.asarray(network_info['precisions'])), \
        np.mean(np.asarray(network_info['recalls'])), np.mean(np.asarray(network_info['f_scores']))]

def valid(data_loader, model, step, config):
    return valid_process("valid", model, step, data_loader, config)

