import torch
import numpy as np
import cv2
from multiprocessing import Pool as ThreadPool 

def tocuda(data):
	# convert tensor data in dictionary to cuda when it is a tensor
	for key in data.keys():
		if type(data[key]) == torch.Tensor:
			data[key] = data[key].cuda()
	return data

def get_pool_result(num_processor, fun, args):
    pool = ThreadPool(num_processor)
    pool_res = pool.map(fun, args)
    pool.close()
    pool.join()
    return pool_res

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M
    
def torch_skew_symmetric(v):

    zero = torch.zeros_like(v[:, 0])

    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)

    return M



def adjust_lr(optimizer, init_lr, gamma):
	for p in optimizer.param_groups:
		p['lr'] = init_lr * gamma
	return optimizer

def tocuda(data):
	# convert tensor data in dictionary to cuda when it is a tensor
	for key in data.keys():
		if type(data[key]) == torch.Tensor:
			data[key] = data[key].cuda()
	return data

def denorm(x, T):
    x = x * np.array([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
    return x

def estimate_pose_norm_kpts(kpts0, kpts1, thresh=1e-3, conf=0.99999):
	if len(kpts0) < 5:
		return None

	E, mask = cv2.findEssentialMat(
	kpts0, kpts1, np.eye(3), threshold=thresh, prob=conf,
	method=cv2.RANSAC)

	assert E is not None

	best_num_inliers = 0
	new_mask = mask
	ret = None
	for _E in np.split(E, len(E) / 3):
		n, R, t, mask_ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
		if n > best_num_inliers:
			best_num_inliers = n
			ret = (R, t[:, 0], mask.ravel() > 0)

	return ret

def estimate_pose_from_E(kpts0, kpts1, mask, E):
    assert E is not None
    mask = mask.astype(np.uint8)
    E = E.astype(np.float64)
    kpts0 = kpts0.astype(np.float64)
    kpts1 = kpts1.astype(np.float64)
    I = np.eye(3).astype(np.float64)

    best_num_inliers = 0
    ret = None

    for _E in np.split(E, len(E) / 3):

        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, I, 1e9, mask=mask)

        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs