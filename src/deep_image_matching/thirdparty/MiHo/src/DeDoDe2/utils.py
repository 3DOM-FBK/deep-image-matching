import warnings
import numpy as np
import math
import cv2
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
import torch
from time import perf_counter


def get_best_device(verbose = False):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if verbose: print (f"Fastest device found is: {device}")
    return device


def recover_pose(E, kpts0, kpts1, K0, K1, mask):
    best_num_inliers = 0
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0_n = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1_n = (K1inv @ (kpts1-K1[None,:2,2]).T).T

    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)
    return ret



# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0 = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1 = (K1inv @ (kpts1-K1[None,:2,2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret


def get_grid(B,H,W, device = get_best_device()):
    x1_n = torch.meshgrid(
    *[
        torch.linspace(
            -1 + 1 / n, 1 - 1 / n, n, device=device
        )
        for n in (B, H, W)
    ]
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n

@torch.no_grad()
def finite_diff_hessian(f: tuple(["B", "H", "W"]), device = get_best_device()):
    dxx = torch.tensor([[0,0,0],[1,-2,1],[0,0,0]], device = device)[None,None]/2
    dxy = torch.tensor([[1,0,-1],[0,0,0],[-1,0,1]], device = device)[None,None]/4
    dyy = dxx.mT
    Hxx = F.conv2d(f[:,None], dxx, padding = 1)[:,0]
    Hxy = F.conv2d(f[:,None], dxy, padding = 1)[:,0]
    Hyy = F.conv2d(f[:,None], dyy, padding = 1)[:,0]
    H = torch.stack((Hxx, Hxy, Hxy, Hyy), dim = -1).reshape(*f.shape,2,2)
    return H

def finite_diff_grad(f: tuple(["B", "H", "W"]), device = get_best_device()):
    dx = torch.tensor([[0,0,0],[-1,0,1],[0,0,0]],device = device)[None,None]/2
    dy = dx.mT
    gx = F.conv2d(f[:,None], dx, padding = 1)
    gy = F.conv2d(f[:,None], dy, padding = 1)
    g = torch.cat((gx, gy), dim = 1)
    return g

def fast_inv_2x2(matrix: tuple[...,2,2], eps = 1e-10):
    return 1/(torch.linalg.det(matrix)[...,None,None]+eps) * torch.stack((matrix[...,1,1],-matrix[...,0,1],
                                                     -matrix[...,1,0],matrix[...,0,0]),dim=-1).reshape(*matrix.shape)

def newton_step(f:tuple["B","H","W"], inds, device = get_best_device()):
    B,H,W = f.shape
    Hess = finite_diff_hessian(f).reshape(B,H*W,2,2)
    Hess = torch.gather(Hess, dim = 1, index = inds[...,None].expand(B,-1,2,2))
    grad = finite_diff_grad(f).reshape(B,H*W,2)
    grad = torch.gather(grad, dim = 1, index = inds)
    Hessinv = fast_inv_2x2(Hess-torch.eye(2, device = device)[None,None])
    step = (Hessinv @ grad[...,None])
    return step[...,0]

@torch.no_grad()
def sample_keypoints(scoremap, num_samples = 8192, device = get_best_device(), use_nms = True, 
                     sample_topk = False, return_scoremap = False, sharpen = False, upsample = False,
                     increase_coverage = False, remove_borders = False):
    #scoremap = scoremap**2
    log_scoremap = (scoremap+1e-10).log()
    if upsample:
        log_scoremap = F.interpolate(log_scoremap[:,None], scale_factor = 3, mode = "bicubic", align_corners = False)[:,0]#.clamp(min = 0)
        scoremap = log_scoremap.exp()
    B,H,W = scoremap.shape
    if increase_coverage:
        weights = (-torch.linspace(-2, 2, steps = 51, device = device)**2).exp()[None,None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d((scoremap[:,None]+1e-6)*10000,weights[...,None,:], padding = (0,51//2))
        local_density = F.conv2d(local_density_x, weights[...,None], padding = (51//2,0))[:,0]
        scoremap = scoremap * (local_density+1e-8)**(-1/2)
    grid = get_grid(B,H,W, device=device).reshape(B,H*W,2)
    if sharpen:
        laplace_operator = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], device = device)/4
        scoremap = scoremap[:,None] - 0.5 * F.conv2d(scoremap[:,None], weight = laplace_operator, padding = 1)
        scoremap = scoremap[:,0].clamp(min = 0)
    if use_nms:
        scoremap = scoremap * (scoremap == F.max_pool2d(scoremap, (3, 3), stride = 1, padding = 1))
    if remove_borders:
        frame = torch.zeros_like(scoremap)
        # we hardcode 4px, could do it nicer, but whatever
        frame[...,4:-4, 4:-4] = 1
        scoremap = scoremap * frame
    if sample_topk:
        inds = torch.topk(scoremap.reshape(B,H*W), k = num_samples).indices
    else:
        inds = torch.multinomial(scoremap.reshape(B,H*W), num_samples = num_samples, replacement=False)
    kps = torch.gather(grid, dim = 1, index = inds[...,None].expand(B,num_samples,2))
    if return_scoremap:
        return kps, torch.gather(scoremap.reshape(B,H*W), dim = 1, index = inds)
    return kps

@torch.no_grad()
def jacobi_determinant(warp, certainty, R = 3, device = get_best_device(), dtype = torch.float32):
    t = perf_counter()
    *dims, _ = warp.shape
    warp = warp.to(dtype)
    certainty = certainty.to(dtype)
    
    dtype = warp.dtype
    match_regions = torch.zeros((*dims, 4, R, R), device = device).to(dtype)
    match_regions[:,1:-1, 1:-1] = warp.unfold(1,R,1).unfold(2,R,1)
    match_regions = rearrange(match_regions,"B H W D R1 R2 -> B H W (R1 R2) D") - warp[...,None,:]
    
    match_regions_cert = torch.zeros((*dims, R, R), device = device).to(dtype)
    match_regions_cert[:,1:-1, 1:-1] = certainty.unfold(1,R,1).unfold(2,R,1)
    match_regions_cert = rearrange(match_regions_cert,"B H W R1 R2 -> B H W (R1 R2)")[..., None]

    #print("Time for unfold", perf_counter()-t)
    #t = perf_counter()
    *dims, N, D = match_regions.shape
    # standardize:
    mu, sigma = match_regions.mean(dim=(-2,-1), keepdim = True), match_regions.std(dim=(-2,-1),keepdim=True)
    match_regions = (match_regions-mu)/(sigma+1e-6)
    x_a, x_b = match_regions.chunk(2,-1)
    

    A = torch.zeros((*dims,2*x_a.shape[-2],4), device = device).to(dtype)
    A[...,::2,:2] = x_a * match_regions_cert
    A[...,1::2,2:] = x_a * match_regions_cert

    a_block = A[...,::2,:2]
    ata = a_block.mT @ a_block
    #print("Time for ata", perf_counter()-t)
    #t = perf_counter()

    #atainv = torch.linalg.inv(ata+1e-5*torch.eye(2,device=device).to(dtype))
    atainv = fast_inv_2x2(ata)
    ATA_inv = torch.zeros((*dims, 4, 4), device = device, dtype = dtype)
    ATA_inv[...,:2,:2] = atainv
    ATA_inv[...,2:,2:] = atainv
    atb = A.mT @ (match_regions_cert*x_b).reshape(*dims,N*2,1)
    theta =  ATA_inv @ atb
    #print("Time for theta", perf_counter()-t)
    #t = perf_counter()

    J = theta.reshape(*dims, 2, 2)
    abs_J_det = torch.linalg.det(J+1e-8*torch.eye(2,2,device = device).expand(*dims,2,2)).abs() # Note: This should always be positive for correct warps, but still taking abs here
    abs_J_logdet = (abs_J_det+1e-12).log()
    B = certainty.shape[0]
    # Handle outliers
    robust_abs_J_logdet = abs_J_logdet.clamp(-3, 3) # Shouldn't be more that exp(3) \approx 8 times zoom
    #print("Time for logdet", perf_counter()-t)
    #t = perf_counter()

    return robust_abs_J_logdet

def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ]
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return torch.cat((x1_n.reshape(B,H,W,2),x2),dim=-1), prob

def unnormalize_coords(x_n,h,w):
    x = torch.stack(
        (w * (x_n[..., 0] + 1) / 2, h * (x_n[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x


def rotate_intrinsic(K, n):
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1.0 / scales[0], 1.0 / scales[1], 1.0])
    return np.dot(scales, K)

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


# From Patch2Pix https://github.com/GrumpyZhou/patch2pix
def get_depth_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize, mode=InterpolationMode.BILINEAR, antialias = False))
    return TupleCompose(ops)


def get_tuple_transform_ops(resize=None, normalize=True, unscale=False, clahe = False):
    ops = []
    if resize:
        ops.append(TupleResize(resize, antialias = True))
    if clahe:
        ops.append(TupleClahe())
    if normalize:
        ops.append(TupleToTensorScaled())
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)

class Clahe:
    def __init__(self, cliplimit = 2, blocksize = 8) -> None:
        self.clahe = cv2.createCLAHE(cliplimit,(blocksize,blocksize))
    def __call__(self, im):
        im_hsv = cv2.cvtColor(np.array(im),cv2.COLOR_RGB2HSV)
        im_v = self.clahe.apply(im_hsv[:,:,2])
        im_hsv[...,2] = im_v
        im_clahe = cv2.cvtColor(im_hsv,cv2.COLOR_HSV2RGB)
        return Image.fromarray(im_clahe)

class TupleClahe:
    def __init__(self, cliplimit = 8, blocksize = 8) -> None:
        self.clahe = Clahe(cliplimit,blocksize)
    def __call__(self, ims):
        return [self.clahe(im) for im in ims]

class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC, antialias = None):
        self.size = size
        self.resize = transforms.Resize(size, mode, antialias = antialias)

    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)

class Normalize:
    def __call__(self,im):
        mean = im.mean(dim=(1,2), keepdims=True)
        std = im.std(dim=(1,2), keepdims=True)
        return (im-mean)/std


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        c,h,w = im_tuple[0].shape
        if c > 3:
            warnings.warn(f"Number of channels {c=} > 3, assuming first 3 are rgb")
        return [self.normalize(im[:3]) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
        
        
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]
    
    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, unnormalize=False, autoscale = False):
    if unnormalize:
        x = x * (imagenet_std[:, None, None].to(x.device)) + (imagenet_mean[:, None, None].to(x.device))
    if autoscale:
        if x.max() == x.min():
            warnings.warn("x max == x min, cant autoscale")
        else:
            x = (x-x.min())/(x.max()-x.min())
        
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_best_device(batch, device=get_best_device()):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def to_cpu(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()
    return batch


def get_pose(calib):
    w, h = np.array(calib["imsize"])[0]
    return np.array(calib["K"]), np.array(calib["R"]), np.array(calib["T"]).T, h, w


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans

def to_pixel_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return flow

def to_normalized_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                2 * (flow[..., 0]) / w1 - 1,
                2 * (flow[..., 1]) / h1 - 1,
            ),
            axis=-1,
        )
    )
    return flow


def warp_to_pixel_coords(warp, h1, w1, h2, w2):
    warp1 = warp[..., :2]
    warp1 = (
        torch.stack(
            (
                w1 * (warp1[..., 0] + 1) / 2,
                h1 * (warp1[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    warp2 = warp[..., 2:]
    warp2 = (
        torch.stack(
            (
                w2 * (warp2[..., 0] + 1) / 2,
                h2 * (warp2[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return torch.cat((warp1,warp2), dim=-1)


def to_homogeneous(x):
    ones = torch.ones_like(x[...,-1:])
    return torch.cat((x, ones), dim = -1)

def from_homogeneous(xh, eps = 1e-12):
    return xh[...,:-1] / (xh[...,-1:]+eps)

def homog_transform(Homog, x):
    xh = to_homogeneous(x)
    yh = (Homog @ xh.mT).mT
    y = from_homogeneous(yh)
    return y

def get_homog_warp(Homog, H, W, device = get_best_device()):
    grid = torch.meshgrid(torch.linspace(-1+1/H,1-1/H,H, device = device), torch.linspace(-1+1/W,1-1/W,W, device = device))
    
    x_A = torch.stack((grid[1], grid[0]), dim = -1)[None]
    x_A_to_B = homog_transform(Homog, x_A)
    mask = ((x_A_to_B > -1) * (x_A_to_B < 1)).prod(dim=-1).float()
    return torch.cat((x_A.expand(*x_A_to_B.shape), x_A_to_B),dim=-1), mask

def dual_log_softmax_matcher(desc_A: tuple['B','N','C'], desc_B: tuple['B','M','C'], inv_temperature = 1, normalize = False):
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    logP = corr.log_softmax(dim = -2) + corr.log_softmax(dim= -1)
    return logP

def dual_softmax_matcher(desc_A: tuple['B','N','C'], desc_B: tuple['B','M','C'], inv_temperature = 1, normalize = False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim = -2) * corr.softmax(dim= -1)
    return P

def conditional_softmax_matcher(desc_A: tuple['B','N','C'], desc_B: tuple['B','M','C'], inv_temperature = 1, normalize = False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P_B_cond_A = corr.softmax(dim = -1)
    P_A_cond_B = corr.softmax(dim = -2)
    
    return P_A_cond_B, P_B_cond_A 