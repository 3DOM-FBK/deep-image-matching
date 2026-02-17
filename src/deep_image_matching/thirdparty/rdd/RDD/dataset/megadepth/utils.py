"""
	Modified from 
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    MegaDepth data handling was adapted from 
    LoFTR official code: https://github.com/zju3dv/LoFTR/blob/master/src/datasets/megadepth.py
"""

import io
import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv
from kornia.geometry.epipolar import essential_from_Rt
from kornia.geometry.epipolar import fundamental_from_essential

import cv2
import torch
import numpy as np
from numba import jit
from copy import deepcopy

try:
    from utils.project_depth_nn_cython_pkg import project_depth_nn_cython

    nn_cython = True
except:
    print('\033[1;41;37mWarning: using python to project depth!!!\033[0m')

    nn_cython = False


class EmptyTensorError(Exception):
    pass


def mutual_NN(cross_matrix, mode: str = 'min'):
    """
    compute mutual nearest neighbor from a cross_matrix, non-differentiable function
    :param cross_matrix: N0xN1
    :param mode: 'min': mutual minimum; 'max':mutual maximum
    :return: index0,index1, Mx2
    """
    if mode == 'min':
        nn0 = cross_matrix == cross_matrix.min(dim=1, keepdim=True)[0]
        nn1 = cross_matrix == cross_matrix.min(dim=0, keepdim=True)[0]
    elif mode == 'max':
        nn0 = cross_matrix == cross_matrix.max(dim=1, keepdim=True)[0]
        nn1 = cross_matrix == cross_matrix.max(dim=0, keepdim=True)[0]
    else:
        raise TypeError("error mode, must be 'min' or 'max'.")

    mutual_nn = nn0 * nn1

    return torch.nonzero(mutual_nn, as_tuple=False)


def mutual_argmax(value, mask=None, as_tuple=True):
    """
    Args:
        value: MxN
        mask:  MxN

    Returns:

    """
    value = value - value.min()  # convert to non-negative tensor
    if mask is not None:
        value = value * mask

    max0 = value.max(dim=1, keepdim=True)  # the col index the max value in each row
    max1 = value.max(dim=0, keepdim=True)

    valid_max0 = value == max0[0]
    valid_max1 = value == max1[0]

    mutual = valid_max0 * valid_max1
    if mask is not None:
        mutual = mutual * mask

    return mutual.nonzero(as_tuple=as_tuple)


def mutual_argmin(value, mask=None):
    return mutual_argmax(-value, mask)


def compute_keypoints_distance(kpts0, kpts1, p=2):
    """
    Args:
        kpts0: torch.tensor [M,2]
        kpts1: torch.tensor [N,2]
        p: (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm

    Returns:
        dist, torch.tensor [N,M]
    """
    dist = kpts0[:, None, :] - kpts1[None, :, :]  # [M,N,2]
    dist = torch.norm(dist, p=p, dim=2)  # [M,N]
    return dist


def keypoints_normal2pixel(kpts_normal, w, h):
    wh = kpts_normal[0].new_tensor([[w - 1, h - 1]])
    kpts_pixel = [(kpts + 1) / 2 * wh for kpts in kpts_normal]
    return kpts_pixel


def plot_keypoints(image, kpts, radius=2, color=(255, 0, 0)):
    image = image.cpu().detach().numpy() if isinstance(image, torch.Tensor) else image
    kpts = kpts.cpu().detach().numpy() if isinstance(kpts, torch.Tensor) else kpts

    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        y0, x0 = kpt
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, radius)

        # cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)
    return out


def plot_matches(image0, image1, kpts0, kpts1, radius=2, color=(255, 0, 0), mcolor=(0, 255, 0), layout='lr'):
    image0 = image0.cpu().detach().numpy() if isinstance(image0, torch.Tensor) else image0
    image1 = image1.cpu().detach().numpy() if isinstance(image1, torch.Tensor) else image1
    kpts0 = kpts0.cpu().detach().numpy() if isinstance(kpts0, torch.Tensor) else kpts0
    kpts1 = kpts1.cpu().detach().numpy() if isinstance(kpts1, torch.Tensor) else kpts1

    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = out0
        out[:H1, W0:, :] = out1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = out0
        out[H0:, :W1, :] = out1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0 = np.round(kpts0).astype(int)
    kpts1 = np.round(kpts1).astype(int)

    for kpt0, kpt1 in zip(kpts0, kpts1):
        (y0, x0), (y1, x1) = kpt0, kpt1

        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=mcolor, thickness=1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=mcolor, thickness=1, lineType=cv2.LINE_AA)

    return out


def interpolate_depth(pos, depth):
    pos = pos.t()[[1, 0]]  # Nx2 -> 2xN; w,h -> h,w(i,j)

    # =============================================== from d2-net
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :].detach()  # TODO: changed here
    j = pos[1, :].detach()  # TODO: changed here

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(torch.min(valid_top_left, valid_top_right),
                              torch.min(valid_bottom_left, valid_bottom_right))

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    ids_valid_corners = deepcopy(ids)
    if ids.size(0) == 0:
        # raise ValueError('empty tensor: ids')
        raise EmptyTensorError

    # Valid depth
    valid_depth = torch.min(torch.min(depth[i_top_left, j_top_left] > 0,
                                      depth[i_top_right, j_top_right] > 0),
                            torch.min(depth[i_bottom_left, j_bottom_left] > 0,
                                      depth[i_bottom_right, j_bottom_right] > 0))

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    ids_valid_depth = deepcopy(ids)
    if ids.size(0) == 0:
        # raise ValueError('empty tensor: ids')
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (w_top_left * depth[i_top_left, j_top_left] +
                          w_top_right * depth[i_top_right, j_top_right] +
                          w_bottom_left * depth[i_bottom_left, j_bottom_left] +
                          w_bottom_right * depth[i_bottom_right, j_bottom_right])

    # pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    pos = pos[:, ids]

    # =============================================== from d2-net
    pos = pos[[1, 0]].t()  # 2xN -> Nx2;  h,w(i,j) -> w,h

    # interpolated_depth: valid interpolated depth
    # pos: valid position (keypoint)
    # ids: indices of valid position (keypoint)

    return [interpolated_depth, pos, ids, ids_valid_corners, ids_valid_depth]


def to_homogeneous(kpts):
    '''
    :param kpts: Nx2
    :return: Nx3
    '''
    ones = kpts.new_ones([kpts.shape[0], 1])
    return torch.cat((kpts, ones), dim=1)


def warp_homography(kpts0, params):
    '''
    :param kpts: Nx2
    :param homography_matrix: 3x3
    :return:
    '''
    homography_matrix = params['homography_matrix']
    w, h = params['width'], params['height']
    kpts0_homogeneous = to_homogeneous(kpts0)
    kpts01_homogeneous = torch.einsum('ij,kj->ki', homography_matrix, kpts0_homogeneous)
    kpts01 = kpts01_homogeneous[:, :2] / kpts01_homogeneous[:, 2:]

    kpts01_ = kpts01.detach()
    # due to float coordinates, the upper boundary should be (w-1) and (h-1).
    # For example, if the image size is 480, then the coordinates should in [0~470].
    # 470.5 is not acceptable.
    valid01 = (kpts01_[:, 0] >= 0) * (kpts01_[:, 0] <= w - 1) * (kpts01_[:, 1] >= 0) * (kpts01_[:, 1] <= h - 1)
    kpts0_valid = kpts0[valid01]
    kpts01_valid = kpts01[valid01]
    ids = torch.nonzero(valid01, as_tuple=False)[:, 0]
    ids_out = torch.nonzero(~valid01, as_tuple=False)[:, 0]

    # kpts0_valid: valid keypoints0, the invalid and inconsistance keypoints are removed
    # kpts01_valid: the warped valid keypoints0
    # ids: the valid indices
    return kpts0_valid, kpts01_valid, ids, ids_out


def project(points3d, K):
    """
    project 3D points to image plane

    Args:
        points3d: [N,3]
        K: [3,3]

    Returns:
        uv, (u,v), [N,2]
    """
    if type(K) == torch.Tensor:
        zuv1 = torch.einsum('jk,nk->nj', K, points3d)  # z*(u,v,1) = K*points3d -> [N,3]
    elif type(K) == np.ndarray:
        zuv1 = np.einsum('jk,nk->nj', K, points3d)
    else:
        raise TypeError("Input type should be 'torch.tensor' or 'numpy.ndarray'")
    uv1 = zuv1 / zuv1[:, -1][:, None]  # (u,v,1) -> [N,3]
    uv = uv1[:, 0:2]  # (u,v) -> [N,2]
    return uv, zuv1[:, -1]


def unproject(uv, d, K):
    """
    unproject pixels uv to 3D points

    Args:
        uv: [N,2]
        d: depth, [N,1]
        K: [3,3]

    Returns:
        3D points, [N,3]
    """
    duv = uv * d  # (u,v) [N,2]
    if type(K) == torch.Tensor:
        duv1 = torch.cat([duv, d], dim=1)  # z*(u,v,1) [N,3]
        K_inv = torch.inverse(K)  # [3,3]
        points3d = torch.einsum('jk,nk->nj', K_inv, duv1)  # [N,3]
    elif type(K) == np.ndarray:
        duv1 = np.concatenate((duv, d), axis=1)  # z*(u,v,1) [N,3]
        K_inv = np.linalg.inv(K)  # [3,3]
        points3d = np.einsum('jk,nk->nj', K_inv, duv1)  # [N,3]
    else:
        raise TypeError("Input type should be 'torch.tensor' or 'numpy.ndarray'")
    return points3d


def warp_se3(kpts0, params):
    pose01 = params['pose01']  # relative motion
    bbox0 = params['bbox0']  # row, col
    bbox1 = params['bbox1']
    depth0 = params['depth0']
    depth1 = params['depth1']
    intrinsics0 = params['intrinsics0']
    intrinsics1 = params['intrinsics1']

    # kpts0_valid: valid kpts0
    # z0_valid: depth of valid kpts0
    # ids0: the indices of valid kpts0 ( valid corners and valid depth)
    # ids0_valid_corners: the valid indices of kpts0 in image ( 0<=x<w, 0<=y<h )
    # ids0_valid_depth: the valid indices of kpts0 with valid depth ( depth > 0 )
    z0_valid, kpts0_valid, ids0, ids0_valid_corners, ids0_valid_depth = interpolate_depth(kpts0, depth0)

    # COLMAP convention
    bkpts0_valid = kpts0_valid + bbox0[[1, 0]][None, :] + 0.5

    # unproject pixel coordinate to 3D points (camera coordinate system)
    bpoints3d0 = unproject(bkpts0_valid, z0_valid.unsqueeze(1), intrinsics0)  # [:,3]
    bpoints3d0_homo = to_homogeneous(bpoints3d0)  # [:,4]

    # warp 3D point (camera 0 coordinate system) to 3D point (camera 1 coordinate system)
    bpoints3d01_homo = torch.einsum('jk,nk->nj', pose01, bpoints3d0_homo)  # [:,4]
    bpoints3d01 = bpoints3d01_homo[:, 0:3]  # [:,3]

    # project 3D point (camera coordinate system) to pixel coordinate
    buv01, z01 = project(bpoints3d01, intrinsics1)  # uv: [:,2], (h,w); z1: [N]

    uv01 = buv01 - bbox1[None, [1, 0]] - .5

    # kpts01_valid: valid kpts01
    # z01_valid: depth of valid kpts01
    # ids01: the indices of valid kpts01 ( valid corners and valid depth)
    # ids01_valid_corners: the valid indices of kpts01 in image ( 0<=x<w, 0<=y<h )
    # ids01_valid_depth: the valid indices of kpts01 with valid depth ( depth > 0 )
    z01_interpolate, kpts01_valid, ids01, ids01_valid_corners, ids01_valid_depth = interpolate_depth(uv01, depth1)

    outimage_mask = torch.ones(ids0.shape[0], device=ids0.device).bool()
    outimage_mask[ids01_valid_corners] = 0
    ids01_invalid_corners = torch.arange(0, ids0.shape[0], device=ids0.device)[outimage_mask]
    ids_outside = ids0[ids01_invalid_corners]

    # ids_valid: matched kpts01 without occlusion
    ids_valid = ids0[ids01]
    kpts0_valid = kpts0_valid[ids01]
    z01_proj = z01[ids01]

    inlier_mask = torch.abs(z01_proj - z01_interpolate) < 0.05

    # indices of kpts01 with occlusion
    ids_occlude = ids_valid[~inlier_mask]

    ids_valid = ids_valid[inlier_mask]
    if ids_valid.size(0) == 0:
        # raise ValueError('empty tensor: ids')
        raise EmptyTensorError

    kpts01_valid = kpts01_valid[inlier_mask]
    kpts0_valid = kpts0_valid[inlier_mask]

    # indices of kpts01 which are no matches in image1 for sure,
    # other projected kpts01 are not sure because of no depth in image0 or imgae1
    ids_out = torch.cat([ids_outside, ids_occlude])

    # kpts0_valid: valid keypoints0, the invalid and inconsistance keypoints are removed
    # kpts01_valid: the warped valid keypoints0
    # ids: the valid indices
    return kpts0_valid, kpts01_valid, ids_valid, ids_out


def warp(kpts0, params: dict):
    mode = params['mode']
    if mode == 'homo':
        return warp_homography(kpts0, params)
    elif mode == 'se3':
        return warp_se3(kpts0, params)
    else:
        raise ValueError('unknown mode!')


def warp_xy(kpts0_xy, params: dict):
    w, h = params['width'], params['height']
    kpts0 = (kpts0_xy / 2 + 0.5) * kpts0_xy.new_tensor([[w - 1, h - 1]])
    kpts0, kpts01, ids = warp(kpts0, params)
    kpts01_xy = 2 * kpts01 / kpts01.new_tensor([[w - 1, h - 1]]) - 1
    kpts0_xy = 2 * kpts0 / kpts0.new_tensor([[w - 1, h - 1]]) - 1
    return kpts0_xy, kpts01_xy, ids


def scale_intrinsics(K, scales):
    scales = np.diag([1. / scales[0], 1. / scales[1], 1.])
    return np.dot(scales, K)


def warp_points3d(points3d0, pose01):
    points3d0_homo = np.concatenate((points3d0, np.ones(points3d0.shape[0])[:, np.newaxis]), axis=1)  # [:,4]

    points3d01_homo = np.einsum('jk,nk->nj', pose01, points3d0_homo)  # [N,4]
    points3d01 = points3d01_homo[:, 0:3]  # [N,3]

    return points3d01


def unproject_depth(depth, K):
    h, w = depth.shape

    wh_range = np.mgrid[0:w, 0:h].transpose(2, 1, 0)  # [H,W,2]

    uv = wh_range.reshape(-1, 2)
    d = depth.reshape(-1, 1)
    points3d = unproject(uv, d, K)

    valid = np.logical_and((d[:, 0] > 0), (points3d[:, 2] > 0))

    return points3d, valid


@jit(nopython=True)
def project_depth_nn_python(uv, z, depth):
    h, w = depth.shape
    # TODO: speed up the for loop
    for idx in range(len(uv)):
        uvi = uv[idx]
        x = int(round(uvi[0]))
        y = int(round(uvi[1]))

        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        if depth[y, x] == 0. or depth[y, x] > z[idx]:
            depth[y, x] = z[idx]
    return depth


def project_nn(uv, z, depth):
    """
    uv: pixel coordinates [N,2]
    z: projected depth (xyz -> z) [N]
    depth: output depth array: [h,w]
    """
    if nn_cython:
        return project_depth_nn_cython(uv.astype(np.float64),
                                       z.astype(np.float64),
                                       depth.astype(np.float64))
    else:
        return project_depth_nn_python(uv, z, depth)


def warp_depth(depth0, intrinsics0, intrinsics1, pose01, shape1):
    points3d0, valid0 = unproject_depth(depth0, intrinsics0)  # [:,3]
    points3d0 = points3d0[valid0]

    points3d01 = warp_points3d(points3d0, pose01)

    uv01, z01 = project(points3d01, intrinsics1)  # uv: [N,2], (h,w); z1: [N]

    depth01 = project_nn(uv01, z01, depth=np.zeros(shape=shape1))

    return depth01


def warp_points2d(uv0, d0, intrinsics0, intrinsics1, pose01):
    points3d0 = unproject(uv0, d0, intrinsics0)
    points3d01 = warp_points3d(points3d0, pose01)
    uv01, z01 = project(points3d01, intrinsics1)
    return uv01, z01


def display_image_in_actual_size(image):
    import matplotlib.pyplot as plt

    dpi = 100
    height, width = image.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    if len(image.shape) == 3:
        ax.imshow(image, cmap='gray')
    elif len(image.shape) == 2:
        if image.dtype == np.uint8:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
            ax.text(20, 20, f"Range: {image.min():g}~{image.max():g}", color='red')
    plt.show()


# ====================================== copied from ASLFeat
from datetime import datetime


class ClassProperty(property):
    """For dynamically obtaining system time"""

    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Notify(object):
    """Colorful printing prefix.
    A quick example:
    print(Notify.INFO, YOUR TEXT, Notify.ENDC)
    """

    def __init__(self):
        pass

    @ClassProperty
    def HEADER(cls):
        return str(datetime.now()) + ': \033[95m'

    @ClassProperty
    def INFO(cls):
        return str(datetime.now()) + ': \033[92mI'

    @ClassProperty
    def OKBLUE(cls):
        return str(datetime.now()) + ': \033[94m'

    @ClassProperty
    def WARNING(cls):
        return str(datetime.now()) + ': \033[93mW'

    @ClassProperty
    def FAIL(cls):
        return str(datetime.now()) + ': \033[91mF'

    @ClassProperty
    def BOLD(cls):
        return str(datetime.now()) + ': \033[1mB'

    @ClassProperty
    def UNDERLINE(cls):
        return str(datetime.now()) + ': \033[4mU'

    ENDC = '\033[0m'

def get_essential(T0, T1):
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]
    
    t0 = T0[:3, 3].reshape(3, 1)
    t1 = T1[:3, 3].reshape(3, 1)
    
    R0 = torch.tensor(R0, dtype=torch.float32)
    R1 = torch.tensor(R1, dtype=torch.float32)
    t0 = torch.tensor(t0, dtype=torch.float32)
    t1 = torch.tensor(t1, dtype=torch.float32)
    
    E = essential_from_Rt(R0, t0, R1, t1)
    
    return E

def get_fundamental(E, K0, K1):
    F = fundamental_from_essential(E, K0, K1)
    
    return F
try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def load_array_from_s3(
    path, client, cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), 1)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def fix_path_from_d2net(path):
    if not path:
        return None

    path = path.replace('Undistorted_SfM/', '')
    path = path.replace('images', 'dense0/imgs')
    path = path.replace('phoenix/S6/zl548/MegaDepth_v1/', '')

    return path

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # resize image
    w, h = image.shape[1], image.shape[0]

    if len(resize) == 2:
        w_new, h_new = resize
    else:
        resize = resize[0]
        w_new, h_new = get_resized_wh(w, h, resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)


    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    #image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2,0,1) / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale

def imread_color(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_COLOR
    # if str(path).startswith('s3://'):
    #     image = load_array_from_s3(str(path), client, cv_type)
    # else:
    #     image = cv2.imread(str(path), cv_type)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if augment_fn is not None:
        image = augment_fn(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (3, h, w)


def read_megadepth_color(path,
                         resize=None,
                         df=None,
                         padding=False,
                         augment_fn=None,
                         rotation=0):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (3, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    image = imread_color(path, augment_fn, client=MEGADEPTH_CLIENT)
    
    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()

    # resize image
    if resize is not None:
        w, h = image.shape[1], image.shape[0]
        if len(resize) == 2:
            w_new, h_new = resize
        else:
            resize = resize[0]
            w_new, h_new = get_resized_wh(w, h, resize)
            w_new, h_new = get_divisible_wh(w_new, h_new, df)


        image = cv2.resize(image, (w_new, h_new))
        scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)
        scale_wh = torch.tensor([w_new, h_new], dtype=torch.float)
    else:
        scale = torch.tensor([1., 1.], dtype=torch.float)
        scale_wh = torch.tensor([image.shape[1], image.shape[0]], dtype=torch.float)
        
    image = image.transpose(2, 0, 1)
    
    if padding:  # padding
        if resize is not None:
            pad_to = max(h_new, w_new)
        else:
            pad_to = 2000
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale

def read_megadepth_depth(path, pad_to=None):

    if str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth

def get_image_name(path):
    return path.split('/')[-1]

def scale_intrinsics(K, scales):
    scales = np.diag([1. / scales[0], 1. / scales[1], 1.])
    return np.dot(scales, K)