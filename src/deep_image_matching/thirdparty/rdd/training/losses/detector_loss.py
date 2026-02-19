import torch
import torch.nn.functional as F

from torch import nn
import cv2
import numpy as np
from copy import deepcopy
from RDD.dataset.megadepth.utils import warp

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

class DetectorLoss(nn.Module):
    def __init__(self, temperature = 0.1, scores_th = 0.1, peaky_weight = 0.5, reprojection_weight = 1, scoremap_weight = 0.5):
        super().__init__()
        self.temperature = temperature
        self.scores_th = scores_th
        self.peaky_weight = peaky_weight
        self.reprojection_weight = reprojection_weight
        self.scoremap_weight = scoremap_weight
        
        self.PeakyLoss = PeakyLoss(scores_th = scores_th)
        self.ReprojectionLocLoss = ReprojectionLocLoss(scores_th = scores_th)
        self.ScoreMapRepLoss = ScoreMapRepLoss(temperature = temperature)
        
    def forward(self, correspondences, pred0_with_rand, pred1_with_rand):
        loss_peaky0 = self.PeakyLoss(pred0_with_rand)
        loss_peaky1 = self.PeakyLoss(pred1_with_rand)
        loss_peaky = (loss_peaky0 + loss_peaky1) / 2.
        
        loss_reprojection = self.ReprojectionLocLoss(pred0_with_rand, pred1_with_rand, correspondences)
        
        loss_score_map_rp = self.ScoreMapRepLoss(pred0_with_rand, pred1_with_rand, correspondences)
        
        loss_kp = loss_peaky * self.peaky_weight + loss_reprojection * self.reprojection_weight + loss_score_map_rp * self.scoremap_weight
        
        return loss_kp

class PeakyLoss(object):
    """ PeakyLoss to avoid an uniform score map """

    def __init__(self, scores_th: float = 0.1):
        super().__init__()
        self.scores_th = scores_th

    def __call__(self, pred):
        b, c, h, w = pred['scores_map'].shape
        loss_mean = 0
        CNT = 0
        
        for idx in range(b):
            n_original = len(pred['score_dispersity'][idx])
            scores_kpts = pred['scores'][idx][:n_original]
            valid = scores_kpts > self.scores_th
            loss_peaky = pred['score_dispersity'][idx][valid]

            loss_mean = loss_mean + loss_peaky.sum()
            CNT = CNT + len(loss_peaky)

        loss_mean = loss_mean / CNT if CNT != 0 else pred['scores_map'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class ReprojectionLocLoss(object):
    """
    Reprojection location errors of keypoints to train repeatable detector.
    """

    def __init__(self, norm: int = 1, scores_th: float = 0.1):
        super().__init__()
        self.norm = norm
        self.scores_th = scores_th

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        loss_mean = 0
        CNT = 0
        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            if self.norm == 2:
                dist = correspondences[idx]['dist']
            elif self.norm == 1:
                dist = correspondences[idx]['dist_l1']
            else:
                raise TypeError('No such norm in correspondence.')

            ids0_d = correspondences[idx]['ids0_d']
            ids1_d = correspondences[idx]['ids1_d']

            scores0 = correspondences[idx]['scores0'].detach()[ids0_d]
            scores1 = correspondences[idx]['scores1'].detach()[ids1_d]
            valid = (scores0 > self.scores_th) * (scores1 > self.scores_th)
            reprojection_errors = dist[ids0_d, ids1_d][valid]

            loss_mean = loss_mean + reprojection_errors.sum()
            CNT = CNT + len(reprojection_errors)

        loss_mean = loss_mean / CNT if CNT != 0 else correspondences[0]['dist'].new_tensor(0)

        assert not torch.isnan(loss_mean)
        return loss_mean


def local_similarity(descriptor_map, descriptors, kpts_wh, radius):
    """
    :param descriptor_map: CxHxW
    :param descriptors: NxC
    :param kpts_wh: Nx2 (W,H)
    :return:
    """
    _, h, w = descriptor_map.shape
    ksize = 2 * radius + 1

    descriptor_map_unflod = torch.nn.functional.unfold(descriptor_map.unsqueeze(0),
                                                       kernel_size=(ksize, ksize),
                                                       padding=(radius, radius))
    descriptor_map_unflod = descriptor_map_unflod[0].t().reshape(h * w, -1, ksize * ksize)
    # find the correspondence patch
    kpts_wh_long = kpts_wh.detach().long()
    patch_ids = kpts_wh_long[:, 0] + kpts_wh_long[:, 1] * h
    desc_patches = descriptor_map_unflod[patch_ids].permute(0, 2, 1).detach()  # N_kpts x s*s x 128

    local_sim = torch.einsum('nsd,nd->ns', desc_patches, descriptors)
    local_sim_sort = torch.sort(local_sim, dim=1, descending=True).values
    local_sim_sort_mean = local_sim_sort[:, 4:].mean(dim=1)  # 4 is safe radius for bilinear interplation

    return local_sim_sort_mean


class ScoreMapRepLoss(object):
    """ Scoremap repetability"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.radius = 2

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]
            kpts01 = correspondences[idx]['kpts01']
            kpts10 = correspondences[idx]['kpts10']  # valid warped keypoints

            # =====================
            scores_kpts10 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            scores_kpts01 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]

            s0 = scores_kpts01 * correspondences[idx]['scores0']  # repeatability
            s1 = scores_kpts10 * correspondences[idx]['scores1']  # repeatability

            # ===================== repetability
            similarity_map_01 = correspondences[idx]['similarity_map_01_valid']
            similarity_map_10 = correspondences[idx]['similarity_map_10_valid']

            pmf01 = ((similarity_map_01.detach() - 1) / self.temperature).exp()
            pmf10 = ((similarity_map_10.detach() - 1) / self.temperature).exp()
            kpts01 = kpts01.detach()
            kpts10 = kpts10.detach()

            pmf01_kpts = torch.nn.functional.grid_sample(pmf01.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts = torch.nn.functional.grid_sample(pmf10.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            repetability01 = torch.diag(pmf01_kpts)
            repetability10 = torch.diag(pmf10_kpts)
            
            # ===================== reliability
            # ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            # descriptor_map0 = pred0['descriptor_map'][idx].detach()
            # descriptor_map1 = pred1['descriptor_map'][idx].detach()
            # descriptors0 = pred0['descriptors'][idx][ids0].detach()
            # descriptors1 = pred1['descriptors'][idx][ids1].detach()
            # kpts0 = pred0['keypoints'][idx][ids0].detach()
            # kpts1 = pred1['keypoints'][idx][ids1].detach()
            # kpts0_wh = (kpts0 / 2 + 0.5) * wh
            # kpts1_wh = (kpts1 / 2 + 0.5) * wh
            # ls0 = local_similarity(descriptor_map0, descriptors0, kpts0_wh, self.radius)
            # ls1 = local_similarity(descriptor_map1, descriptors1, kpts1_wh, self.radius)
            # reliability0 = 1 - ((ls0 - 1) / self.temperature).exp()
            # reliability1 = 1 - ((ls1 - 1) / self.temperature).exp()

            fs0 = repetability01  # * reliability0
            fs1 = repetability10  # * reliability1

            if s0.sum() != 0:
                loss01 = (1 - fs0) * s0 * len(s0) / s0.sum()
                loss_mean = loss_mean + loss01.sum()
                CNT = CNT + len(loss01)
            if s1.sum() != 0:
                loss10 = (1 - fs1) * s1 * len(s1) / s1.sum()
                loss_mean = loss_mean + loss10.sum()
                CNT = CNT + len(loss10)
        
        loss_mean = loss_mean / CNT if CNT != 0 else pred0['scores_map'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean
    
    

#+++++++++++++++++++++++++++++++++++++++++++++++++++Taken from ALIKE+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def compute_keypoints_distance(kpts0, kpts1, p=2, debug=False):
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

def compute_correspondence(model, pred0, pred1, batch, radius = 2, rand=True, train_gt_th = 5, debug=False):
    b, c, h, w = pred0['scores_map'].shape
    wh = pred0['scores_map'][0].new_tensor([[w - 1, h - 1]])
    
    pred0_with_rand = pred0
    pred1_with_rand = pred1
    pred0_with_rand['scores'] = []
    pred1_with_rand['scores'] = []
    pred0_with_rand['descriptors'] = []
    pred1_with_rand['descriptors'] = []
    pred0_with_rand['num_det'] = []
    pred1_with_rand['num_det'] = []

    kps, score_dispersity, scores = model.softdetect.detect_keypoints(pred0['scores_map'])
    
    pred0_with_rand['keypoints'] = kps
    pred0_with_rand['score_dispersity'] = score_dispersity
    
    
    kps, score_dispersity, scores = model.softdetect.detect_keypoints(pred1['scores_map'])
    pred1_with_rand['keypoints'] = kps
    pred1_with_rand['score_dispersity'] = score_dispersity

    correspondences = []
    for idx in range(b):
        # =========================== prepare keypoints
        kpts0, kpts1 = pred0['keypoints'][idx], pred1['keypoints'][idx]  # (x,y), shape: Nx2
        
        # additional random keypoints
        if rand:
            rand0 = torch.rand(len(kpts0), 2, device=kpts0.device) * 2 - 1  # -1~1
            rand1 = torch.rand(len(kpts1), 2, device=kpts1.device) * 2 - 1  # -1~1
            kpts0 = torch.cat([kpts0, rand0])
            kpts1 = torch.cat([kpts1, rand1])

            pred0_with_rand['keypoints'][idx] = kpts0
            pred1_with_rand['keypoints'][idx] = kpts1
        
        scores_map0 = pred0['scores_map'][idx]
        scores_map1 = pred1['scores_map'][idx]
        scores_kpts0 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                        mode='bilinear', align_corners=True).squeeze()
        scores_kpts1 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                        mode='bilinear', align_corners=True).squeeze()

        kpts0_wh_ = (kpts0 / 2 + 0.5) * wh  # N0x2, (w,h)
        kpts1_wh_ = (kpts1 / 2 + 0.5) * wh  # N1x2, (w,h)

        # ========================= nms
        dist = compute_keypoints_distance(kpts0_wh_.detach(), kpts0_wh_.detach())
        local_mask = dist < radius
        valid_cnt = torch.sum(local_mask, dim=1)
        indices_need_nms = torch.where(valid_cnt > 1)[0]
        
        for i in indices_need_nms:
            if valid_cnt[i] > 0:
                kpt_indices = torch.where(local_mask[i])[0]
                scs_max_idx = scores_kpts0[kpt_indices].argmax()

                tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                tmp_mask[scs_max_idx] = False
                suppressed_indices = kpt_indices[tmp_mask]

                valid_cnt[suppressed_indices] = 0
        
        valid_mask = valid_cnt > 0
        kpts0_wh = kpts0_wh_[valid_mask]
        kpts0 = kpts0[valid_mask]
        scores_kpts0 = scores_kpts0[valid_mask]
        pred0_with_rand['keypoints'][idx] = kpts0

        valid_mask = valid_mask[:len(pred0_with_rand['score_dispersity'][idx])]
        pred0_with_rand['score_dispersity'][idx] = pred0_with_rand['score_dispersity'][idx][valid_mask]
        pred0_with_rand['num_det'].append(valid_mask.sum())

        dist = compute_keypoints_distance(kpts1_wh_.detach(), kpts1_wh_.detach())
        local_mask = dist < radius
        valid_cnt = torch.sum(local_mask, dim=1)
        indices_need_nms = torch.where(valid_cnt > 1)[0]
        for i in indices_need_nms:
            if valid_cnt[i] > 0:
                kpt_indices = torch.where(local_mask[i])[0]
                scs_max_idx = scores_kpts1[kpt_indices].argmax()

                tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                tmp_mask[scs_max_idx] = False
                suppressed_indices = kpt_indices[tmp_mask]

                valid_cnt[suppressed_indices] = 0
                
        valid_mask = valid_cnt > 0
        kpts1_wh = kpts1_wh_[valid_mask]
        kpts1 = kpts1[valid_mask]
        scores_kpts1 = scores_kpts1[valid_mask]
        pred1_with_rand['keypoints'][idx] = kpts1

        valid_mask = valid_mask[:len(pred1_with_rand['score_dispersity'][idx])]
        pred1_with_rand['score_dispersity'][idx] = pred1_with_rand['score_dispersity'][idx][valid_mask]
        pred1_with_rand['num_det'].append(valid_mask.sum())

        # del dist, local_mask, valid_cnt, indices_need_nms, scs_max_idx, tmp_mask, suppressed_indices, valid_mask
        # torch.cuda.empty_cache()
        # ========================= nms

        pred0_with_rand['scores'].append(scores_kpts0)
        pred1_with_rand['scores'].append(scores_kpts1)
        
        descriptor_map0, descriptor_map1 = pred0['descriptor_map'][idx], pred1['descriptor_map'][idx]
        descriptor_map0 = F.normalize(descriptor_map0, dim=0)
        descriptor_map1 = F.normalize(descriptor_map1, dim=0)

        desc0 = torch.nn.functional.grid_sample(descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                mode='bilinear', align_corners=True)[0, :, 0, :].t()
        desc1 = torch.nn.functional.grid_sample(descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                mode='bilinear', align_corners=True)[0, :, 0, :].t()
        desc0 = F.normalize(desc0, dim=-1)
        desc1 = F.normalize(desc1, dim=-1)

        pred0_with_rand['descriptors'].append(desc0)
        pred1_with_rand['descriptors'].append(desc1)
        
        # =========================== prepare warp parameters
        warp01_params = {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[idx]
        warp10_params = {}
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[idx]

        # =========================== warp keypoints across images
        try:
            kpts0_wh, kpts01_wh, ids0, ids0_out = warp(kpts0_wh, warp01_params)
            kpts1_wh, kpts10_wh, ids1, ids1_out = warp(kpts1_wh, warp10_params)
        except:
            correspondences.append({'correspondence0': None, 'correspondence1': None,
                                    'dist': kpts0_wh.new_tensor(0),
                                    })
            continue

        if debug:
            from training.utils import save_image_in_actual_size
            
            image0 = batch['image0'][idx].cpu().detach().numpy().transpose(1, 2, 0)
            image1 = batch['image1'][idx].cpu().detach().numpy().transpose(1, 2, 0)
            
            p0 = kpts0_wh[:, [1, 0]].cpu().detach().numpy()
            img_kpts0 = plot_keypoints(image0, p0, radius=5, color=(255, 0, 0))
            # display_image_in_actual_size(img_kpts0)

            p1 = kpts1_wh[:, [1, 0]].cpu().detach().numpy()
            img_kpts1 = plot_keypoints(image1, p1, radius=5, color=(255, 0, 0))
            # display_image_in_actual_size(img_kpts1)

            p01 = kpts01_wh[:, [1, 0]].cpu().detach().numpy()
            img_kpts01 = plot_keypoints(img_kpts1, p01, radius=5, color=(0, 255, 0))
            save_image_in_actual_size(img_kpts01, name='kpts01.png')
            
            p10 = kpts10_wh[:, [1, 0]].cpu().detach().numpy()
            img_kpts10 = plot_keypoints(img_kpts0, p10, radius=5, color=(0, 255, 0))
            save_image_in_actual_size(img_kpts10, name='kpts10.png')

        # ============================= compute reprojection error
        dist01 = compute_keypoints_distance(kpts0_wh, kpts10_wh)
        dist10 = compute_keypoints_distance(kpts1_wh, kpts01_wh)

        dist_l2 = (dist01 + dist10.t()) / 2.
        # find mutual correspondences by calculating the distance
        # between keypoints (I1) and warpped keypoints (I2->I1)
        mutual_min_indices = mutual_argmin(dist_l2)

        dist_mutual_min = dist_l2[mutual_min_indices]
        valid_dist_mutual_min = dist_mutual_min.detach() < train_gt_th

        ids0_d = mutual_min_indices[0][valid_dist_mutual_min]
        ids1_d = mutual_min_indices[1][valid_dist_mutual_min]

        correspondence0 = ids0[ids0_d]
        correspondence1 = ids1[ids1_d]

        # L1 distance
        dist01_l1 = compute_keypoints_distance(kpts0_wh, kpts10_wh, p=1)
        dist10_l1 = compute_keypoints_distance(kpts1_wh, kpts01_wh, p=1)

        dist_l1 = (dist01_l1 + dist10_l1.t()) / 2.

        # =========================== compute cross image descriptor similarity_map
        similarity_map_01 = (desc0 @ descriptor_map1.reshape(h*w, 256).t()) * 20
        similarity_map_01 = similarity_map_01.softmax(dim = -2) * similarity_map_01.softmax(dim= -1)
        similarity_map_01 = similarity_map_01.reshape(desc0.shape[0], h, w)
        similarity_map_01 = torch.clamp(similarity_map_01, 1e-6, 1-1e-6)
        
        similarity_map_10 = (desc1 @ descriptor_map0.reshape(h*w, 256).t()) * 20
        similarity_map_10 = similarity_map_10.softmax(dim = -2) * similarity_map_10.softmax(dim= -1)
        similarity_map_10 = similarity_map_10.reshape(desc1.shape[0], h, w)
        similarity_map_10 = torch.clamp(similarity_map_10, 1e-6, 1-1e-6)

        similarity_map_01_valid = similarity_map_01[ids0]  # valid descriptors
        similarity_map_10_valid = similarity_map_10[ids1]

        kpts01 = 2 * kpts01_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
        kpts10 = 2 * kpts10_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]

        correspondences.append({'correspondence0': correspondence0,  # indices of matched kpts0 in all kpts
                                'correspondence1': correspondence1,  # indices of matched kpts1 in all kpts
                                'scores0': scores_kpts0[ids0],
                                'scores1': scores_kpts1[ids1],
                                'kpts01': kpts01, 'kpts10': kpts10,  # warped valid kpts
                                'ids0': ids0, 'ids1': ids1,  # valid indices of kpts0 and kpts1
                                'ids0_out': ids0_out, 'ids1_out': ids1_out,
                                'ids0_d': ids0_d, 'ids1_d': ids1_d,  # match indices of valid kpts0 and kpts1
                                'dist_l1': dist_l1,  # cross distance matrix of valid kpts using L1 norm
                                'dist': dist_l2,  # cross distance matrix of valid kpts using L2 norm
                                'similarity_map_01': similarity_map_01,  # all
                                'similarity_map_10': similarity_map_10,  # all
                                'similarity_map_01_valid': similarity_map_01_valid,  # valid
                                'similarity_map_10_valid': similarity_map_10_valid,  # valid
                                })

    return correspondences, pred0_with_rand, pred1_with_rand


class EmptyTensorError(Exception):
    pass