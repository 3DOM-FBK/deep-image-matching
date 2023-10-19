import torch
from torch import nn
import torch.nn.functional as F


# coordinates system
#  ------------------------------>  [ x: range=-1.0~1.0; w: range=0~W ]
#  | -----------------------------
#  | |                           |
#  | |                           |
#  | |                           |
#  | |         image             |
#  | |                           |
#  | |                           |
#  | |                           |
#  | |---------------------------|
#  v
# [ y: range=-1.0~1.0; h: range=0~H ]

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def sample_descriptor(descriptor_map, kpts, bilinear_interp=False):
    """
    :param descriptor_map: BxCxHxW
    :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
    :param bilinear_interp: bool, whether to use bilinear interpolation
    :return: descriptors: list, len=B, each is NxD
    """
    batch_size, channel, height, width = descriptor_map.shape

    descriptors = []
    for index in range(batch_size):
        kptsi = kpts[index]  # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = torch.nn.functional.grid_sample(descriptor_map[index].unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            kptsi = (kptsi + 1) / 2 * kptsi.new_tensor([[width - 1, height - 1]])
            kptsi = kptsi.long()
            descriptors_ = descriptor_map[index, :, kptsi[:, 1], kptsi[:, 0]]  # CxN

        descriptors_ = torch.nn.functional.normalize(descriptors_, p=2, dim=0)
        descriptors.append(descriptors_.t())

    return descriptors


class DKD(nn.Module):
    def __init__(self, radius=2, top_k=0, scores_th=0.2, n_limit=20000):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
                                                   else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1  # tuned temperature
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)

        # local xy grid
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        # (kernel_size*kernel_size) x 2 : (w,h)
        self.hw_grid = torch.stack(torch.meshgrid([x, x])).view(2, -1).t()[:, [1, 0]]

    def detect_keypoints(self, scores_map, sub_pixel=True):
        b, c, h, w = scores_map.shape
        scores_nograd = scores_map.detach()
        # nms_scores = simple_nms(scores_nograd, self.radius)
        nms_scores = simple_nms(scores_nograd, 2)

        # remove border
        nms_scores[:, :, :self.radius + 1, :] = 0
        nms_scores[:, :, :, :self.radius + 1] = 0
        nms_scores[:, :, h - self.radius:, :] = 0
        nms_scores[:, :, :, w - self.radius:] = 0

        # detect keypoints without grad
        if self.top_k > 0:
            topk = torch.topk(nms_scores.view(b, -1), self.top_k)
            indices_keypoints = topk.indices  # B x top_k
        else:
            if self.scores_th > 0:
                masks = nms_scores > self.scores_th
                if masks.sum() == 0:
                    th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                    masks = nms_scores > th.reshape(b, 1, 1, 1)
            else:
                th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                masks = nms_scores > th.reshape(b, 1, 1, 1)
            masks = masks.reshape(b, -1)

            indices_keypoints = []  # list, B x (any size)
            scores_view = scores_nograd.reshape(b, -1)
            for mask, scores in zip(masks, scores_view):
                indices = mask.nonzero(as_tuple=False)[:, 0]
                if len(indices) > self.n_limit:
                    kpts_sc = scores[indices]
                    sort_idx = kpts_sc.sort(descending=True)[1]
                    sel_idx = sort_idx[:self.n_limit]
                    indices = indices[sel_idx]
                indices_keypoints.append(indices)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        if sub_pixel:
            # detect soft keypoints with grad backpropagation
            patches = self.unfold(scores_map)  # B x (kernel**2) x (H*W)
            self.hw_grid = self.hw_grid.to(patches)  # to device
            for b_idx in range(b):
                patch = patches[b_idx].t()  # (H*W) x (kernel**2)
                indices_kpt = indices_keypoints[b_idx]  # one dimension vector, say its size is M
                patch_scores = patch[indices_kpt]  # M x (kernel**2)

                # max is detached to prevent undesired backprop loops in the graph
                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                x_exp = ((patch_scores - max_v) / self.temperature).exp()  # M * (kernel**2), in [0, 1]

                # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
                xy_residual = x_exp @ self.hw_grid / x_exp.sum(dim=1)[:, None]  # Soft-argmax, Mx2

                hw_grid_dist2 = torch.norm((self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius,
                                           dim=-1) ** 2
                scoredispersity = (x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)

                # compute result keypoints
                keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms + xy_residual
                keypoints_xy = keypoints_xy / keypoints_xy.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)

                kptscore = torch.nn.functional.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)
        else:
            for b_idx in range(b):
                indices_kpt = indices_keypoints[b_idx]  # one dimension vector, say its size is M
                keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms / keypoints_xy_nms.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                kptscore = torch.nn.functional.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN
                keypoints.append(keypoints_xy)
                scoredispersitys.append(None)
                kptscores.append(kptscore)

        return keypoints, scoredispersitys, kptscores

    def forward(self, scores_map, descriptor_map, sub_pixel=False):
        """
        :param scores_map:  Bx1xHxW
        :param descriptor_map: BxCxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """
        keypoints, scoredispersitys, kptscores = self.detect_keypoints(scores_map,
                                                                       sub_pixel)

        descriptors = sample_descriptor(descriptor_map, keypoints, sub_pixel)

        # keypoints: B M 2
        # descriptors: B M D
        # scoredispersitys:
        return keypoints, descriptors, kptscores, scoredispersitys
