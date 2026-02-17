from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..utils.utils import gridify

log = utils.get_pylogger(__name__)


class KeypointSampler(nn.Module):
    """
    Sample keypoints according to a Heatmap
    Adapted from: https://github.com/verlab/DALF_CVPR_2023/blob/main/modules/models/DALF.py
    """

    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.idx_cells = None  # Cache for meshgrid indices

    def sample(self, grid):
        """
        Sample keypoints given a grid where each cell has logits stacked in last dimension
        Input
          grid: [B, C, H//w, W//w, w*w]

        Returns
          log_probs: [B, C, H//w, W//w ] - logprobs of selected samples
          choices: [B, C, H//w, W//w] indices of choices
          accept_mask: [B, C, H//w, W//w] mask of accepted keypoints

        """
        chooser = torch.distributions.Categorical(logits=grid)
        choices = chooser.sample()
        logits_selected = torch.gather(grid, -1, choices.unsqueeze(-1)).squeeze(-1)

        flipper = torch.distributions.Bernoulli(logits=logits_selected)
        accepted_choices = flipper.sample()

        # Sum log-probabilities is equivalent to multiplying the probabilities
        log_probs = chooser.log_prob(choices) + flipper.log_prob(accepted_choices)

        accept_mask = accepted_choices.gt(0)

        return (
            log_probs.squeeze(1),
            choices,
            accept_mask.squeeze(1),
            logits_selected.squeeze(1),
        )

    def precompute_idx_cells(self, H, W, device):
        idx_cells = gridify(
            torch.dstack(
                torch.meshgrid(
                    torch.arange(H, dtype=torch.float32, device=device),
                    torch.arange(W, dtype=torch.float32, device=device),
                )
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(1, -1, -1, -1),
            window_size=self.window_size,
        )

        return idx_cells

    def forward(self, x, mask_padding=None):
        """
        Sample keypoints from a heatmap
        Input
          x: [B, C, H, W] Heatmap
          mask_padding: [B, 1, H, W] Mask for padding (optional)
        Returns
            keypoints: [B, H//w, W//w, 2] Keypoints in (x, y) format
            log_probs: [B, H//w, W//w] Log probabilities of selected keypoints
            mask: [B, H//w, W//w] Mask of accepted keypoints
            mask_padding: [B, 1, H//w, W//w] Mask of padding (optional)
            logits_selected: [B, H//w, W//w] Logits of selected keypoints
        """

        B, C, H, W = x.shape

        keypoint_cells = gridify(x, self.window_size)

        mask_padding = (
            (torch.min(gridify(mask_padding, self.window_size), dim=4).values) if mask_padding is not None else None
        )

        if self.idx_cells is None or self.idx_cells.shape[2:4] != (
            H // self.window_size,
            W // self.window_size,
        ):
            self.idx_cells = self.precompute_idx_cells(H, W, x.device)

        log_probs, idx, mask, logits_selected = self.sample(keypoint_cells)

        keypoints = (
            torch.gather(
                self.idx_cells.expand(B, -1, -1, -1, -1),
                -1,
                idx.repeat(1, 2, 1, 1).unsqueeze(-1),
            )
            .squeeze(-1)
            .permute(0, 2, 3, 1)
        )

        # flip keypoints to (x, y) format
        return keypoints.flip(-1), log_probs, mask, mask_padding, logits_selected


class RIPE(nn.Module):
    """
    Base class for extracting keypoints and descriptors
    Input
      x: [B, C, H, W] Images

    Returns
      kpts:
        list of size [B] with detected keypoints
      descs:
        list of size [B] with descriptors
    """

    def __init__(
        self,
        net,
        upsampler,
        window_size: int = 8,
        non_linearity_dect=None,
        desc_shares: Optional[List[int]] = None,
        descriptor_dim: int = 256,
        device=None,
    ):
        super().__init__()
        self.net = net

        self.detector = KeypointSampler(window_size)
        self.upsampler = upsampler
        self.sampler = None
        self.window_size = window_size
        self.non_linearity_dect = non_linearity_dect if non_linearity_dect is not None else nn.Identity()

        log.info(f"Training with window size {window_size}.")
        log.info(f"Use {non_linearity_dect} as final non-linearity before the detection heatmap.")

        dim_coarse_desc = self.get_dim_raw_desc()

        if desc_shares is not None:
            assert upsampler.name == "HyperColumnFeatures", (
                "Individual descriptor convolutions are only supported with HyperColumnFeatures"
            )
            assert len(desc_shares) == 4, "desc_shares should have 4 elements"
            assert sum(desc_shares) == descriptor_dim, f"sum of desc_shares should be {descriptor_dim}"

            self.conv_dim_reduction_coarse_desc = nn.ModuleList()

            for dim_in, dim_out in zip(dim_coarse_desc, desc_shares):
                log.info(f"Training dim reduction descriptor with {dim_in} -> {dim_out} 1x1 conv")
                self.conv_dim_reduction_coarse_desc.append(
                    nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
                )
        else:
            if descriptor_dim is not None:
                log.info(f"Training dim reduction descriptor with {sum(dim_coarse_desc)} -> {descriptor_dim} 1x1 conv")
                self.conv_dim_reduction_coarse_desc = nn.Conv1d(
                    sum(dim_coarse_desc),
                    descriptor_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            else:
                log.warning(
                    f"No descriptor dimension specified, no 1x1 conv will be applied! Direct usage of {sum(dim_coarse_desc)}-dimensional raw descriptor"
                )
                self.conv_dim_reduction_coarse_desc = nn.Identity()

    def get_dim_raw_desc(self):
        layers_dims_encoder = self.net.get_dim_layers_encoder()

        if self.upsampler.name == "InterpolateSparse2d":
            return [layers_dims_encoder[-1]]
        elif self.upsampler.name == "HyperColumnFeatures":
            return layers_dims_encoder
        else:
            raise ValueError(f"Unknown interpolator {self.upsampler.name}")

    @torch.inference_mode()
    def detectAndCompute(self, img, threshold=0.5, top_k=2048, output_aux=False):
        self.train(False)

        if img.dim() == 3:
            img = img.unsqueeze(0)

        out = self(img, training=False)
        B, K, H, W = out["heatmap"].shape

        assert B == 1, "Batch size should be 1"

        kpts = [{"xy": self.NMS(out["heatmap"][b], threshold)} for b in range(B)]

        if top_k is not None:
            for b in range(B):
                scores = out["heatmap"][b].squeeze(0)[kpts[b]["xy"][:, 1].long(), kpts[b]["xy"][:, 0].long()]
                sorted_idx = torch.argsort(-scores)
                kpts[b]["xy"] = kpts[b]["xy"][sorted_idx[:top_k]]
                if "logprobs" in kpts[b]:
                    kpts[b]["logprobs"] = kpts[b]["xy"][sorted_idx[:top_k]]

        if kpts[0]["xy"].shape[0] == 0:
            raise RuntimeError("No keypoints detected")

        # the following works for batch size 1 only

        descs = self.get_descs(out["coarse_descs"], img, kpts[0]["xy"].unsqueeze(0), H, W)
        descs = descs.squeeze(0)

        score_map = out["heatmap"][0].squeeze(0)

        kpts = kpts[0]["xy"]

        scores = score_map[kpts[:, 1], kpts[:, 0]]
        scores /= score_map.max()

        sort_idx = torch.argsort(-scores)
        kpts, descs, scores = kpts[sort_idx], descs[sort_idx], scores[sort_idx]

        if output_aux:
            return (
                kpts.float(),
                descs,
                scores,
                {
                    "heatmap": out["heatmap"],
                    "descs": out["coarse_descs"],
                    "conv": self.conv_dim_reduction_coarse_desc,
                },
            )

        return kpts.float(), descs, scores

    def NMS(self, x, threshold=3.0, kernel_size=3):
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)

        pos = (x == local_max) & (x > threshold)
        return pos.nonzero()[..., 1:].flip(-1)

    def get_descs(self, feature_map, guidance, kpts, H, W):
        descs = self.upsampler(feature_map, kpts, H, W)

        if isinstance(self.conv_dim_reduction_coarse_desc, nn.ModuleList):
            # individual descriptor convolutions for each layer
            desc_conv = []
            for desc, conv in zip(descs, self.conv_dim_reduction_coarse_desc):
                desc_conv.append(conv(desc.permute(0, 2, 1)).permute(0, 2, 1))
            desc = torch.cat(desc_conv, dim=-1)
        else:
            desc = torch.cat(descs, dim=-1)
            desc = self.conv_dim_reduction_coarse_desc(desc.permute(0, 2, 1)).permute(0, 2, 1)

        desc = F.normalize(desc, dim=2)

        return desc

    def forward(self, x, mask_padding=None, training=False):
        B, C, H, W = x.shape
        out = self.net(x)
        out["heatmap"] = self.non_linearity_dect(out["heatmap"])
        # print(out['map'].shape, out['descr'].shape)
        if training:
            kpts, log_probs, mask, mask_padding, logits_selected = self.detector(out["heatmap"], mask_padding)

            filter_A = kpts[:, :, :, 0] >= 16
            filter_B = kpts[:, :, :, 1] >= 16
            filter_C = kpts[:, :, :, 0] < W - 16
            filter_D = kpts[:, :, :, 1] < H - 16
            filter_all = filter_A * filter_B * filter_C * filter_D

            mask = mask * filter_all

            return (
                kpts.view(B, -1, 2),
                log_probs.view(B, -1),
                mask.view(B, -1),
                mask_padding.view(B, -1),
                logits_selected.view(B, -1),
                out,
            )
        else:
            return out


def output_number_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Number of trainable parameters: {nb_params:d}")
