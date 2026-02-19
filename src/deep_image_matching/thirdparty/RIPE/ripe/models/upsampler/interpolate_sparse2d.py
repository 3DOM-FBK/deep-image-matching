import torch
import torch.nn.functional as F
from torch import nn


class InterpolateSparse2d(nn.Module):
    """
    Interpolate 3D tensor given N sparse 2D positions
    Input
      x: list([C, H, W]) feature tensors at different scales (e.g. from a U-Net), ONLY the last one is used
      pos: [N, 2] tensor of positions
      H: int, height of the OUTPUT map
      W: int, width of the OUTPUT map

    Returns
      [N, C] sampled features at 2d positions
    """

    def __init__(self, mode="bicubic"):
        super().__init__()
        self.mode = mode
        self.name = "InterpolateSparse2d"

    def normgrid(self, x, H, W):
        return 2.0 * (x / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype))) - 1.0

    def forward(self, x, pos, H, W):
        x = x[-1]  # only use the last layer

        # check if grid is float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        grid = self.normgrid(pos, H, W).unsqueeze(-2)

        x = F.grid_sample(x, grid, mode=self.mode, align_corners=True)
        return [x.permute(0, 2, 3, 1).squeeze(-2)]
