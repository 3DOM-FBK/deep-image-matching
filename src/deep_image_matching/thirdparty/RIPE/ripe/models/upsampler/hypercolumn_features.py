import torch
import torch.nn.functional as F
from torch import nn


class HyperColumnFeatures(nn.Module):
    """
    Interpolate 3D tensor given N sparse 2D positions
    Input
      x: list([C, H, W]) list of feature tensors at different scales (e.g. from a U-Net) -> extract hypercolumn features
      pos: [N, 2] tensor of positions
      H: int, height of the OUTPUT map
      W: int, width of the OUTPUT map

    Returns
      [N, C] sampled features at 2d positions
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.name = "HyperColumnFeatures"

    def normgrid(self, x, H, W):
        return 2.0 * (x / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype))) - 1.0

    def extract_values_at_poses(self, x, pos, H, W):
        """Extract values from tensor x at the positions given by pos.

        Args:
        - x (Tensor): Tensor of size (C, H, W).
        - pos (Tensor): Tensor of size (N, 2) containing the x, y positions.

        Returns:
        - values (Tensor): Tensor of size (N, C) with the values from f at the positions given by p.
        """

        # check if grid is float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        grid = self.normgrid(pos, H, W).unsqueeze(-2)

        x = F.grid_sample(x, grid, mode=self.mode, align_corners=True)
        return x.permute(0, 2, 3, 1).squeeze(-2)

    def forward(self, x, pos, H, W):
        descs = []

        for layer in x:
            desc = self.extract_values_at_poses(layer, pos, H, W)
            descs.append(desc)

        return descs
