try:
    import math
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    file_path = Path(__file__)
    for f in file_path.parent.glob("get_patches*.so"):
        torch.ops.load_library(f)

    class get_patches(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fmap, points, kernel_size):
            fmap = fmap.contiguous()
            points = points.contiguous()
            patches = torch.ops.custom_ops.get_patches_forward(
                fmap, points, kernel_size
            )

            ctx.save_for_backward(points, torch.tensor(fmap.shape))

            return patches

        @staticmethod
        def backward(ctx, d_patches):
            points, shape = ctx.saved_tensors
            H = shape[1].cpu().item()
            W = shape[2].cpu().item()
            d_fmap = torch.ops.custom_ops.get_patches_backward(
                d_patches.contiguous(), points, H, W
            )
            return d_fmap, None, None

    def get_patches_torch(fmap: Tensor, points: Tensor, K: int):
        # fmap: CxHxW
        # points: Nx2
        # pad the fmap
        N = points.shape[0]
        C = fmap.shape[0]
        radius = (K - 1.0) / 2.0
        pad_left_top = math.floor(radius)
        pad_right_bottom = math.ceil(radius)
        # K=2, radius=0.5, pad_left_top=0, pad_right_bottom=1
        # K=3, radius=1.0, pad_left_top=1, pad_right_bottom=1
        # K=4, radius=1.5, pad_left_top=1, pad_right_bottom=2
        # K=5, radius=2.0, pad_left_top=2, pad_right_bottom=2
        # Cx(H+K-1)x(W+K-1)
        map_pad = F.pad(
            fmap.unsqueeze(0),
            (pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom),
        ).squeeze(0)
        patches_left = (points[:, 1] - pad_left_top).long()
        patches_top = (points[:, 0] - pad_left_top).long()
        patches_right = patches_left + K
        patches_bottom = patches_top + K

        patches = map_pad[:, patches_top:patches_bottom, patches_left:patches_right]

        return patches

except:
    print("\033[1;41;37m Please build the custom operations first!!\033[0m")
    exit(-1)
