import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np
from ...DeDoDe.utils import get_best_device


class DeDoDeDescriptor(nn.Module):
    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        import torchvision.transforms as transforms

        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(
        self,
        batch,
    ):
        if "im_A" in batch:
            images = torch.cat((batch["im_A"], batch["im_B"]))
        else:
            images = batch["image"]
        features, sizes = self.encoder(images)
        descriptor = 0
        context = None
        scales = self.decoder.scales
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_descriptor, context = self.decoder(
                feature_map, scale=scale, context=context
            )
            descriptor = descriptor + delta_descriptor
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                descriptor = F.interpolate(
                    descriptor, size=size, mode="bilinear", align_corners=False
                )
                context = F.interpolate(
                    context, size=size, mode="bilinear", align_corners=False
                )
        return {"description_grid": descriptor}

    @torch.inference_mode()
    def describe_keypoints(self, batch, keypoints):
        self.train(False)
        description_grid = self.forward(batch)["description_grid"]
        described_keypoints = F.grid_sample(
            description_grid.float(),
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].mT
        return {"descriptions": described_keypoints}

    def read_image(self, im_path, H=784, W=784, device=get_best_device()):
        return (
            self.normalizer(
                torch.from_numpy(
                    np.array(Image.open(im_path).resize((W, H))) / 255.0
                ).permute(2, 0, 1)
            )
            .float()
            .to(device)[None]
        )

    def describe_keypoints_from_path(self, im_path, keypoints, H=784, W=784):
        batch = {"image": self.read_image(im_path, H=H, W=W)}
        return self.describe_keypoints(batch, keypoints)
