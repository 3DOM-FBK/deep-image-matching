import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np

from ...DeDoDe.utils import (
    sample_keypoints,
    to_pixel_coords,
    to_normalized_coords,
    get_best_device,
)


class DeDoDeDetector(nn.Module):
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
        logits = 0
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(
                feature_map, context=context, scale=scale
            )
            logits = (
                logits + delta_logits.float()
            )  # ensure float (need bf16 doesnt have f.interpolate)
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(
                    logits, size=size, mode="bicubic", align_corners=False
                )
                context = F.interpolate(
                    context.float(), size=size, mode="bilinear", align_corners=False
                )
        return {"keypoint_logits": logits.float()}

    @torch.inference_mode()
    def detect(self, batch, num_keypoints=10_000):
        self.train(False)
        keypoint_logits = self.forward(batch)["keypoint_logits"]
        B, K, H, W = keypoint_logits.shape
        keypoint_p = (
            keypoint_logits.reshape(B, K * H * W)
            .softmax(dim=-1)
            .reshape(B, K, H * W)
            .sum(dim=1)
        )
        keypoints, confidence = sample_keypoints(
            keypoint_p.reshape(B, H, W),
            use_nms=False,
            sample_topk=True,
            num_samples=num_keypoints,
            return_scoremap=True,
            sharpen=False,
            upsample=False,
            increase_coverage=True,
        )
        return {"keypoints": keypoints, "confidence": confidence}

    @torch.inference_mode()
    def detect_dense(self, batch):
        self.train(False)
        keypoint_logits = self.forward(batch)["keypoint_logits"]
        return {"dense_keypoint_logits": keypoint_logits}

    def read_image(self, im_path, H=784, W=784, device=get_best_device()):
        pil_im = Image.open(im_path).resize((W, H))
        standard_im = np.array(pil_im) / 255.0
        return (
            self.normalizer(torch.from_numpy(standard_im).permute(2, 0, 1))
            .float()
            .to(device)[None]
        )

    def detect_from_path(
        self, im_path, num_keypoints=30_000, H=784, W=784, dense=False
    ):
        batch = {"image": self.read_image(im_path, H=H, W=W)}
        if dense:
            return self.detect_dense(batch)
        else:
            return self.detect(batch, num_keypoints=num_keypoints)

    def to_pixel_coords(self, x, H, W):
        return to_pixel_coords(x, H, W)

    def to_normalized_coords(self, x, H, W):
        return to_normalized_coords(x, H, W)
