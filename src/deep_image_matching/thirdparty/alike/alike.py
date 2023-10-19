import logging
import os
import cv2
import torch
from copy import deepcopy
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import math
import time

from .alnet import ALNet
from .soft_detect import DKD

configs = {
    "alike-t": {
        "c1": 8,
        "c2": 16,
        "c3": 32,
        "c4": 64,
        "dim": 64,
        "single_head": True,
        "radius": 2,
        "model_path": os.path.join(os.path.split(__file__)[0], "models", "alike-t.pth"),
    },
    "alike-s": {
        "c1": 8,
        "c2": 16,
        "c3": 48,
        "c4": 96,
        "dim": 96,
        "single_head": True,
        "radius": 2,
        "model_path": os.path.join(os.path.split(__file__)[0], "models", "alike-s.pth"),
    },
    "alike-n": {
        "c1": 16,
        "c2": 32,
        "c3": 64,
        "c4": 128,
        "dim": 128,
        "single_head": True,
        "radius": 2,
        "model_path": os.path.join(os.path.split(__file__)[0], "models", "alike-n.pth"),
    },
    "alike-l": {
        "c1": 32,
        "c2": 64,
        "c3": 128,
        "c4": 128,
        "dim": 128,
        "single_head": False,
        "radius": 2,
        "model_path": os.path.join(os.path.split(__file__)[0], "models", "alike-l.pth"),
    },
}


class ALike(ALNet):
    def __init__(
        self,
        # ================================== feature encoder
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        c4: int = 128,
        dim: int = 128,
        single_head: bool = False,
        # ================================== detect parameters
        radius: int = 2,
        top_k: int = 500,
        scores_th: float = 0.5,
        n_limit: int = 5000,
        device: str = "cpu",
        model_path: str = "",
    ):
        super().__init__(c1, c2, c3, c4, dim, single_head)
        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(
            radius=self.radius,
            top_k=self.top_k,
            scores_th=self.scores_th,
            n_limit=self.n_limit,
        )
        self.device = device

        if model_path != "":
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logging.info(f"Loaded model parameters from {model_path}")
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB"
            )

    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = super().forward(image)

        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # ====================================================

        # BxCxHxW
        descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)

        if ret_dict:
            return {
                "descriptor_map": descriptor_map,
                "scores_map": scores_map,
            }
        else:
            return descriptor_map, scores_map

    def forward(self, img, image_size_max=99999, sort=False, sub_pixel=False):
        """
        :param img: np.array HxWx3, RGB
        :param image_size_max: maximum image size, otherwise, the image will be resized
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        :return: a dictionary with 'keypoints', 'descriptors', 'scores', and 'time'
        """
        H, W, three = img.shape
        assert three == 3, "input image shape should be [HxWx3]"

        # ==================== image size constraint
        image = deepcopy(img)
        max_hw = max(H, W)
        if max_hw > image_size_max:
            ratio = float(image_size_max / max_hw)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        # ==================== convert image to tensor
        image = (
            torch.from_numpy(image)
            .to(self.device)
            .to(torch.float32)
            .permute(2, 0, 1)[None]
            / 255.0
        )

        # ==================== extract keypoints
        start = time.time()

        with torch.no_grad():
            descriptor_map, scores_map = self.extract_dense_map(image)
            keypoints, descriptors, scores, _ = self.dkd(
                scores_map, descriptor_map, sub_pixel=sub_pixel
            )
            keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
            keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])

        if sort:
            indices = torch.argsort(scores, descending=True)
            keypoints = keypoints[indices]
            descriptors = descriptors[indices]
            scores = scores[indices]

        end = time.time()

        return {
            "keypoints": keypoints.cpu().numpy(),
            "descriptors": descriptors.cpu().numpy(),
            "scores": scores.cpu().numpy(),
            "scores_map": scores_map.cpu().numpy(),
            "time": end - start,
        }


if __name__ == "__main__":
    import numpy as np
    from thop import profile

    net = ALike(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False)

    image = np.random.random((640, 480, 3)).astype(np.float32)
    flops, params = profile(net, inputs=(image, 9999, False), verbose=False)
    print("{:<30}  {:<8} GFLops".format("Computational complexity: ", flops / 1e9))
    print("{:<30}  {:<8} KB".format("Number of parameters: ", params / 1e3))
