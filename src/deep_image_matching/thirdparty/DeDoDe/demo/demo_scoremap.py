import torch
from PIL import Image
import numpy as np

from DeDoDe import dedode_detector_L
from DeDoDe.utils import tensor_to_pil, get_best_device


if __name__ == "__main__":
    device = get_best_device()
    detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
    H, W = 784, 784
    im_path = "assets/im_A.jpg"

    out = detector.detect_from_path(im_path, dense = True, H = H, W = W)

    logit_map = out["dense_keypoint_logits"].clone()
    min = logit_map.max() - 3
    logit_map[logit_map < min] = min
    logit_map = (logit_map-min)/(logit_map.max()-min)
    logit_map = logit_map.cpu()[0].expand(3,H,W)
    im_A = torch.tensor(np.array(Image.open(im_path).resize((W,H)))/255.).permute(2,0,1)
    tensor_to_pil(logit_map * logit_map  +  0.15 * (1-logit_map) * im_A).save("demo/dense_logits.png")
