from typing import Union

import cv2
import kornia.feature as KF
import numpy as np
import torch
import torchvision.transforms as transforms

from ..thirdparty.DeDoDe.DeDoDe import dedode_descriptor_G, dedode_detector_L
from .extractor_base import ExtractorBase, FeaturesDict


class DeDoDeExtractor(ExtractorBase):
    dedode_detector_L_url = (
        "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth"
    )
    dedode_descriptor_G_url = (
        "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth"
    )
    dedode_descriptor_B_url = (
        "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth"
    )

    _default_conf = {
        "name": "dedode",
    }
    required_inputs = ["image"]
    grayscale = False
    descriptor_size = 256
    detection_noise = 2.0

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        cfg = self.config.get("extractor")

        # Load extractor and descriptor
        device = torch.device(self._device if torch.cuda.is_available() else "cpu")
        self.detector = dedode_detector_L(
            weights=torch.hub.load_state_dict_from_url(self.dedode_detector_L_url, map_location=device)
        )
        self.descriptor = dedode_descriptor_G(
            weights=torch.hub.load_state_dict_from_url(self.dedode_descriptor_G_url, map_location=device)
        )

        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_features = cfg["n_features"]

    @torch.inference_mode()
    def _extract(self, image: Union[np.ndarray, torch.Tensor]) -> dict:
        H, W, C = image.shape
        resized_image = cv2.resize(image, (784, 784))
        standard_im = np.array(resized_image) / 255.0
        norm_image = self.normalizer(torch.from_numpy(standard_im).permute(2, 0, 1)).float().to(self._device)[None]
        batch = {"image": norm_image}
        detections_A = self.detector.detect(batch, num_keypoints=self.num_features)
        keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
        description_A = self.descriptor.describe_keypoints(batch, keypoints_A)["descriptions"]
        kpts = keypoints_A.cpu().detach().numpy()[0]
        des = description_A.cpu().detach().numpy()[0]

        kpts[:, 0] = (kpts[:, 0] + 1) * W / 2
        kpts[:, 1] = (kpts[:, 1] + 1) * H / 2
        feats = FeaturesDict(keypoints=kpts, descriptors=des.T)

        return feats

    def _preprocess_input(self, image: np.ndarray, device: str = "cuda"):
        """
        Convert a frame to a tensor.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cuda')
        """
        if len(image.shape) == 2:
            image = image[None][None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)[None]
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)
