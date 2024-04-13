import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from .extractor_base import ExtractorBase, FeaturesDict


class KeyNet(ExtractorBase):
    _default_conf = {
        "name:": "",
    }
    required_inputs = ["image"]
    grayscale = True
    descriptor_size = 128
    detection_noise = 2.0

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        cfg = self.config.get("extractor")

        # Load extractor
        self._extractor = KF.KeyNetAffNetHardNet(
            num_features=cfg["n_features"],
            upright=cfg["upright"],
            device=self._device,
        )

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        image = K.image_to_tensor(image, False).float() / 255.0
        if self._device == "cpu":
            image = image.cpu()
        if self._device == "cuda":
            image = image.cuda()
        keypts = self._extractor(image)
        laf = keypts[0].cpu().detach().numpy()
        kpts = keypts[0].cpu().detach().numpy()[-1, :, :, -1]
        des = keypts[2].cpu().detach().numpy()[-1, :, :].T
        feats = FeaturesDict(keypoints=kpts, descriptors=des)

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
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
