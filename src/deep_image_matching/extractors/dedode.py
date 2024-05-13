from typing import Union

import kornia.feature as KF
import numpy as np
import torch

from .extractor_base import ExtractorBase, FeaturesDict


class DeDoDeExtractor(ExtractorBase):
    # The DeDoDe extractor.
    # detector_weights: The weights to load for the detector. One of:
    #     'L-upright' (original paper, https://arxiv.org/abs/2308.08479),
    #     'L-C4', 'L-SO2' (from steerers, better for rotations, https://arxiv.org/abs/2312.02152),
    #     'L-C4-v2' (from dedode v2, better at rotations, less clustering, https://arxiv.org/abs/2404.08928)
    #     Default is 'L-C4-v2', but perhaps it should be 'L-C4-v2'?
    # descriptor_weights: The weights to load for the descriptor. One of:
    #     'B-upright','G-upright' (original paper, https://arxiv.org/abs/2308.08479),
    #     'B-C4', 'B-SO2', 'G-C4' (from steerers, better for rotations, https://arxiv.org/abs/2312.02152).
    #     Default is 'G-upright'.
    # amp_dtype: the dtype to use for the model. One of torch.float16 or torch.float32.
    # Default is torch.float16, suitable for CUDA. Use torch.float32 for CPU or MPS

    _default_conf = {
        "name": "dedode",
        "max_keypoints": 10_000,
        "detector_weights": "L-SO2",  # [L-upright, L-C4, L-SO2]
        "descriptor_weights": "G-C4",  # [B-upright, G-upright, B-C4, B-SO2, G-C4]
        "amp_dtype": torch.float16,
    }
    required_inputs = []
    grayscale = False

    # descriptor_size = 512 if "G" in _default_conf["descriptor_weights"] else 256
    descriptor_size = 256

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Update the descriptor size based on the config
        # self.descriptor_size = 512 if "G" in self._default_conf["descriptor_weights"] else 256

        # Load extractor
        cfg = self.config.extractor
        self._extractor = KF.DeDoDe.from_pretrained(
            detector_weights=cfg["detector_weights"],
            descriptor_weights=cfg["descriptor_weights"],
            amp_dtype=cfg["amp_dtype"],
        ).to(self._device)

    @torch.inference_mode()
    def _extract(self, image: Union[np.ndarray, torch.Tensor]) -> dict:
        # Convert image from numpy array to tensor
        image_ = self._preprocess_input(image, self._device)

        # Extract features
        cfg = self.config.extractor
        kpts, scores, descr = self._extractor(
            image_,
            n=cfg["max_keypoints"],
            apply_imagenet_normalization=True,
            pad_if_not_divisible=True,
        )

        # Convert to numpy
        kpts = kpts.cpu().detach().numpy()[0]
        descr = descr.cpu().detach().numpy()[0]
        scores = scores.cpu().detach().numpy()[0]
        feats = FeaturesDict(keypoints=kpts, descriptors=descr.T, scores=scores)

        return feats

    def _preprocess_input(self, image: np.ndarray, device: str = "cuda"):
        """
        Convert a frame to a tensor.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cuda')
        """
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)[None]
        elif image.ndim == 2:
            # Repeat the image 3 times to make it RGB and add a batch dimension
            image = np.repeat(image[None], 3, axis=0)[None]

        return torch.tensor(image / 255.0, dtype=self.config.extractor["amp_dtype"]).to(device)
