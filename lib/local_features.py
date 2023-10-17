import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from easydict import EasyDict as edict


from .deep_image_matcher.thirdparty.SuperGlue.models.matching import Matching
from .deep_image_matcher.thirdparty.LightGlue.lightglue.superpoint import SuperPoint
from .deep_image_matcher.thirdparty.alike.alike import ALike, configs
from .deep_image_matcher.thirdparty.LightGlue.lightglue.utils import load_image


class LocalFeatures:
    def __init__(
        self,
        method: str,
        n_features: int,
        cfg: dict = None,
    ) -> None:
        self.n_features = n_features
        self.method = method

        self.kpts = []
        self.descriptors = []
        self.lafs = []

        # If method is ALIKE, load Alike model weights
        if self.method == "ALIKE":
            self.alike_cfg = cfg
            self.model = ALike(
                **configs[self.alike_cfg["model"]],
                device=self.alike_cfg["device"],
                top_k=self.alike_cfg["top_k"],
                scores_th=self.alike_cfg["scores_th"],
                n_limit=self.alike_cfg["n_limit"],
            )

        elif self.method == "ORB":
            self.orb_cfg = cfg

        elif self.method == "DISK":
            self.orb_cfg = cfg
            self.device = torch.device("cuda")
            self.disk = KF.DISK.from_pretrained('depth').to(self.device)

        elif self.method == "KeyNetAffNetHardNet":
            self.kornia_cfg = cfg
            self.device = torch.device("cuda")

        elif self.method == "SuperPoint":
            self.kornia_cfg = cfg

    def ORB(self, images: np.ndarray, w_size : int):
        self.kpts = []
        self.descriptors = []
        self.lafs = []
        for img in images:
            orb = cv2.ORB_create(
                nfeatures=self.n_features,
                scaleFactor=self.orb_cfg["scaleFactor"],
                nlevels=self.orb_cfg["nlevels"],
                edgeThreshold=self.orb_cfg["edgeThreshold"],
                firstLevel=self.orb_cfg["firstLevel"],
                WTA_K=self.orb_cfg["WTA_K"],
                scoreType=self.orb_cfg["scoreType"],
                patchSize=self.orb_cfg["patchSize"],
                fastThreshold=self.orb_cfg["fastThreshold"],
            )

            kp = orb.detect(img, None)
            kp, des = orb.compute(img, kp)
            kpts = cv2.KeyPoint_convert(kp)
            des = des.astype(float)
            laf = None

            self.kpts.append(kpts)
            self.descriptors.append(des)
            self.lafs.append(laf)

        return self.kpts, self.descriptors, laf

    def ALIKE(self, images: np.ndarray, w_size : int):
        self.kpts = []
        self.descriptors = []
        self.lafs = []
        with torch.inference_mode():
            for img in images:
                features = self.model(img, sub_pixel=self.alike_cfg["subpixel"])
                laf = None
                self.kpts.append(features["keypoints"])
                self.descriptors.append(features["descriptors"])
                self.lafs.append(laf)
            return self.kpts, self.descriptors, self.lafs


    def DISK(self, images: np.ndarray, w_size : int):
        # Inspired by: https://github.com/ducha-aiki/imc2023-kornia-starter-pack/blob/main/DISK-adalam-pycolmap-3dreconstruction.ipynb
        self.kpts = []
        self.descriptors = []
        self.lafs = []
        disk = self.disk
        with torch.inference_mode():
            for img in images:
                img = K.image_to_tensor(img, False).float() / 255.0
                img = img.to(self.device)
                features = disk(img, self.n_features, pad_if_not_divisible=True)[0]
                kps1, descs = features.keypoints, features.descriptors
                laf = None

                self.kpts.append(kps1.cpu().detach().numpy())
                self.descriptors.append(descs.cpu().detach().numpy())
                self.lafs.append(laf)

        return self.kpts, self.descriptors, self.lafs

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Normalize the image tensor and reorder the dimensions."""
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        elif image.ndim == 2:
            image = image[None]  # add channel axis
        else:
            raise ValueError(f"Not an image: {image.shape}")
        return torch.tensor(image / 255.0, dtype=torch.float).to(device)

    def SuperPoint(self, images: np.ndarray, w_size : int):
        self.kpts = []
        self.descriptors = []
        self.lafs = []
    
        with torch.inference_mode():
            for img in images:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                image = self._frame2tensor(img, device)
                extractor = SuperPoint(max_num_keypoints=self.n_features).eval().to(device)
                feats = extractor.extract(image, resize=w_size)
                kpt = feats['keypoints'].cpu().detach().numpy()
                desc = feats['descriptors'].cpu().detach().numpy()
                self.kpts.append(kpt.reshape(-1, kpt.shape[-1]))
                self.descriptors.append(desc.reshape(-1, desc.shape[-1]))
                laf = None
                self.lafs.append(laf)

        return self.kpts, self.descriptors, laf

    def KeyNetAffNetHardNet(self, images: np.ndarray, w_size : int):
        self.kpts = []
        self.descriptors = []
        self.lafs = []
        with torch.inference_mode():
            for img in images:
                #img = self.load_torch_image(str(im_path)).to(self.device)
                image = K.image_to_tensor(img, False).float() / 255.0
                image = K.color.rgb_to_grayscale(K.color.bgr_to_rgb(image))
                image = image.cuda()
                keypts = KF.KeyNetAffNetHardNet(
                    num_features=self.n_features, upright=self.kornia_cfg["upright"], device=self.device
                ).forward(image)

                laf = keypts[0].cpu().detach().numpy()
                self.kpts.append(keypts[0].cpu().detach().numpy()[-1, :, :, -1])
                self.descriptors.append(keypts[2].cpu().detach().numpy()[-1, :, :])
                self.lafs.append(laf)

        return self.kpts, self.descriptors, laf


class LocalFeatureExtractor:
    def __init__(
        self,
        local_feature: str = "ORB",
        local_feature_cfg: dict = None,
        n_features: int = 1024,
    ) -> None:
        
        self.local_feature = local_feature
        self.detector_and_descriptor = LocalFeatures(
            local_feature, n_features, local_feature_cfg
        )

    def run(self, im0, im1, w_size) -> None:
        extract = getattr(self.detector_and_descriptor, self.local_feature)
        kpts, descriptors, lafs = extract([im0, im1], w_size)
        return kpts, descriptors, lafs