import yaml
import numpy as np
import torch
from pathlib import Path

from ..thirdparty.liftfeat.models.liftfeat_wrapper import MODEL_PATH, LiftFeat
from ..thirdparty.RIPE.ripe import vgg_hyper
from .extractor_base import ExtractorBase


class RIPEExtractor(ExtractorBase):
    _default_conf = {
        "name": "ripe",
        "max_keypoints": 4000,
        "detect_threshold": 0.5,
    }
    required_inputs = []
    grayscale = False
    descriptor_size = 128
    

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self.config.get("extractor")

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor = vgg_hyper().to(dev)
        self._extractor.eval()
        
        # Enable cuDNN autotuner for better performance
        if dev.type == "cuda":
            torch.backends.cudnn.benchmark = True
            
            # Try to use half precision (FP16) for ~2x speedup on modern GPUs
            # Using autocast instead of .half() to avoid dtype mismatch issues
            self._use_fp16 = True
            print("RIPE: Using FP16 (half precision) with autocast for faster inference")
        else:
            self._use_fp16 = False
            
        # Try to compile the model with PyTorch 2.0+ for additional speedup
        try:
            if hasattr(torch, 'compile'):
                self._extractor = torch.compile(self._extractor, mode="reduce-overhead")
                print("RIPE: Model compiled with torch.compile for faster inference")
        except Exception:
            pass  # torch.compile not available or failed
        
        self.max_num_keypoints = cfg.get("max_keypoints", self._default_conf["max_keypoints"])
        self.detect_threshold = cfg.get("detect_threshold", self._default_conf["detect_threshold"])

    @torch.no_grad()
    def _extract(self, image: np.ndarray) -> np.ndarray:
        # Convert numpy array to tensor
        device = next(self._extractor.parameters()).device
        image_tensor = self._frame2tensor(image, device)
        
        # Extract features using RIPE's detectAndCompute method with FP16 if enabled
        if self._use_fp16:
            with torch.cuda.amp.autocast():
                kpts, descs, scores = self._extractor.detectAndCompute(
                    image_tensor, 
                    threshold=self.detect_threshold,
                    top_k=self.max_num_keypoints
                )
        else:
            kpts, descs, scores = self._extractor.detectAndCompute(
                image_tensor, 
                threshold=self.detect_threshold,
                top_k=self.max_num_keypoints
            )
        
        # Convert tensors to numpy arrays
        feats = {
            "keypoints": kpts.cpu().numpy(),
            "descriptors": descs.cpu().numpy(),
            "scores": scores.cpu().numpy()
        }
        
        # Transpose descriptors to match expected format (D x N)
        feats["descriptors"] = feats["descriptors"].T

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        """
        Convert a frame to a tensor.

        Args:
            image: The image to be converted
            device: The device to convert to (defaults to 'cuda')
        """
        # Ensure image is contiguous and in the right format
        if len(image.shape) == 2:
            image = image[None][None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)[None, :, :, :]
        
        # Make sure array is contiguous
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        # Use torch.from_numpy for zero-copy conversion (much faster)
        # then divide and move to device
        tensor = torch.from_numpy(image).float().div_(255.0).to(device, non_blocking=True)
        return tensor

    def _rbd(self, data: dict) -> dict:
        """Remove batch dimension from elements in data"""
        return {
            k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()
        }


if __name__ == "__main__":
    pass
