import numpy as np
import torch
import os
import traceback

from .matcher_base import FeaturesDict, MatcherBase
from ..thirdparty.accelerated_features.modules.lighterglue import LighterGlue


def featuresDict2Lightglue(feats: dict, device: torch.device) -> dict:
    """
    Convert a feature dictionary to LightGlue-compatible format.

    Ensures:
        keypoints   : (B, N, 2)
        descriptors : (B, N, D)
        scores      : (B, N) (if present)
    """

    # 1. Unwrap list / tuple
    feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in feats.items()}

    # 2. Sanity checks
    if "keypoints" not in feats or "descriptors" not in feats:
        raise KeyError("features must contain 'keypoints' and 'descriptors'")

    kpts = feats["keypoints"]          # (N, 2)
    desc = feats["descriptors"]        # (N, D) or (D, N)

    if kpts.ndim != 2 or kpts.shape[1] != 2:
        raise ValueError(f"Invalid keypoints shape: {kpts.shape}")

    N = kpts.shape[0]

    # 3. Fix descriptor layout using keypoint count
    if desc.ndim != 2:
        raise ValueError(f"Invalid descriptors shape: {desc.shape}")

    # Case A: (D, N) → transpose
    if desc.shape[1] == N and desc.shape[0] != N:
        desc = desc.T

    # Case B: (N, D) → OK
    elif desc.shape[0] == N:
        pass

    else:
        raise ValueError(
            f"Descriptor / keypoint mismatch: "
            f"descriptors={desc.shape}, keypoints={kpts.shape}"
        )

    feats["descriptors"] = desc  # now guaranteed (N, D)

    # 4. Remove unused keys (LightGlue don't use)
    feats.pop("feature_path", None)
    feats.pop("im_path", None)

    # 5. Add batch dimension
    feats = {k: v[None] for k, v in feats.items()}

    # 6. Convert to torch.Tensor
    feats = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in feats.items()
    }

    return feats


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


class LighterGlueMatcher(MatcherBase):
    _default_conf = {
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
    required_inputs = []
    min_matches = 20
    max_feat_no_tiling = 200000

    def __init__(self, config, local_features="superpoint") -> None:
        """Initializes a LighterGlueMatcher"""
        self._localfeatures = local_features
        super().__init__(config)
        if self._localfeatures != "xfeat":
            raise ValueError(f"Unsupported local feature extractor: {self._localfeatures}")
        # Load LighterGlue with weights from thirdparty accelerated_features
        weights_path = os.path.join(
            os.path.dirname(__file__),
            '../thirdparty/accelerated_features/weights/xfeat-lighterglue.pt'
        )
        self._lightergule = LighterGlue(weights=weights_path)

    @torch.no_grad()
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
    ) -> np.ndarray:
        """
        Match features using XFeat's match_lighterglue method.
        
        match_lighterglue expects:
        - keypoints: (N, 2) as numpy array or torch tensor
        - descriptors: (N, D) as numpy array or torch tensor
        - image_size: (H, W) as numpy array
        
        Returns:
        - idxs: match indices (N, 2) with [idx_in_img0, idx_in_img1]
        """
        
        # Prepare feature dicts - DO NOT use featuresDict2Lightglue as it adds batch dims
        d0 = {}
        d1 = {}
        
        # Helper function to unwrap and squeeze
        def prepare_feature(feat, name):
            # Unwrap from list/tuple
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            # Convert tensor to numpy
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            # Remove batch dimension if present
            if isinstance(feat, np.ndarray) and feat.ndim == 3:
                feat = feat[0]
            return feat
        
        # Prepare keypoints
        kpts0 = prepare_feature(feats0['keypoints'], 'keypoints0')
        kpts1 = prepare_feature(feats1['keypoints'], 'keypoints1')
        
        # Prepare descriptors
        desc0 = prepare_feature(feats0['descriptors'], 'descriptors0')
        desc1 = prepare_feature(feats1['descriptors'], 'descriptors1')
        
        # Validate and fix descriptor shape (should be N, D not D, N)
        N0 = kpts0.shape[0]
        N1 = kpts1.shape[0]
        
        if desc0.ndim != 2:
            raise ValueError(f"Descriptors0 must be 2D, got shape {desc0.shape}")
        if desc1.ndim != 2:
            raise ValueError(f"Descriptors1 must be 2D, got shape {desc1.shape}")
        
        # Fix descriptor layout if needed (D, N) → (N, D)
        if desc0.shape[0] != N0 and desc0.shape[1] == N0:
            desc0 = desc0.T
        if desc1.shape[0] != N1 and desc1.shape[1] == N1:
            desc1 = desc1.T
        
        if desc0.shape[0] != N0:
            raise ValueError(f"Descriptors0 shape {desc0.shape} doesn't match keypoints {kpts0.shape}")
        if desc1.shape[0] != N1:
            raise ValueError(f"Descriptors1 shape {desc1.shape} doesn't match keypoints {kpts1.shape}")
        
        # Keep as numpy float32 (XFeat will handle torch conversion internally)
        d0['keypoints'] = kpts0.astype(np.float32)
        d1['keypoints'] = kpts1.astype(np.float32)
        d0['descriptors'] = desc0.astype(np.float32)
        d1['descriptors'] = desc1.astype(np.float32)
        
        # Handle image_size - XFeat will convert with torch.tensor()
        if 'image_size' in feats0:
            img_size0 = feats0['image_size']
            if isinstance(img_size0, (list, tuple)):
                img_size0 = img_size0[0]
            if isinstance(img_size0, torch.Tensor):
                img_size0 = img_size0.cpu().numpy()
            # Keep as int for image dimensions
            d0['image_size'] = np.asarray(img_size0, dtype=np.int32)
        
        if 'image_size' in feats1:
            img_size1 = feats1['image_size']
            if isinstance(img_size1, (list, tuple)):
                img_size1 = img_size1[0]
            if isinstance(img_size1, torch.Tensor):
                img_size1 = img_size1.cpu().numpy()
            # Keep as int for image dimensions
            d1['image_size'] = np.asarray(img_size1, dtype=np.int32)
        
        # If image_size is not available, raise error
        if 'image_size' not in d0 or 'image_size' not in d1:
            raise ValueError("image_size not found in features - required by XFeat's match_lighterglue")
        
        # Call LighterGlue's forward method
        # Prepare data dict for LighterGlue forward - need to convert to torch tensors with batch dimension
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LightGlue in kornia expects UNNORMALIZED keypoints in pixel coordinates
        # It will normalize them internally using image_size
        img_size0 = d0['image_size'].astype(np.float32)
        img_size1 = d1['image_size'].astype(np.float32)
        
        # Swap image_size from [H, W] to [W, H] format that kornia LightGlue expects
        img_size0_wh = np.array([img_size0[1], img_size0[0]], dtype=np.float32)
        img_size1_wh = np.array([img_size1[1], img_size1[0]], dtype=np.float32)
        
        # LightGlue expects batch dimension: (B, N, 2) for keypoints and (B, N, D) for descriptors
        # image_size also needs batch dimension: (B, 2)
        data = {
            'keypoints0': torch.from_numpy(d0['keypoints']).unsqueeze(0).to(device),
            'descriptors0': torch.from_numpy(d0['descriptors']).unsqueeze(0).to(device),
            'image_size0': torch.from_numpy(img_size0_wh).unsqueeze(0).to(device),
            'keypoints1': torch.from_numpy(d1['keypoints']).unsqueeze(0).to(device),
            'descriptors1': torch.from_numpy(d1['descriptors']).unsqueeze(0).to(device),
            'image_size1': torch.from_numpy(img_size1_wh).unsqueeze(0).to(device),
        }
        
        try:
            filter_threshold = self.config["matcher"]["filter_threshold"]
            result = self._lightergule(data, min_conf=filter_threshold)
        except Exception as e:
            print(f"\n{'='*80}")
            print("Full traceback for LighterGlue forward error:")
            traceback.print_exc()
            print(f"{'='*80}\n")
            raise RuntimeError(f"LighterGlue forward failed: {str(e)}") from e
        
        # Extract match indices from result
        # LightGlue returns matches in the result dict
        # result['matches'] is a list containing match pairs (one per batch item)
        if 'matches' in result and isinstance(result['matches'], list) and len(result['matches']) > 0:
            # Get matches from batch 0
            matches_tensor = result['matches'][0]
            
            if isinstance(matches_tensor, torch.Tensor):
                idxs = matches_tensor.cpu().numpy()
            else:
                idxs = np.array(matches_tensor)
            
            print(f"DEBUG: Using result['matches']: shape={idxs.shape}, dtype={idxs.dtype}")
            print(f"DEBUG: Sample matches (first 10):\n{idxs[:10]}")
        else:
            # Fallback: return empty matches
            idxs = np.empty((0, 2), dtype=np.int32)
        
        return idxs.astype(np.int32)
