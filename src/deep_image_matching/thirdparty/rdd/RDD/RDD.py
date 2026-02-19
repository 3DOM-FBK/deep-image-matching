# Description: RDD model
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .utils import NestedTensor, nested_tensor_from_tensor_list, to_pixel_coords, read_config
from .models.detector import build_detector
from .models.descriptor import build_descriptor
from .models.soft_detect import SoftDetect
from .models.interpolator import InterpolateSparse2d

class RDD(nn.Module):

    def __init__(self, detector, descriptor, detection_threshold=0.5, top_k=4096, train_detector=False, device='cuda'):
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor
        self.interpolator = InterpolateSparse2d('bicubic')
        self.detection_threshold = detection_threshold
        self.top_k = top_k
        self.device = device
        if train_detector:
            for p in self.detector.parameters():
                p.requires_grad = True
            for p in self.descriptor.parameters():
                p.requires_grad = False
        else:
            for p in self.detector.parameters():
                p.requires_grad = False
            for p in self.descriptor.parameters():
                p.requires_grad = True
        
        self.softdetect = None
        self.stride = descriptor.stride

    def train(self, mode=True):
        super().train(mode)
        
    def eval(self):
        super().eval()
        self.set_softdetect(top_k=self.top_k, scores_th=0.01)
        
    def forward(self, samples: NestedTensor):
        
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        scoremap = self.detector(samples)
        
        feats, matchibility = self.descriptor(samples)
        
        return feats, scoremap, matchibility
    
    def set_softdetect(self, top_k=4096, scores_th=0.01):
        self.softdetect = SoftDetect(radius=2, top_k=top_k, scores_th=scores_th)
    
    @torch.inference_mode()
    def filter(self, matchibility):
        # Filter out keypoints on the border
        B, _, H, W = matchibility.shape
        frame = torch.zeros(B, H, W, device=matchibility.device)
        frame[:, self.stride:-self.stride, self.stride:-self.stride] = 1
        matchibility = matchibility * frame
        return matchibility
    
    @torch.inference_mode()
    def extract(self, x):
        if self.softdetect is None:
            self.eval()
        
        x, rh1, rw1 = self.preprocess_tensor(x)
        x = x.to(self.device).float()
        B, _, _H1, _W1 = x.shape
        M1, K1, H1 = self.forward(x)
        M1 = F.normalize(M1, dim=1)
        
        keypoints, kptscores, scoredispersitys = self.softdetect(K1)
        
        keypoints = torch.vstack([keypoints[b].unsqueeze(0) for b in range(B)])
        kptscores = torch.vstack([kptscores[b].unsqueeze(0) for b in range(B)])
        
        keypoints = to_pixel_coords(keypoints, _H1, _W1)
        
        feats = self.interpolator(M1, keypoints, H = _H1, W = _W1)
        
        feats = F.normalize(feats, dim=-1)
		
        # Correct kpt scale
        keypoints = keypoints * torch.tensor([rw1,rh1], device=keypoints.device).view(1, -1)
        valid = kptscores > self.detection_threshold

        return [  
                    {'keypoints': keypoints[b][valid[b]],
                    'scores': kptscores[b][valid[b]],
                    'descriptors': feats[b][valid[b]]} for b in range(B) 
                ]	
        
    @torch.inference_mode()
    def extract_3rd_party(self, x, model='aliked'):
        """
        one image per batch
        """
        x, rh1, rw1 = self.preprocess_tensor(x)
        B, _, _H1, _W1 = x.shape
        if model == 'aliked':
            from third_party import extract_aliked_kpts
            img = x
            mkpts, scores = extract_aliked_kpts(img, self.device)
        else:
            raise ValueError('Unknown model')
    
        M1, _ = self.descriptor(x)
        M1 = F.normalize(M1, dim=1)
        
        if mkpts.shape[1] > self.top_k:
            idx = torch.argsort(scores, descending=True)[0][:self.top_k]
            mkpts = mkpts[:,idx]
            scores = scores[:,idx]

        feats = self.interpolator(M1, mkpts, H = _H1, W = _W1)
        feats = F.normalize(feats, dim=-1)
        mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)
        
        return [  
				   {'keypoints': mkpts[b],
                    'scores': scores[b],
					'descriptors': feats[b]} for b in range(B) 
			   ]
        
    @torch.inference_mode()
    def extract_dense(self, x, n_limit=30000, thr=0.01):
        
        img_size = x.shape[-2:]
        
        x, rh1, rw1 = self.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.forward(x)
        M1 = F.normalize(M1, dim=1)
        
        keypoints, kptscores, scoredispersitys = self.softdetect(K1)
        
        keypoints = torch.vstack([keypoints[b].unsqueeze(0) for b in range(B)])
        kptscores = torch.vstack([kptscores[b].unsqueeze(0) for b in range(B)])
        
        keypoints = to_pixel_coords(keypoints, _H1, _W1)
        
        feats = self.interpolator(M1, keypoints, H = _H1, W = _W1)
        
        feats = F.normalize(feats, dim=-1)
        
        H1 = self.filter(H1)
        
        dense_kpts, dense_scores, inds = self.sample_dense_kpts(H1, n_limit=n_limit)
 
        dense_keypoints = to_pixel_coords(dense_kpts, _H1, _W1)

        dense_feats = self.interpolator(M1, dense_keypoints, H = _H1, W = _W1)
        
        dense_feats = F.normalize(dense_feats, dim=-1)
        
        keypoints = keypoints * torch.tensor([rw1,rh1], device=keypoints.device).view(1, -1)
        dense_keypoints = dense_keypoints * torch.tensor([rw1,rh1], device=dense_keypoints.device).view(1, -1)	

        valid = kptscores > self.detection_threshold
        valid_dense = dense_scores > thr		

        return [  
                    {'keypoints': keypoints[b][valid[b]],
                    'scores': kptscores[b][valid[b]],
                    'descriptors': feats[b][valid[b]], 
                    'keypoints_dense': dense_keypoints[b][valid_dense[b]],
                    'scores_dense': dense_scores[b][valid_dense[b]],
                    'descriptors_dense': dense_feats[b][valid_dense[b]],
                    'image_size': img_size,
                    } for b in range(B)
                ]
        
    @torch.inference_mode()
    def sample_dense_kpts(self, keypoint_logits, threshold=0.01, n_limit=30000, force_kpts = True):
        
        B, K, H, W = keypoint_logits.shape

        if n_limit < 0 or n_limit > H*W:
            n_limit = min(H*W - 1, n_limit)

        scoremap = keypoint_logits.permute(0,2,3,1)

        scoremap = scoremap.reshape(B, H, W)

        frame = torch.zeros(B, H, W, device=keypoint_logits.device)

        frame[:, 1:-1, 1:-1] = 1

        scoremap = scoremap * frame

        scoremap = scoremap.reshape(B, H*W)

        grid = self.get_grid(B, H, W, device = keypoint_logits.device)

        inds = torch.topk(scoremap, n_limit, dim=1).indices

        # inds = torch.multinomial(scoremap, top_k, replacement=False)
        kpts = torch.gather(grid, 1, inds[..., None].expand(B, n_limit, 2))
        scoremap = torch.gather(scoremap, 1, inds)
        if force_kpts:
            valid = scoremap > threshold
            kpts = kpts[valid][None]
            scoremap = scoremap[valid][None]

        return kpts, scoremap, inds

    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            x = torch.tensor(x).permute(2,0,1)[None]
        x = x.to(self.device).float()

        H, W = x.shape[-2:]

        _H, _W = (H//32) * 32, (W//32) * 32

        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw

    @torch.inference_mode()
    def get_grid(self, B, H, W, device = None):
        x1_n = torch.meshgrid(
        *[
            torch.linspace(
                -1 + 1 / n, 1 - 1 / n, n, device=device
            )
            for n in (B, H, W)
        ]
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        return x1_n

def build(config=None, weights=None):
    if config is None:
        config = read_config('./configs/default.yaml')
    if weights is not None:
        config['weights'] = weights
    device = torch.device(config['device'])
    detector = build_detector(config)
    descriptor = build_descriptor(config)
    model = RDD(
        detector, 
        descriptor, 
        detection_threshold=config['detection_threshold'], 
        top_k=config['top_k'], 
        train_detector=config['train_detector'],
        device=device
    )
    if 'weights' in config and config['weights'] is not None:
        model.load_state_dict(torch.load(config['weights'], map_location='cpu'))
    model.to(device)
    return model