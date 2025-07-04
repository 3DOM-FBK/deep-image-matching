import torch
import deep_image_matching as dim

from ..ncc import refinement_laf
from .load_image import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class disk_lightglue_module:
    def __init__(self, **args):
        self.nmax_keypoints = 8000
        for k, v in args.items():
           setattr(self, k, v)

        self.grayscale = False
        self.as_float = None
        params = {
            "dir": "./thirdparty/deep-image-matching/assets/example_cyprus", # placeholder
            "pipeline": "disk+lightglue",
            "strategy": "bruteforce",
            "quality": "high",
            "tiling": "none",
            "skip_reconstruction": False,
            "force": True,
            "camera_options": "./thirdparty/deep-image-matching/config/cameras.yaml", # placeholder
            "openmvg": None,
            "verbose": False,
        }
        self.config = dim.Config(params)

        self.config.extractor['max_keypoints'] = self.nmax_keypoints
        self.config.matcher['depth_confidence'] = 0.95
        self.config.matcher['width_confidence'] = 0.99
        self.config.matcher['filter_threshold'] = 0.1

        self.extractor =  dim.extractors.DiskExtractor(self.config)
        self.matcher = dim.matchers.LightGlueMatcher(self.config, local_features="disk")


    def get_id(self):
        return ('disk_nfeat_' + str(self.max_keypoints)).lower()

            
    def run(self, **args):

        with torch.inference_mode():
            image1 = load_image(args['im1'], self.grayscale, self.as_float)
            image2 = load_image(args['im2'], self.grayscale, self.as_float)
            feats1 = self.extractor._extract(image1)
            feats2 = self.extractor._extract(image2)
            matches = self.matcher._match_pairs(feats1, feats2)
            # print(feats1['keypoints'].shape, feats2['keypoints'].shape, matches.shape)

        kps1 = torch.tensor(feats1['keypoints'][matches[:,0],:]).to(device)
        kps2 = torch.tensor(feats2['keypoints'][matches[:,1],:]).to(device)
        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!
    
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_laf}
