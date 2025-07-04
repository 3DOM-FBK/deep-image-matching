import numpy as np
import torch
from PIL import Image
from ..ncc import refinement_laf
from . import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G
from .matchers.dual_softmax_matcher import DualSoftMaxMatcher
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", category=UserWarning)

# import matplotlib.pyplot as plt
# import cv2
class dedode2_module:
    def __init__(self, **args):
        self.threshold = 0.01
        self.num_features = 8000
        self.upright = False
        
        for k, v in args.items():
           setattr(self, k, v)

        with torch.inference_mode():
            self.detector = dedode_detector_L(weights=None)
            self.descriptor = dedode_descriptor_B(weights=None)
            self.matcher = DualSoftMaxMatcher()
        
        
    def get_id(self):
        return ('dedodev2_th_' + str(self.threshold) + '_nfeat_' + str(self.num_features) + '_upright_' + str(self.upright)).lower()

    
    def run(self, **args):
        im1 = Image.open(args['im1']).convert('RGB')
        im2 = Image.open(args['im2']).convert('RGB')
        
        # im2 = Image.fromarray(np.transpose(np.flip(np.asarray(im2), axis=0), (1, 0, 2)))
        # im2.save('rot_test.png')
        # im2 = Image.open('rot_test.png')
        
        W1, H1 = im1.size
        W2, H2 = im2.size

        with torch.inference_mode():
            detections1 = self.detector.detect_from_image(im1, num_keypoints = self.num_features, H=H1, W=W1)
            kps1_, p1 = detections1["keypoints"], detections1["confidence"]
            descs1 = self.descriptor.describe_keypoints_from_image(im1, kps1_, H=H1, W=W1)["descriptions"]

            detections2 = self.detector.detect_from_image(im2, num_keypoints = self.num_features, H=H2, W=W2)
            kps2_, p2 = detections2["keypoints"], detections2["confidence"]
            descs2 = self.descriptor.describe_keypoints_from_image(im2, kps2_, H=H2, W=W2)["descriptions"]
                        
            m1, m2, batch_ids = self.matcher.match(kps1_, descs1, kps2_, descs2, P_A=p1, P_B=p2, normalize=True, inv_temp=20, threshold=self.threshold)
                       
        kps1, kps2 = self.matcher.to_pixel_coords(m1, m2, H1, W1, H2, W2)

        if not self.upright:
            W2_orig, H2_orig = im2.size

            r_best = 0
            W2_best, H2_best = im2.size
            kps1_best = kps1            
            kps2_best = kps2            

            for r in range(1, 4):
                im2 = Image.fromarray(np.transpose(np.flip(np.asarray(im2), axis=0), (1, 0, 2)))
                W2, H2 = im2.size

                with torch.inference_mode():
                    detections2 = self.detector.detect_from_image(im2, num_keypoints=self.num_features, H=H2, W=W2)
                    kps2_, p2 = detections2["keypoints"], detections2["confidence"]
                    descs2 = self.descriptor.describe_keypoints_from_image(im2, kps2_, H=H2, W=W2)["descriptions"]
                       
                    m1, m2, batch_ids = self.matcher.match(kps1_, descs1, kps2_, descs2, P_A=p1, P_B=p2, normalize=True, inv_temp=20, threshold=self.threshold)
                      
                kps1, kps2 = self.matcher.to_pixel_coords(m1, m2, H1, W1, H2, W2)
                   
                if kps1.shape[0] > kps1_best.shape[0]:
                    r_best = r
                    W2_best = W2
                    H2_best = H2                   
                    kps1_best = kps1
                    kps2_best = kps2

            a = -r_best / 2.0 * np.pi
            R = torch.tensor([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], device=device)
            kps2 = ((R @ (kps2_best.permute(1, 0) - (torch.tensor([[W2_best], [H2_best]], device=device) / 2) ).type(torch.double)) + (torch.tensor([[W2_orig], [H2_orig]], device=device) / 2).type(torch.double)).permute(1, 0)

        # plt.figure()
        # plt.axis('off')
        # img = cv2.imread('rot_test.png', cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img)
        # plt.plot(kps2[:, 0].detach().cpu().numpy(), kps2[:, 1].detach().cpu().numpy(), linestyle='', color='red', marker='.')

        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!

        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_laf}