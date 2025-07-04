import numpy as np
import torch
import kornia as K
import kornia.feature as KF

try:
    import pydegensac
    pydegensac_off = True
except:
    pydegensac_off = False
    import warnings
    warnings.warn("cannot load pydegensac - DegenSAC module will return no matches")
	
import cv2
import poselib
from lightglue import LightGlue as lg_lightglue, SuperPoint as lg_superpoint, DISK as lg_disk, SIFT as lg_sift, ALIKED as lg_aliked, DoGHardNet as lg_doghardnet
from lightglue.utils import load_image as lg_load_image, rbd as lg_rbd
from .ncc import refinement_laf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# from kornia moons
def laf_from_opencv_kpts(kpts, mrSize=6.0, device=torch.device('cpu')):
    N = len(kpts)
    xy = torch.tensor([(x.pt[0], x.pt[1]) for x in kpts ], device=device, dtype=torch.float).view(1, N, 2)
    scales = torch.tensor([(mrSize * x.size) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1, 1)
    angles = torch.tensor([(-x.angle) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1)
    laf = K.feature.laf_from_center_scale_ori(xy, scales, angles).reshape(1, -1, 2, 3)
    return laf.reshape(1, -1, 2, 3)


class keynetaffnethardnet_module:
    def __init__(self, **args):
        self.upright = False
        self.th = 0.99
        self.num_features = 8000
        
        for k, v in args.items():
           setattr(self, k, v)

        with torch.inference_mode():
            self.detector = K.feature.KeyNetAffNetHardNet(num_features=self.num_features, upright=self.upright, device=device)
        
        
    def get_id(self):
        return ('keynetaffnethardnet_upright_' + str(self.upright) + '_th_' + str(self.th) + '_nfeat_' + str(self.num_features)).lower()

    
    def run(self, **args):    
        with torch.inference_mode():
            kps1, _ , descs1 = self.detector(K.io.load_image(args['im1'], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            kps2, _ , descs2 = self.detector(K.io.load_image(args['im2'], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
            val, idxs = K.feature.match_smnn(descs1.squeeze(), descs2.squeeze(), self.th)        
        
        pt1 = None
        pt2 = None
        kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)
        
        pt1, pt2, Hs_laf = refinement_laf(None, None, data1=kps1, data2=kps2, img_patches=False)    
    
        return {'pt1': pt1, 'pt2': pt2, 'kp1': kps1, 'kp2': kps2, 'Hs': Hs_laf, 'val': val}


class pydegensac_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
        self.mode = 'fundamental_matrix'
              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('pydegensac_' + self.mode + '_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters)).lower()

    if pydegensac_off:
        def run(self, **args):  
            pt1 = args['pt1']
            pt2 = args['pt2']
            Hs = args['Hs']
        
            if torch.is_tensor(pt1):
                pt1 = np.ascontiguousarray(pt1.detach().cpu())
                pt2 = np.ascontiguousarray(pt2.detach().cpu())

            F = None
            mask = []
                        
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]            
            
            return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}
    else:    
        def run(self, **args):  
            pt1 = args['pt1']
            pt2 = args['pt2']
            Hs = args['Hs']
        
            if torch.is_tensor(pt1):
                pt1 = np.ascontiguousarray(pt1.detach().cpu())
                pt2 = np.ascontiguousarray(pt2.detach().cpu())

            F = None
            mask = []
            
            if self.mode == 'fundamental_matrix':           
                if (pt1.shape)[0] > 7:                        
                    F, mask = pydegensac.findFundamentalMatrix(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
        
                pt1 = args['pt1'][mask]
                pt2 = args['pt2'][mask]     
                Hs = args['Hs'][mask]
            else:
                if (pt1.shape)[0] > 3:                        
                    F, mask = pydegensac.findHomography(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
                
                pt1 = args['pt1'][mask]
                pt2 = args['pt2'][mask]     
                Hs = args['Hs'][mask]            
            
            return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}


class magsac_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
        self.mode = 'fundamental_matrix'
                              
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('opencv_magsac_' + self.mode + '_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters)).lower()

        
    def run(self, **args):  
        pt1 = args['pt1']
        pt2 = args['pt2']
        Hs = args['Hs']
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.mode == 'fundamental_matrix':           
            if (pt1.shape)[0] > 7:  
                try:                     
                    F, mask = cv2.findFundamentalMat(pt1, pt2, cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                except:
                    try:
                        idx = np.random.permutation(pt1.shape[0])
                        jdx = np.argsort(idx)
                        F, mask = cv2.findFundamentalMat(pt1[idx], pt2[idx], cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                        mask = mask[jdx]
                    except:
                        F, mask = pydegensac.findFundamentalMatrix(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)
                        
            if not isinstance(mask, np.ndarray):
                mask = []
            else:
                if len(mask.shape) > 1: mask = mask.squeeze(1) > 0
        
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
        else:
            if (pt1.shape)[0] > 3:    
                try:                    
                    F, mask = cv2.findHomography(pt1, pt2, cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                except:
                    try:
                        idx = np.random.permutation(pt1.shape[0])
                        jdx = np.argsort(idx)
                        F, mask = cv2.findHomography(pt1[idx], pt2[idx], cv2.USAC_MAGSAC, self.px_th, self.conf, self.max_iters)
                        mask = mask[jdx]
                    except:                    
                        F, mask = pydegensac.findHomography(pt1, pt2, px_th=self.px_th, conf=self.conf, max_iters=self.max_iters)

            if not isinstance(mask, np.ndarray):
                mask = []
            else:
                if len(mask.shape) > 1: mask = mask.squeeze(1) > 0
                
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]            
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}


class poselib_module:
    def __init__(self, **args):
        self.px_th = 3
        self.conf = 0.9999
        self.max_iters = 100000
        self.min_iters = 1000                    
        self.mode = 'fundamental_matrix'
          
        for k, v in args.items():
           setattr(self, k, v)
       
        
    def get_id(self):
        return ('poselib_' + self.mode + '_th_' + str(self.px_th) + '_conf_' + str(self.conf) + '_max_iters_' + str(self.max_iters) + '_min_iters_' + str(self.max_iters)).lower()

        
    def run(self, **args):  
        pt1 = args['pt1']
        pt2 = args['pt2']
        Hs = args['Hs']
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
        
        params = {         
            'max_iterations' : self.max_iters,
            'min_iterations' : self.min_iters,
            'success_prob' : self.conf,
            'max_epipolar_error' : self.px_th
            }
            
        if self.mode == 'fundamental_matrix':           
            if (pt1.shape)[0] > 7:  

                F, info = poselib.estimate_fundamental(pt1, pt2, params, {})
                mask = info['inliers']

            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]
        else:
            if (pt1.shape)[0] > 3:                        
                F, info = poselib.estimate_homography(pt1, pt2, params, {})
                mask = info['inliers']
                
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]     
            Hs = args['Hs'][mask]            
            
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'F': F, 'mask': mask}


class sift_module:
    def __init__(self, **args):
        self.upright = False
        self.th = 0.99
        self.num_features = 8000
        self.rootsift = True
        
        for k, v in args.items():
           setattr(self, k, v)

        with torch.inference_mode():
            self.detector = cv2.SIFT_create(self.num_features, contrastThreshold=-10000, edgeThreshold=10000)


    def get_id(self):
        return ('sift_upright_' + str(self.upright) + '_rootsift_' + str(self.rootsift) + '_th_' + str(self.th) + '_nfeat_' + str(self.num_features)).lower()


    def run(self, **args):    
        
        im1 = cv2.imread(args['im1'], cv2.IMREAD_GRAYSCALE)
        kps1 = self.detector.detect(im1, None)

        if self.upright:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kps1]), axis=0, return_index=True)[1]
            kps1 = [kps1[ii] for ii in idx]
            for ii in range(len(kps1)):
                kps1[ii].angle = 0           
        kps1, descs1 = self.detector.compute(im1, kps1)

        if self.rootsift:
            descs1 /= descs1.sum(axis=1, keepdims=True) + 1e-8
            descs1 = np.sqrt(descs1)

        im2 = cv2.imread(args['im2'], cv2.IMREAD_GRAYSCALE)
        kps2 = self.detector.detect(im2, None)

        if self.upright:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kps2]), axis=0, return_index=True)[1]
            kps2 = [kps2[ii] for ii in idx]
            for ii in range(len(kps2)):
                kps2[ii].angle = 0           
        kps2, descs2 = self.detector.compute(im2, kps2)

        if self.rootsift:
            descs2 /= descs2.sum(axis=1, keepdims=True) + 1e-8
            descs2 = np.sqrt(descs2)

        with torch.inference_mode():            
            val, idxs = K.feature.match_smnn(torch.from_numpy(descs1).cuda(), torch.from_numpy(descs2).cuda(), self.th)

        pt1 = None
        pt2 = None
        kps1 = laf_from_opencv_kpts(kps1, device=device)
        kps2 = laf_from_opencv_kpts(kps2, device=device)
                
        kps1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        kps2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)

        pt1, pt2, Hs_laf = refinement_laf(None, None, data1=kps1, data2=kps2, img_patches=False)    

        return {'pt1': pt1, 'pt2': pt2, 'kp1': kps1, 'kp2': kps2, 'Hs': Hs_laf, 'val': val}


# import matplotlib.pyplot as plt
class lightglue_module:
    def __init__(self, **args):
        self.upright = True
        self.num_features = 8000
        self.what = 'superpoint'
        self.resize = 1024 # this is default, set to None to disable
        self.aliked_model = "aliked-n16rot" # default is "aliked-n16"
                
        for k, v in args.items():
           setattr(self, k, v)

        with torch.inference_mode():
            if self.what == 'disk':            
                self.extractor = lg_disk(max_num_keypoints=self.num_features).eval().to(device)
                self.matcher = lg_lightglue(features='disk').eval().to(device)            
            elif self.what == 'aliked':            
                self.extractor = lg_aliked(max_num_keypoints=self.num_features, model_name=self.aliked_model).eval().to(device)
                self.matcher = lg_lightglue(features='aliked').eval().to(device)            
            elif self.what == 'sift':            
                self.extractor = lg_sift(max_num_keypoints=self.num_features).eval().to(device)
                self.matcher = lg_lightglue(features='sift').eval().to(device)                            
            elif self.what == 'doghardnet':            
                self.extractor = lg_doghardnet(max_num_keypoints=self.num_features).eval().to(device)
                self.matcher = lg_lightglue(features='doghardnet').eval().to(device)            
            else:   
                self.what = 'superpoint'
                self.extractor = lg_superpoint(max_num_keypoints=self.num_features).eval().to(device)
                self.matcher = lg_lightglue(features='superpoint').eval().to(device)            


    def get_id(self):
        if self.what == 'aliked':
            plus_str = "_model_" + self.aliked_model
        else:
            plus_str = ''
        return (self.what + '_lightglue_upright_' + str(self.upright) + '_nfeat_' + str(self.num_features) + '_resize_' + str(self.resize) + plus_str).lower()


    def run(self, **args):           
        # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
        img1 = lg_load_image(args['im1']).to(device)
        img2 = lg_load_image(args['im2']).to(device)
        
        # img2 = (img2.flip(1).permute(0, 2, 1).permute(1,2,0)[:, :, [2, 1, 0]] * 255).type(torch.uint8).detach().cpu().numpy()
        # cv2.imwrite('rot_test.png', img2)
        # img2 = lg_load_image('rot_test.png').to(device)

        with torch.inference_mode():
            feats1 = self.extractor.extract(img1, resize=self.resize)
            feats2 = self.extractor.extract(img2, resize=self.resize)
            matches12 = self.matcher({'image0': feats1, 'image1': feats2})
            feats1_, feats2_, matches12 = [lg_rbd(x) for x in [feats1, feats2, matches12]]
            idxs = matches12['matches']        
            kps1 = feats1_['keypoints']
            kps2 = feats2_['keypoints']
        
        hw2 = torch.tensor(img2.shape[1:], device=device)
        
        if not self.upright:
            hw2_orig = hw2

            r_best = 0
            hw2_best = hw2
            kps2_best = kps2            
            idxs_best = idxs

            for r in range(1, 4):
                img2 = img2.flip(1).permute(0, 2, 1)
                hw2 = torch.tensor(img2.shape[1:], device=device)

                with torch.inference_mode():
                    feats2 = self.extractor.extract(img2, resize=self.resize)
                    matches12 = self.matcher({'image0': feats1, 'image1': feats2})
                    feats1_, feats2_, matches12 = [lg_rbd(x) for x in [feats1, feats2, matches12]]
                    idxs = matches12['matches']
                    kps2 = feats2_['keypoints']
            
                if idxs.shape[0] > idxs_best.shape[0]:
                    idxs_best = idxs
                    r_best = r
                    hw2_best = hw2
                    kps2_best = kps2

            a = -r_best / 2.0 * np.pi
            R = torch.tensor([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], device=device)
            kps2 = ((R @ (kps2_best.permute(1, 0) - (torch.tensor([[hw2_best[1]], [hw2_best[0]]], device=device) / 2) ).type(torch.double)) + (torch.tensor([[hw2_orig[1]], [hw2_orig[0]]], device=device) / 2).type(torch.double)).permute(1, 0)
            idxs = idxs_best

            # plt.figure()
            # plt.axis('off')
            # img = cv2.imread('rot_test.png', cv2.IMREAD_GRAYSCALE)
            # plt.imshow(img)
            # plt.plot(kps2[:, 0].detach().cpu().numpy(), kps2[:, 1].detach().cpu().numpy(), linestyle='', color='red', marker='.')
    
        pt1 = kps1.squeeze().detach()[idxs[:, 0]].to(device)
        pt2 = kps2.squeeze().detach()[idxs[:, 1]].to(device)
        
        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=pt1, pt2=pt2, img_patches=False) # No refinement LAF!!!
        
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_laf}
    

# import matplotlib.pyplot as plt
class loftr_module:
    def __init__(self, **args):
        self.outdoor = True
        self.upright = True
        self.resize = None
        # self.resize = [800, 600]

        for k, v in args.items():
           setattr(self, k, v)

        if self.outdoor == True:
            pretrained = 'outdoor'
        else:
            pretrained = 'indoor_new'

        with torch.inference_mode():
            self.matcher = KF.LoFTR(pretrained=pretrained).to(device).eval()


    def get_id(self):
        return ('loftr_outdoor_' + str(self.outdoor) + '_upright_' + str(self.upright)).lower()


    def run(self, **args):
        image0 = K.io.load_image(args['im1'], K.io.ImageLoadType.GRAY32, device=device)
        image1 = K.io.load_image(args['im2'], K.io.ImageLoadType.GRAY32, device=device)

        # image1 = (image1 * 255).flip(1).permute(0, 2, 1).squeeze().type(torch.uint8).detach().cpu().numpy()
        # cv2.imwrite('rot_test.png', image1)
        # image1 = K.io.load_image('rot_test.png', K.io.ImageLoadType.GRAY32, device=device)

        hw1 = image0.shape[1:]
        hw2 = image1.shape[1:]

        if not (self.resize is None):        
            ms = min(self.resize)
            Ms = max(self.resize)

            if hw1[0] > hw1[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw1[0]) / hw1[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw1[1]) / hw1[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            K.geometry.resize(image0, (sz[0], sz[1]), antialias=True)

            if hw2[0] > hw2[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw2[0]) / hw2[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw2[1]) / hw2[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            K.geometry.resize(image1, (sz[0], sz[1]), antialias=True)
                    
        hw1_ = image0.shape[1:]
        hw2_ = image1.shape[1:]

        input_dict = {
            "image0": image0.unsqueeze(0),  # LofTR works on grayscale images only
            "image1": image1.unsqueeze(0),
        }

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        kps1 = correspondences["keypoints0"]
        kps2 = correspondences["keypoints1"]
        
        if not self.upright:
            hw2_orig = hw2_

            r_best = 0
            hw2_best = hw2_
            kps1_best = kps1            
            kps2_best = kps2            

            for r in range(1, 4):
                image1 = image1.flip(1).permute(0, 2, 1)
                hw2__ = torch.tensor(image1.shape[1:], device=device)

                input_dict = {
                    "image0": image0.unsqueeze(0),  # LofTR works on grayscale images only
                    "image1": image1.unsqueeze(0),
                }
        
                with torch.inference_mode():
                    correspondences = self.matcher(input_dict)
        
                kps1 = correspondences["keypoints0"]
                kps2 = correspondences["keypoints1"]
            
                if kps1.shape[0] > kps1_best.shape[0]:
                    r_best = r
                    hw2_best = hw2__
                    kps1_best = kps1
                    kps2_best = kps2

            a = -r_best / 2.0 * np.pi
            R = torch.tensor([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], device=device)
            kps2 = ((R @ (kps2_best.permute(1, 0) - (torch.tensor([[hw2_best[1]], [hw2_best[0]]], device=device) / 2) ).type(torch.double)) + (torch.tensor([[hw2_orig[1]], [hw2_orig[0]]], device=device) / 2).type(torch.double)).permute(1, 0)
        
        kps1 = kps1.squeeze().detach().to(device).clone()
        kps2 = kps2.squeeze().detach().to(device).clone()

        kps1[:, 0] = kps1[:, 0] * (hw1[1] / float(hw1_[1]))
        kps1[:, 1] = kps1[:, 1] * (hw1[0] / float(hw1_[0]))
    
        kps2[:, 0] = kps2[:, 0] * (hw2[1] / float(hw2_[1]))
        kps2[:, 1] = kps2[:, 1] * (hw2[0] / float(hw2_[0]))
        
        # plt.figure()
        # plt.axis('off')
        # img = cv2.imread('rot_test.png', cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img)
        # plt.plot(kps2[:, 0].detach().cpu().numpy(), kps2[:, 1].detach().cpu().numpy(), linestyle='', color='red', marker='.')
        
        pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False)
        # pt1, pt2, Hs_laf = refinement_laf(None, None, pt1=kps1, pt2=kps2, img_patches=False) # No refinement LAF!!!

        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs_laf}
