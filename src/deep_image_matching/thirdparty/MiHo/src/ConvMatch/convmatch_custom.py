import numpy as np
import torch
import cv2
import sys
import os
import gdown
import zipfile
from PIL import Image
from .config_test import get_config
import uuid
import shutil

torch.set_grad_enabled(False)
sys.path.append(os.path.join(os.path.split(__file__)[0], 'core'))

from convmatch import ConvMatch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def norm_kp(cx, cy, fx, fy, kp):
    # New kp
    kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
    return kp


def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = torch.from_numpy(desc_ii), torch.from_numpy(desc_jj)
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:,0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2= nnIdx2.squeeze()
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(0, nnIdx1.shape[0]).long()).numpy()
    ratio_test = (distVals[:,0] / distVals[:,1].clamp(min=1e-10)).numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.numpy()]
    return idx_sort, ratio_test, mutual_nearest


def draw_matching(img1, img2, pt1, pt2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1,h2), w1+w2, 3), np.uint8)
    vis[:h1, :w1] = img1

    vis[:h2, w1:w1+w2] = img2

    green = (0, 255, 0)
    thickness = 1

    for i in range(pt1.shape[0]):
        x1 = int(pt1[i, 0])
        y1 = int(pt1[i, 1])
        x2 = int(pt2[i, 0] + w1)
        y2 = int(pt2[i, 1])

        cv2.line(vis, (x1, y1), (x2, y2), green, int(thickness))
    return vis


class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)
        self.num_kp = num_kp
    def run(self, img):
        img = img.astype(np.uint8)
    #   img = cv2.imread(img)
        cv_kp, desc = self.sift.detectAndCompute(img, None)

        kp = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp]) # N*2

        return kp[:self.num_kp], desc[:self.num_kp]


def demo(opt, img1_path, img2_path):
    print("=======> Loading images")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    print("=======> Generating initial matching")
    SIFT = ExtractSIFT(num_kp=2000)

    kpts1, desc1 = SIFT.run(img1)
    kpts2, desc2 = SIFT.run(img2)

    idx_sort, ratio_test, mutual_nearest = computeNN(desc1, desc2)

    kpts2 = kpts2[idx_sort[1],:]

    cx1 = (img1.shape[1] - 1.0) * 0.5
    cy1 = (img1.shape[0] - 1.0) * 0.5
    f1 = max(img1.shape[1] - 1.0, img1.shape[0] - 1.0)

    cx2 = (img2.shape[1] - 1.0) * 0.5
    cy2 = (img2.shape[0] - 1.0) * 0.5
    f2 = max(img2.shape[1] - 1.0, img2.shape[0] - 1.0)

    kpts1_n = norm_kp(cx1, cy1, f1, f1, kpts1)
    kpts2_n = norm_kp(cx2, cy2, f2, f2, kpts2)

    xs = np.concatenate([kpts1_n, kpts2_n], axis=-1)
    ys = np.ones(xs.shape[0])

    print("=======> Loading pretrained model")
    model = ConvMatch(opt)
    checkpoint = torch.load('../pretrained-model/yfcc100m/model_best.pth', map_location=torch.device('cuda'))

    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    xs = torch.from_numpy(xs).float().cuda()

    print("=======> Pruning")
    data = {}
    data['xs'] = xs.unsqueeze(0).unsqueeze(1)
    y_hat, e_hat = model(data)
    y = y_hat[-1][0, :].cpu().numpy()  
    matching = draw_matching(img1, img2, kpts1[y > opt.inlier_threshold], kpts2[y > opt.inlier_threshold])
    cv2.imwrite('./inliers.jpg', matching)


class convmatch_module:
    current_net = None
    current_obj_id = None
    
    def __init__(self, **args):
        convmatch_dir = os.path.split(__file__)[0]
        model_dir = os.path.join(convmatch_dir, 'convmatch_models')

        file_to_download = os.path.join(convmatch_dir, 'convmatch_weights.zip')    
        if not os.path.isfile(file_to_download):    
            url = "https://drive.google.com/file/d/11TZKe2_VnEEUp5lUCBGCOPSSuzMKoCVG/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)        

        file_to_unzip = file_to_download
        if not os.path.isdir(model_dir):    
            with zipfile.ZipFile(file_to_unzip,"r") as zip_ref:
                zip_ref.extractall(path=convmatch_dir)
                shutil.move(os.path.join(convmatch_dir, 'models'), model_dir) 

        self.outdoor = True
        self.prev_outdoor = True

        self.opt, unparsed = get_config()    
        self.model = ConvMatch(self.opt)
        self.inlier_threshold = self.opt.inlier_threshold

        for k, v in args.items():
           setattr(self, k, v)

        if self.outdoor:
            model_path = os.path.join(convmatch_dir, 'convmatch_models', 'yfcc100m', 'model_best.pth')
        else:
            model_path = os.path.join(convmatch_dir, 'convmatch_models', 'sun3d', 'model_best.pth')
 
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        self.model.eval()
        
        self.convmatch_id = uuid.uuid4()
                           
        
    def get_id(self):
        return ('convmatch_outdoor_' + str(self.outdoor) + '_inlier_threshold_' + str(self.inlier_threshold)).lower()


    def run(self, **args):
        
        force_reload = False
        if (self.outdoor != self.prev_outdoor):
            force_reload = True
            self.prev_outdoor = self.outdoor

            convmatch_dir = os.path.split(__file__)[0]     
            if self.outdoor:
                model_path = os.path.join(convmatch_dir, 'convmatch_models', 'yfcc100m', 'model_best.pth')
            else:
                model_path = os.path.join(convmatch_dir, 'convmatch_models', 'sun3d', 'model_best.pth')

        if (convmatch_module.current_obj_id != self.convmatch_id) or force_reload:
            if not (convmatch_module.current_obj_id is None):
                checkpoint = torch.load(model_path, map_location=torch.device(device))
                self.model.load_state_dict(checkpoint['state_dict'])
                self.model.cuda()
                self.model.eval()      
                
        sz1 = Image.open(args['im1']).size
        sz2 = Image.open(args['im2']).size              
        
        pt1 = np.ascontiguousarray(args['pt1'].detach().cpu())
        pt2 = np.ascontiguousarray(args['pt2'].detach().cpu())
                
        l = pt1.shape[0]
        
        if l > 1:                
            cx1 = (sz1[1] - 1.0) * 0.5
            cy1 = (sz1[0] - 1.0) * 0.5
            f1 = max(sz1[1] - 1.0, sz1[0] - 1.0)
            
            cx2 = (sz2[1] - 1.0) * 0.5
            cy2 = (sz2[0] - 1.0) * 0.5
            f2 = max(sz2[1] - 1.0, sz2[0] - 1.0)

            kpts1_n = norm_kp(cx1, cy1, f1, f1, pt1)
            kpts2_n = norm_kp(cx2, cy2, f2, f2, pt2)

            xs = np.concatenate([kpts1_n, kpts2_n], axis=-1)
            
            data = {}
            data['xs'] = torch.tensor(xs, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(1)
            y_hat, e_hat = self.model(data)
            y = y_hat[-1][0, :].cpu().numpy()

            mask = y > self.inlier_threshold

            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]            
            Hs = args['Hs'][mask]            
        else:
            pt1 = args['pt1']
            pt2 = args['pt2']           
            Hs = args['Hs']   
            mask = []
                        
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
