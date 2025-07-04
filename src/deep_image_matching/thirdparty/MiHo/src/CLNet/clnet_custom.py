import numpy as np
import torch
import os
import cv2
import gdown
import zipfile
from PIL import Image
from .model import CLNet
from .config import get_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)


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
    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    vis[:h1, :w1] = img1

    vis[h1:h1 + h2, :w2] = img2

    green = (0, 255, 0)
    thickness = 1

    for i in range(pt1.shape[0]):
        x1 = int(pt1[i, 0])
        y1 = int(pt1[i, 1])
        x2 = int(pt2[i, 0])
        y2 = int(pt2[i, 1] + h1)

        cv2.line(vis, (x1, y1), (x2, y2), green, int(thickness))
    return vis


class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)
        self.num_kp = num_kp
    def run(self, img):
        img = img.astype(np.uint8)
    #    img = cv2.imread(img)
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
    model = CLNet(opt)
    checkpoint = torch.load('../pretrained_models/clnet_yfcc_sift.pth', map_location=torch.device('cpu'))

    state_dict = {}
    for key in checkpoint['state_dict'].keys():
        key_new = key.split('module')[1][1:]
        state_dict[key_new] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict)
    model.eval()

    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()

    print("=======> Pruning")
    ws, _, e_hat, y_hat = model(xs[None, None], ys[None])

    w0 = ws[1].squeeze(0) ## weights for 1st pruning
    w1 = ws[3].squeeze(0) ## weights for 2nd pruning
    w2 = ws[4].squeeze(0) ## weights for picking up inliers from candidates

    w0 = torch.sort(w0, dim=-1, descending=True)[1][:1000].numpy().astype(np.int32)
    w1 = torch.sort(w1, dim=-1, descending=True)[1][:500].numpy().astype(np.int32)
    w2 = w2.numpy()

    print("=======> Done")
    ## init matching
    init_matching = draw_matching(img1, img2, kpts1, kpts2)
    cv2.imwrite('./init_matching.png', init_matching)

    ## 1st pruning
    kpts1 = kpts1[w0]
    kpts2 = kpts2[w0]
    matching = draw_matching(img1, img2, kpts1, kpts2)
    cv2.imwrite('./1st_prune_matching.png', matching)

    ## 2nd pruning
    kpts1 = kpts1[w1]
    kpts2 = kpts2[w1]
    matching = draw_matching(img1, img2, kpts1, kpts2)
    cv2.imwrite('./2nd_prune_matching.png', matching)

    ## picking up inliers from candidates
    matching = draw_matching(img1, img2, kpts1[w2 >= 0], kpts2[w2 >= 0])
    cv2.imwrite('./inliers.png', matching)


class clnet_module:    
    def __init__(self, **args):
        clnet_dir = os.path.split(__file__)[0]
        model_dir = os.path.join(clnet_dir, 'clnet_models')

        file_to_download = os.path.join(clnet_dir, 'clnet_weights.zip')    
        if not os.path.isfile(file_to_download):    
            url = "https://drive.google.com/file/d/1y-8xQ22byGeop3ZBbyxZ_2GVLoebxvk8/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)        

        file_to_unzip = file_to_download
        if not os.path.isdir(model_dir):    
            with zipfile.ZipFile(file_to_unzip,"r") as zip_ref:
                zip_ref.extractall(path=clnet_dir)
                # shutil.move(os.path.join(clnet_dir, 'models'), model_dir) 

        self.opt, unparsed = get_config()    
        self.model = CLNet(self.opt)
        checkpoint = torch.load(os.path.join(clnet_dir, 'CLNet_models/clnet_yfcc_sift.pth'), map_location=torch.device(device))
    
        state_dict = {}
        for key in checkpoint['state_dict'].keys():
            key_new = key.split('module')[1][1:]
            state_dict[key_new] = checkpoint['state_dict'][key]
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.inlier_threshold = 1e-4

        for k, v in args.items():
           setattr(self, k, v)


    def get_id(self):
        return ('clnet_inlier_threshold_' + str(self.inlier_threshold)).lower()


    def run(self, **args):
        sz1 = Image.open(args['im1']).size
        sz2 = Image.open(args['im2']).size              
        
        pt1 = np.ascontiguousarray(args['pt1'].detach().cpu())
        pt2 = np.ascontiguousarray(args['pt2'].detach().cpu())
                
        l = pt1.shape[0]
        
        if l > 11:                
            cx1 = (sz1[1] - 1.0) * 0.5
            cy1 = (sz1[0] - 1.0) * 0.5
            f1 = max(sz1[1] - 1.0, sz1[0] - 1.0)
            
            cx2 = (sz2[1] - 1.0) * 0.5
            cy2 = (sz2[0] - 1.0) * 0.5
            f2 = max(sz2[1] - 1.0, sz2[0] - 1.0)

            kpts1_n = norm_kp(cx1, cy1, f1, f1, pt1)
            kpts2_n = norm_kp(cx2, cy2, f2, f2, pt2)

            xs = np.concatenate([kpts1_n, kpts2_n], axis=-1)
            ys = np.ones(xs.shape[0])

            xs = torch.from_numpy(xs).float()
            ys = torch.from_numpy(ys).float()

            ws, _, e_hat, y_hat = self.model(xs[None, None], ys[None])
        
            mask = y_hat.squeeze(0) < self.inlier_threshold

            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]            
            Hs = args['Hs'][mask]            
        else:
            pt1 = args['pt1']
            pt2 = args['pt2']           
            Hs = args['Hs']   
            mask = []
                        
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
