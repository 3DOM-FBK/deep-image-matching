import torch
import cv2
import os
import gdown
from .fcgnn import GNN as fcgnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class fcgnn_module:    
    def __init__(self, **args):
        fcgnn_dir = os.path.split(__file__)[0]
        model_dir = os.path.join(fcgnn_dir, 'weights')
        os.makedirs(model_dir, exist_ok=True)        
        self.pad = 1

        file_to_download = os.path.join(model_dir, 'fcgnn.model')    
        if not os.path.isfile(file_to_download):    
            url = "https://drive.google.com/file/d/1IrZIrKyNTzKffw1hz4atrzGdFdOUl8nU/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)        

        for k, v in args.items():
           setattr(self, k, v)

        self.fcgnn_refiner = fcgnn().to(device)
        
                                   
    def get_id(self):
        return ('fcgnn').lower()


    def run(self, **args):      
        img1 = cv2.imread(args['im1'], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args['im2'], cv2.IMREAD_GRAYSCALE)
        
        img1_ = torch.tensor(img1.astype('float32') / 255.)[None, None].to(device)
        img2_ = torch.tensor(img2.astype('float32') / 255.)[None, None].to(device)
        
        pt1 = args['pt1']
        pt2 = args['pt2']        
                
        l = pt1.shape[0]
                        
        if l > 1:            
            mask = torch.zeros(l, dtype=torch.bool, device=device)

            to_take =  ~(((pt1 < self.pad) | (pt2 < self.pad) | (pt1 > torch.tensor([img1.shape[1], img1.shape[0]], device=device) - self.pad) | (pt2 > torch.tensor([img2.shape[1], img2.shape[0]], device=device) - self.pad)).any(dim=1))
            
            matches = torch.hstack((pt1, pt2))
            matches_refined, mask_ = self.fcgnn_refiner.optimize_matches_custom(img1_, img2_, matches[to_take], thd=0.999, min_matches=10)
                  
            pt1[to_take] = matches_refined[:,:2]
            pt2[to_take] = matches_refined[:,2:]
            
            mask[to_take] = mask_
            
            pt1 = args['pt1'][mask]
            pt2 = args['pt2'][mask]            
            Hs = args['Hs'][mask]            
        else:
            pt1 = args['pt1']
            pt2 = args['pt2']           
            Hs = args['Hs']   
            mask = []
                        
        return {'pt1': pt1, 'pt2': pt2, 'Hs': Hs, 'mask': mask}
