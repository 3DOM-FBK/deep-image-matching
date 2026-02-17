import os
import cv2
import torch
import numpy as np
import yaml
import matplotlib.cm as cm
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.liftfeat_wrapper import LiftFeat,MODEL_PATH
from utils.post_process import match_features
os.environ['CUDA_VISIBLE_DEVICES']='0'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class VideoHandler:
    def __init__(self,video_path,size=[640,360]):
        self.video_path=video_path
        self.size=size
        self.cap=cv2.VideoCapture(video_path)
        
    def get_frame(self):
        ret,frame=self.cap.read()
        if ret==True:
            frame=cv2.resize(frame,(int(self.size[0]),int(self.size[1])))
        return ret,frame
    
def draw_video_match(img0,img1,kpts0,kpts1,mkpts0,mkpts1,match_scores,mask,max_match_num=512,margin=15):
    H0, W0, c = img0.shape
    H1, W1, c = img1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    
    # 构建画布，把两个图像先拼接到一起
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = img0
    out[:H1, W0+margin:, :] = img1
    #out = np.stack([out]*3, -1)

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    mkpts0_correct,mkpts1_correct=mkpts0[mask],mkpts1[mask]
    mkpts0_wrong,mkpts1_wrong=mkpts0[~mask],mkpts1[~mask]
    match_s=match_scores[mask]
    
    print(f"correct: {mkpts0_correct.shape[0]} wrong: {mkpts0_wrong.shape[0]}")
    
    if mkpts0_correct.shape[0] > max_match_num:
        # perm=np.random.randint(low=0,high=mkpts0_correct.shape[0],size=max_match_num)
        # mkpts0_show,mkpts1_show=mkpts0_correct[perm],mkpts1_correct[perm]
        mkpts0_show,mkpts1_show=mkpts0_correct,mkpts1_correct
    else:
        mkpts0_show,mkpts1_show=mkpts0_correct,mkpts1_correct
        
    # 普通的点
    vis_normal_point = True
    if (vis_normal_point):
        for x, y in mkpts0_show:
            cv2.circle(out, (x, y), 2, (47,132,250), -1, lineType=cv2.LINE_AA)
        for x, y in mkpts1_show:
            cv2.circle(out, (x + margin + W0, y), 2, (47,132,250), -1,lineType=cv2.LINE_AA)
    
    vis_match_line = True
    if (vis_match_line):
        for pt0, pt1,score in zip(mkpts0_show, mkpts1_show,match_s):
            color_cm = cm.jet(1.0 - score, alpha=0)  
            color = (int(color_cm[0] * 255), int(color_cm[1] * 255), int(color_cm[2] * 255))
            cv2.line(out, pt0, (W0 + margin + pt1[0], pt1[1]), color, 1)
            
    return out
    
def run_video_demo(std_img_path,video_path):
   
    
    liftfeat=LiftFeat(weight=MODEL_PATH,detect_threshold=0.15)
    
    std_img=cv2.imread(std_img_path)
    std_img=cv2.resize(std_img,(640,360))
    
    handler=VideoHandler(video_path)
    
    # 定义编解码器并创建VideoWriter对象
    if not os.path.exists('./output'):
        os.makedirs('./output')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    out = cv2.VideoWriter('./output/video_demo.mp4', fourcc, 20.0, (1300, 360))
    K=[[1084.8,0,640.24],[0,1085,354.87],[0,0,1]]
    K=np.array(K)
    data_std=liftfeat.extract(std_img)
    
    while True:
        ret,frame=handler.get_frame()
        if ret==False:
            break
       
        if frame is not None:
            data=liftfeat.extract(frame)
            idx0, idx1, match_scores=match_features(data_std["descriptors"],data["descriptors"],-1)
            mkpts0=data_std["keypoints"][idx0]
            mkpts1=data["keypoints"][idx1]
            mkpts0_np=mkpts0.cpu().numpy()
            mkpts1_np=mkpts1.cpu().numpy()
            match_scores_np=match_scores.detach().cpu().numpy()
            kpts0 = (mkpts0_np - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]
            kpts1 = (mkpts1_np - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]
            
            # normalize ransac threshold
            ransac_thr = 0.5 / np.mean([K[0, 0], K[1, 1], K[0, 0], K[1, 1]])
            
            if mkpts0_np.shape[0] < 5:
                print(f"mkpts size less then 5")
            else:
                # compute pose with cv2
                
                E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=0.999, method=cv2.RANSAC)
                if E is None:
                    print("\nE is None while trying to recover pose.\n")
                    continue
                match_mask=mask.squeeze(axis=1)>0
                show_kpts0,show_kpts1=mkpts0_np[match_mask],mkpts1_np[match_mask]
                show_match_scores=match_scores_np[match_mask]
                show_mask=np.ones(show_kpts0.shape[0])>0
                match_img=draw_video_match(std_img,frame,show_kpts0,show_kpts1,show_kpts0,show_kpts1,show_match_scores,show_mask,margin=20)
                kpts0_num,kpts1_num=data_std["keypoints"].shape[0],data["keypoints"].shape[0]
                cv2.putText(match_img,f"LiftFeat",(10,20),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,241))
                cv2.putText(match_img,f"Keypoints: {kpts0_num}:{kpts1_num}",(10,40),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255))
                cv2.putText(match_img,f"Matches: {show_kpts0.shape[0]}",(10,60),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255))
                out.write(match_img)
                
                
    out.release()
    


            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run LiftFeat video matching demo.")
    parser.add_argument('--img', type=str, required=True, help='Path to the template image')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    
    args = parser.parse_args()
    
    run_video_demo(args.img, args.video)
