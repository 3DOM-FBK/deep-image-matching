"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
    training script
"""

import argparse
import os
import time
import sys
sys.path.append(os.path.dirname(__file__))

def parse_arguments():
    parser = argparse.ArgumentParser(description="LiftFeat training script.")
    parser.add_argument('--name',type=str,default='LiftFeat',help='set process name')
    
    # MegaDepth dataset setting
    parser.add_argument('--use_megadepth',action='store_true')
    parser.add_argument('--megadepth_root_path', type=str,
                        default='/home/yepeng_liu/code_python/dataset/MegaDepth/phoenix/S6/zl548',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--megadepth_batch_size', type=int, default=6)
    
    # COCO20k dataset setting
    parser.add_argument('--use_coco',action='store_true')
    parser.add_argument('--coco_root_path', type=str, default='/home/yepeng_liu/code_python/dataset/coco_20k',
                        help='Path to the COCO20k dataset root directory.')
    parser.add_argument('--coco_batch_size',type=int,default=4)
    
    parser.add_argument('--ckpt_save_path', type=str, default='/home/yepeng_liu/code_python/LiftFeat/trained_weights/test',
                        help='Path to save the checkpoints.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--use_coord_loss',action='store_true',help='Enable coordinate loss')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import numpy as np
import tqdm
import glob

from models.model import LiftFeatSPModel
from loss.loss import LiftFeatLoss
from utils.config import featureboost_config
from models.interpolator import InterpolateSparse2d
from utils.depth_anything_wrapper import DepthAnythingExtractor
from utils.alike_wrapper import ALikeExtractor

from dataset import megadepth_wrapper
from dataset import coco_wrapper
from dataset.megadepth import MegaDepthDataset
from dataset.coco_augmentor import COCOAugmentor

import setproctitle


class Trainer():
    def __init__(self, megadepth_root_path,use_megadepth,megadepth_batch_size,
                       coco_root_path,use_coco,coco_batch_size,
                       ckpt_save_path, 
                       model_name = 'LiftFeat',
                       n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500, use_coord_loss = False):
        print(f'MegeDepth: {use_megadepth}-{megadepth_batch_size}')
        print(f'COCO20k: {use_coco}-{coco_batch_size}')
        print(f'Coordinate loss: {use_coord_loss}')
        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # training model
        self.net = LiftFeatSPModel(featureboost_config, use_kenc=False, use_normal=True, use_cross=True).to(self.dev)
        self.loss_fn=LiftFeatLoss(self.dev,lam_descs=1,lam_kpts=2,lam_heatmap=1)
        
        # depth-anything model
        self.depth_net=DepthAnythingExtractor('vits',self.dev,256)
        
        # alike model
        self.alike_net=ALikeExtractor('alike-t',self.dev)

        #Setup optimizer 
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10_000, gamma=gamma_steplr)

        ##################### COCO INIT ##########################
        self.use_coco=use_coco
        self.coco_batch_size=coco_batch_size
        if self.use_coco:
            self.augmentor=COCOAugmentor(
                img_dir=coco_root_path,
                device=self.dev,load_dataset=True,
                batch_size=self.coco_batch_size,
                out_resolution=training_res,
                warp_resolution=training_res,
                sides_crop=0.1,
                max_num_imgs=3000,
                num_test_imgs=5,
                photometric=True,
                geometric=True,
                reload_step=4000
            )
        ##################### COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        self.use_megadepth=use_megadepth
        self.megadepth_batch_size=megadepth_batch_size
        if self.use_megadepth:
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            megadepth_dataset = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )

            self.megadepth_dataloader = DataLoader(megadepth_dataset, batch_size=megadepth_batch_size, shuffle=True)
            self.megadepth_data_iter = iter(self.megadepth_dataloader)
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name
        self.use_coord_loss = use_coord_loss
        
        
    def generate_train_data(self):
        imgs1_t,imgs2_t=[],[]
        imgs1_np,imgs2_np=[],[]
        # norms0,norms1=[],[]
        positives_coarse=[]
        
        if self.use_coco:
                coco_imgs1, coco_imgs2, H1, H2 = coco_wrapper.make_batch(self.augmentor, 0.1)
                h_coarse, w_coarse = coco_imgs1[0].shape[-2] // 8, coco_imgs1[0].shape[-1] // 8
                _ , positives_coco_coarse = coco_wrapper.get_corresponding_pts(coco_imgs1, coco_imgs2, H1, H2, self.augmentor, h_coarse, w_coarse)
                coco_imgs1=coco_imgs1.mean(1,keepdim=True);coco_imgs2=coco_imgs2.mean(1,keepdim=True)
                imgs1_t.append(coco_imgs1);imgs2_t.append(coco_imgs2)
                positives_coarse += positives_coco_coarse
                    
        if self.use_megadepth:
            try:
                megadepth_data=next(self.megadepth_data_iter)
            except StopIteration:
                print('End of MD DATASET')
                self.megadepth_data_iter=iter(self.megadepth_dataloader)
                megadepth_data=next(self.megadepth_data_iter)
            if megadepth_data is not None:
                for k in megadepth_data.keys():
                    if isinstance(megadepth_data[k],torch.Tensor):
                        megadepth_data[k]=megadepth_data[k].to(self.dev)
                megadepth_imgs1_t,megadepth_imgs2_t=megadepth_data['image0'],megadepth_data['image1']
                megadepth_imgs1_t=megadepth_imgs1_t.mean(1,keepdim=True);megadepth_imgs2_t=megadepth_imgs2_t.mean(1,keepdim=True)
                imgs1_t.append(megadepth_imgs1_t);imgs2_t.append(megadepth_imgs2_t)
                megadepth_imgs1_np,megadepth_imgs2_np=megadepth_data['image0_np'],megadepth_data['image1_np']
                for np_idx in range(megadepth_imgs1_np.shape[0]):
                    img1_np,img2_np=megadepth_imgs1_np[np_idx].squeeze(0).cpu().numpy(),megadepth_imgs2_np[np_idx].squeeze(0).cpu().numpy()
                    imgs1_np.append(img1_np);imgs2_np.append(img2_np)
                positives_megadepth_coarse=megadepth_wrapper.spvs_coarse(megadepth_data,8)
                positives_coarse += positives_megadepth_coarse
                
        with torch.no_grad():
            imgs1_t=torch.cat(imgs1_t,dim=0)
            imgs2_t=torch.cat(imgs2_t,dim=0)
            
        return imgs1_t,imgs2_t,imgs1_np,imgs2_np,positives_coarse


    def train(self):
        self.net.train()

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                # import pdb;pdb.set_trace()
                imgs1_t,imgs2_t,imgs1_np,imgs2_np,positives_coarse=self.generate_train_data()

                #Check if batch is corrupted with too few correspondences
                is_corrupted = False
                for p in positives_coarse:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                # import pdb;pdb.set_trace()
                #Forward pass
                # start=time.perf_counter()
                feats1,kpts1,normals1 = self.net.forward1(imgs1_t)
                feats2,kpts2,normals2 = self.net.forward1(imgs2_t)
                
                coordinates,fb_coordinates=[],[]
                alike_kpts1,alike_kpts2=[],[]
                DA_normals1,DA_normals2=[],[]
                
                # import pdb;pdb.set_trace()
                
                fb_feats1,fb_feats2=[],[]
                for b in range(feats1.shape[0]):
                    feat1=feats1[b].permute(1,2,0).reshape(-1,feats1.shape[1])
                    feat2=feats2[b].permute(1,2,0).reshape(-1,feats2.shape[1])
                    
                    coordinate=self.net.fine_matcher(torch.cat([feat1,feat2],dim=-1))
                    coordinates.append(coordinate)
                    
                    fb_feat1=self.net.forward2(feats1[b].unsqueeze(0),kpts1[b].unsqueeze(0),normals1[b].unsqueeze(0))
                    fb_feat2=self.net.forward2(feats2[b].unsqueeze(0),kpts2[b].unsqueeze(0),normals2[b].unsqueeze(0))
                    
                    fb_coordinate=self.net.fine_matcher(torch.cat([fb_feat1,fb_feat2],dim=-1))
                    fb_coordinates.append(fb_coordinate)
                    
                    fb_feats1.append(fb_feat1.unsqueeze(0));fb_feats2.append(fb_feat2.unsqueeze(0))
                    
                    img1,img2=imgs1_t[b],imgs2_t[b]
                    img1=img1.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255
                    img2=img2.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255
                    alike_kpt1=torch.tensor(self.alike_net.extract_alike_kpts(img1),device=self.dev)
                    alike_kpt2=torch.tensor(self.alike_net.extract_alike_kpts(img2),device=self.dev)
                    alike_kpts1.append(alike_kpt1);alike_kpts2.append(alike_kpt2)
                
                # import pdb;pdb.set_trace()
                for b in range(len(imgs1_np)):
                    megadepth_depth1,megadepth_norm1=self.depth_net.extract(imgs1_np[b])
                    megadepth_depth2,megadepth_norm2=self.depth_net.extract(imgs2_np[b])
                    DA_normals1.append(megadepth_norm1);DA_normals2.append(megadepth_norm2)
                    
                # import pdb;pdb.set_trace()
                fb_feats1=torch.cat(fb_feats1,dim=0)
                fb_feats2=torch.cat(fb_feats2,dim=0)
                fb_feats1=fb_feats1.reshape(feats1.shape[0],feats1.shape[2],feats1.shape[3],-1).permute(0,3,1,2)
                fb_feats2=fb_feats2.reshape(feats2.shape[0],feats2.shape[2],feats2.shape[3],-1).permute(0,3,1,2)
                
                coordinates=torch.cat(coordinates,dim=0)
                coordinates=coordinates.reshape(feats1.shape[0],feats1.shape[2],feats1.shape[3],-1).permute(0,3,1,2)
                
                fb_coordinates=torch.cat(fb_coordinates,dim=0)
                fb_coordinates=fb_coordinates.reshape(feats1.shape[0],feats1.shape[2],feats1.shape[3],-1).permute(0,3,1,2)
                
                # end=time.perf_counter()
                # print(f"forward1 cost {end-start} seconds")

                loss_items = []

                # import pdb;pdb.set_trace()
                loss_info=self.loss_fn(
                    feats1,fb_feats1,kpts1,normals1,
                    feats2,fb_feats2,kpts2,normals2,
                    positives_coarse,
                    coordinates,fb_coordinates,
                    alike_kpts1,alike_kpts2,
                    DA_normals1,DA_normals2,
                    self.megadepth_batch_size,self.coco_batch_size)
                
                loss_descs,acc_coarse=loss_info['loss_descs'],loss_info['acc_coarse']
                loss_coordinates,acc_coordinates=loss_info['loss_coordinates'],loss_info['acc_coordinates']
                loss_fb_descs,acc_fb_coarse=loss_info['loss_fb_descs'],loss_info['acc_fb_coarse']
                loss_fb_coordinates,acc_fb_coordinates=loss_info['loss_fb_coordinates'],loss_info['acc_fb_coordinates']
                loss_kpts,acc_kpt=loss_info['loss_kpts'],loss_info['acc_kpt']
                loss_normals=loss_info['loss_normals']
                
                loss_items.append(loss_fb_descs.unsqueeze(0))
                loss_items.append(loss_kpts.unsqueeze(0))
                loss_items.append(loss_normals.unsqueeze(0))
                
                if self.use_coord_loss:
                    loss_items.append(loss_fb_coordinates.unsqueeze(0))

                # nb_coarse = len(m1)
                # nb_coarse = len(fb_m1)
                loss = torch.cat(loss_items, -1).mean()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                # import pdb;pdb.set_trace()
                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')

                pbar.set_description(
'Loss: {:.4f} \
loss_descs: {:.3f} acc_coarse: {:.3f} \
loss_coordinates: {:.3f} acc_coordinates: {:.3f} \
loss_fb_descs: {:.3f} acc_fb_coarse: {:.3f} \
loss_fb_coordinates: {:.3f} acc_fb_coordinates: {:.3f} \
loss_kpts: {:.3f} acc_kpts: {:.3f} \
loss_normals: {:.3f}'.format( \
loss.item(), \
loss_descs.item(), acc_coarse, \
loss_coordinates.item(), acc_coordinates, \
loss_fb_descs.item(), acc_fb_coarse, \
loss_fb_coordinates.item(), acc_fb_coordinates, \
loss_kpts.item(), acc_kpt, \
loss_normals.item()) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/acc_coarse', acc_coarse, i)
                self.writer.add_scalar('Accuracy/acc_coordinates', acc_coordinates, i)
                self.writer.add_scalar('Accuracy/acc_fb_coarse', acc_fb_coarse, i)
                self.writer.add_scalar('Accuracy/acc_fb_coordinates', acc_fb_coordinates, i)
                self.writer.add_scalar('Loss/descs', loss_descs.item(), i)
                self.writer.add_scalar('Loss/coordinates', loss_coordinates.item(), i)
                self.writer.add_scalar('Loss/fb_descs', loss_fb_descs.item(), i)
                self.writer.add_scalar('Loss/fb_coordinates', loss_fb_coordinates.item(), i)
                self.writer.add_scalar('Loss/kpts', loss_kpts.item(), i)
                self.writer.add_scalar('Loss/normals', loss_normals.item(), i)



if __name__ == '__main__':
    
    setproctitle.setproctitle(args.name)

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path, 
        use_megadepth=args.use_megadepth,
        megadepth_batch_size=args.megadepth_batch_size,
        coco_root_path=args.coco_root_path, 
        use_coco=args.use_coco,
        coco_batch_size=args.coco_batch_size,
        ckpt_save_path=args.ckpt_save_path,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
        use_coord_loss=args.use_coord_loss
    )

    #The most fun part
    trainer.train()
