import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def dual_softmax_loss(X, Y, temp = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf


class LiftFeatLoss(nn.Module):
    def __init__(self,device,lam_descs=1,lam_fb_descs=1,lam_kpts=1,lam_heatmap=1,lam_normals=1,lam_coordinates=1,lam_fb_coordinates=1,depth_spvs=False):
        super().__init__()
        
        # loss parameters
        self.lam_descs=lam_descs
        self.lam_fb_descs=lam_fb_descs
        self.lam_kpts=lam_kpts
        self.lam_heatmap=lam_heatmap
        self.lam_normals=lam_normals
        self.lam_coordinates=lam_coordinates
        self.lam_fb_coordinates=lam_fb_coordinates
        self.depth_spvs=depth_spvs
        self.running_descs_loss=0
        self.running_kpts_loss=0
        self.running_heatmaps_loss=0
        self.loss_descs=0
        self.loss_fb_descs=0
        self.loss_kpts=0
        self.loss_heatmaps=0
        self.loss_normals=0
        self.loss_coordinates=0
        self.loss_fb_coordinates=0
        self.acc_coarse=0
        self.acc_fb_coarse=0
        self.acc_kpt=0
        self.acc_coordinates=0
        self.acc_fb_coordinates=0
        
        # device
        self.dev=device
        
    
    def check_accuracy(self,m1,m2,pts1=None,pts2=None,plot=False):
        with torch.no_grad():
            #dist_mat = torch.cdist(X,Y)
            dist_mat = m1 @ m2.t()
            nn = torch.argmax(dist_mat, dim=1)
            #nn = torch.argmin(dist_mat, dim=1)
            correct = nn == torch.arange(len(m1), device = m1.device)

            if pts1 is not None and plot:
                import matplotlib.pyplot as plt
                canvas = torch.zeros((60, 80),device=m1.device)
                pts1 = pts1[~correct]
                canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
                canvas = canvas.cpu().numpy()
                plt.imshow(canvas), plt.show()

            acc = correct.sum().item() / len(m1)
            return acc    
    
    def compute_descriptors_loss(self,descs1,descs2,pts):
        loss=[]
        acc=0
        B,_,H,W=descs1.shape
        conf_list=[]
        
        for b in range(B):
            pts1,pts2=pts[b][:,:2],pts[b][:,2:]
            m1=descs1[b,:,pts1[:,1].long(),pts1[:,0].long()].permute(1,0)
            m2=descs2[b,:,pts2[:,1].long(),pts2[:,0].long()].permute(1,0)
            
            loss_per,conf_per=dual_softmax_loss(m1,m2)
            loss.append(loss_per.unsqueeze(0))
            conf_list.append(conf_per)
            
            acc_coarse_per=self.check_accuracy(m1,m2)
            acc += acc_coarse_per
            
        loss=torch.cat(loss,dim=-1).mean()
        acc /= B
        return loss,acc,conf_list
    
    
    def alike_distill_loss(self,kpts,alike_kpts):
        C, H, W = kpts.shape
        kpts = kpts.permute(1,2,0)
        # get ALike keypoints
        with torch.no_grad():
            labels = torch.ones((H, W), dtype = torch.long, device = kpts.device) * 64 # -> Default is non-keypoint (bin 64)
            offsets = (((alike_kpts/8) - (alike_kpts/8).long())*8).long()
            offsets =  offsets[:, 0] + 8*offsets[:, 1]  # Linear IDX
            labels[(alike_kpts[:,1]/8).long(), (alike_kpts[:,0]/8).long()] = offsets

        kpts = kpts.view(-1,C)
        labels = labels.view(-1)

        mask = labels < 64
        idxs_pos = mask.nonzero().flatten()
        idxs_neg = (~mask).nonzero().flatten()
        perm = torch.randperm(idxs_neg.size(0))[:len(idxs_pos)//32]
        idxs_neg = idxs_neg[perm]
        idxs = torch.cat([idxs_pos, idxs_neg])

        kpts = kpts[idxs]
        labels = labels[idxs]

        with torch.no_grad():
            predicted = kpts.max(dim=-1)[1]
            acc =  (labels == predicted)
            acc = acc.sum() / len(acc)

        kpts = F.log_softmax(kpts,dim=-1)
        loss = F.nll_loss(kpts, labels, reduction = 'mean')

        return loss, acc
    
    
    def compute_keypoints_loss(self,kpts1,kpts2,alike_kpts1,alike_kpts2):
        loss=[]
        acc=0
        B,_,H,W=kpts1.shape
        
        for b in range(B):
            loss_per1,acc_per1=self.alike_distill_loss(kpts1[b],alike_kpts1[b])
            loss_per2,acc_per2=self.alike_distill_loss(kpts2[b],alike_kpts2[b])
            loss_per=(loss_per1+loss_per2)
            acc_per=(acc_per1+acc_per2)/2
            loss.append(loss_per.unsqueeze(0))
            acc += acc_per
            
        loss=torch.cat(loss,dim=-1).mean()
        acc /= B
        return loss,acc
    
    
    def compute_heatmaps_loss(self,heatmaps1,heatmaps2,pts,conf_list):
        loss=[]
        B,_,H,W=heatmaps1.shape
        
        for b in range(B):
            pts1,pts2=pts[b][:,:2],pts[b][:,2:]
            h1=heatmaps1[b,0,pts1[:,1].long(),pts1[:,0].long()]
            h2=heatmaps2[b,0,pts2[:,1].long(),pts2[:,0].long()]
            
            conf=conf_list[b]
            loss_per1=F.l1_loss(h1,conf)
            loss_per2=F.l1_loss(h2,conf)
            loss_per=(loss_per1+loss_per2)
            loss.append(loss_per.unsqueeze(0))
            
        loss=torch.cat(loss,dim=-1).mean()
        return loss
    
    
    def normal_loss(self,normal,target_normal):
        # import pdb;pdb.set_trace()
        normal = normal.permute(1, 2, 0)
        target_normal = target_normal.permute(1,2,0)
        # loss = F.l1_loss(d_feat, depth_anything_normal_feat)
        dot = torch.cosine_similarity(normal, target_normal, dim=2)
        valid_mask = target_normal[:, :, 0].float() \
                    * (dot.detach() < 0.999).float() \
                    * (dot.detach() > -0.999).float()
        valid_mask = valid_mask > 0.0
        al = torch.acos(dot[valid_mask])
        loss = torch.mean(al)
        return loss
    
    
    def compute_normals_loss(self,normals1,normals2,DA_normals1,DA_normals2,megadepth_batch_size,coco_batch_size):
        loss=[]
        
        # import pdb;pdb.set_trace()
        
        # only MegaDepth image need depth-normal
        normals1=normals1[coco_batch_size:,...]
        normals2=normals2[coco_batch_size:,...]
        for b in range(len(DA_normals1)):
            normal1,normal2=normals1[b],normals2[b]
            loss_per1=self.normal_loss(normal1,DA_normals1[b].permute(2,0,1))
            loss_per2=self.normal_loss(normal2,DA_normals2[b].permute(2,0,1))
            loss_per=(loss_per1+loss_per2)
            loss.append(loss_per.unsqueeze(0))
        
        loss=torch.cat(loss,dim=-1).mean()
        return loss
    
    
    def coordinate_loss(self,coordinate,conf,pts1):
        with torch.no_grad():
            coordinate_detached = pts1 * 8
            offset_detached = (coordinate_detached/8) - (coordinate_detached/8).long()
            offset_detached = (offset_detached * 8).long()
            label = offset_detached[:, 0] + 8*offset_detached[:, 1]

        #pdb.set_trace()
        coordinate_log = F.log_softmax(coordinate, dim=-1)

        predicted = coordinate.max(dim=-1)[1]
        acc =  (label == predicted)
        acc = acc[conf > 0.1]
        acc = acc.sum() / len(acc)

        loss = F.nll_loss(coordinate_log, label, reduction = 'none')
        
        #Weight loss by confidence, giving more emphasis on reliable matches
        conf = conf / conf.sum()
        loss = (loss * conf).sum()

        return loss*2., acc
    
    def compute_coordinates_loss(self,coordinates,pts,conf_list):
        loss=[]
        acc=0
        B,_,H,W=coordinates.shape
        
        for b in range(B):
            pts1,pts2=pts[b][:,:2],pts[b][:,2:]
            coordinate=coordinates[b,:,pts1[:,1].long(),pts1[:,0].long()].permute(1,0)
            conf=conf_list[b]
            
            loss_per,acc_per=self.coordinate_loss(coordinate,conf,pts1)
            loss.append(loss_per.unsqueeze(0))
            acc += acc_per
            
        loss=torch.cat(loss,dim=-1).mean()
        acc /= B
        
        return loss,acc
        
        
    def forward(self,
                descs1,fb_descs1,kpts1,normals1,
                descs2,fb_descs2,kpts2,normals2,
                pts,coordinates,fb_coordinates,
                alike_kpts1,alike_kpts2,
                DA_normals1,DA_normals2,
                megadepth_batch_size,coco_batch_size
                ):
        # import pdb;pdb.set_trace()
        self.loss_descs,self.acc_coarse,conf_list=self.compute_descriptors_loss(descs1,descs2,pts)
        self.loss_fb_descs,self.acc_fb_coarse,fb_conf_list=self.compute_descriptors_loss(fb_descs1,fb_descs2,pts)
        
        # start=time.perf_counter()
        self.loss_kpts,self.acc_kpt=self.compute_keypoints_loss(kpts1,kpts2,alike_kpts1,alike_kpts2)
        # end=time.perf_counter()
        # print(f"kpts loss cost {end-start} seconds")
        
        # start=time.perf_counter()
        self.loss_normals=self.compute_normals_loss(normals1,normals2,DA_normals1,DA_normals2,megadepth_batch_size,coco_batch_size)
        # end=time.perf_counter()
        # print(f"normal loss cost {end-start} seconds")
        
        self.loss_coordinates,self.acc_coordinates=self.compute_coordinates_loss(coordinates,pts,conf_list)
        self.loss_fb_coordinates,self.acc_fb_coordinates=self.compute_coordinates_loss(fb_coordinates,pts,fb_conf_list)
        
        return {
            'loss_descs':self.lam_descs*self.loss_descs,
            'acc_coarse':self.acc_coarse,
            'loss_coordinates':self.lam_coordinates*self.loss_coordinates,
            'acc_coordinates':self.acc_coordinates,
            'loss_fb_descs':self.lam_fb_descs*self.loss_fb_descs,
            'acc_fb_coarse':self.acc_fb_coarse,
            'loss_fb_coordinates':self.lam_fb_coordinates*self.loss_fb_coordinates,
            'acc_fb_coordinates':self.acc_fb_coordinates,
            'loss_kpts':self.lam_kpts*self.loss_kpts,
            'acc_kpt':self.acc_kpt,
            'loss_normals':self.lam_normals*self.loss_normals,
        }

