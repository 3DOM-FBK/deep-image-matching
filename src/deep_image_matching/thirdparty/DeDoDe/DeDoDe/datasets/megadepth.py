import os
from PIL import Image
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as tvf
from tqdm import tqdm

from DeDoDe.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops
import DeDoDe
from DeDoDe.utils import *

class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        ht=512,
        wt=512,
        min_overlap=0.0,
        max_overlap=1.0,
        shake_t=0,
        scene_info_detections=None,
        scene_info_detections3D=None,
        normalize=True,
        max_num_pairs = 100_000,
        scene_name = None,
        use_horizontal_flip_aug = False,
        grayscale = False,
        clahe = False,
    ) -> None:
        self.data_root = data_root
        self.scene_name = os.path.splitext(scene_name)[0]+f"_{min_overlap}_{max_overlap}"
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        self.detections = scene_info_detections
        self.tracks3D = scene_info_detections3D
        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=normalize, clahe = clahe,
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
            resize=(ht, wt), normalize=False
        )
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.grayscale = grayscale

    def load_im(self, im_B, crop=None):
        im = Image.open(im_B)
        return im
    
    def horizontal_flip(self, im_A, im_B, depth_A, depth_B,  K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1) 
        flip_mat = torch.tensor([[-1, 0, self.wt],[0,1,0],[0,0,1.]]).to(K_A.device)
        K_A = flip_mat@K_A  
        K_B = flip_mat@K_B  
        
        return im_A, im_B, depth_A, depth_B, K_A, K_B
        
    def load_depth(self, depth_ref, crop=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def scale_detections(self, detections, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        return detections * torch.tensor([[sx,sy]])
    
    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=(2))
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def tracks_to_detections(self, tracks3D, pose, intrinsics, H, W):
        tracks3D = tracks3D.double()
        intrinsics = intrinsics.double()
        bearing_vectors = pose[...,:3,:3] @ tracks3D.mT + pose[...,:3,3:]        
        hom_pixel_coords = (intrinsics @ bearing_vectors).mT
        pixel_coords = hom_pixel_coords[...,:2] / (hom_pixel_coords[...,2:]+1e-12)
        legit_detections = (pixel_coords > 0).prod(dim = -1) * (pixel_coords[...,0] < W - 1) * (pixel_coords[...,1] < H - 1) * (tracks3D != 0).prod(dim=-1)
        return pixel_coords.float(), legit_detections.bool()

    def __getitem__(self, pair_idx):
        try:
            # read intrinsics of original size
            idx1, idx2 = self.pairs[pair_idx]
            K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
            K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)

            # read and compute relative poses
            T1 = self.poses[idx1]
            T2 = self.poses[idx2]
            T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[
                :4, :4
            ]  # (4, 4)

            # Load positive pair data
            im_A, im_B = self.image_paths[idx1], self.image_paths[idx2]
            depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
            im_A_ref = os.path.join(self.data_root, im_A)
            im_B_ref = os.path.join(self.data_root, im_B)
            depth_A_ref = os.path.join(self.data_root, depth1)
            depth_B_ref = os.path.join(self.data_root, depth2)
            # return torch.randn((1000,1000))
            im_A = self.load_im(im_A_ref)
            im_B = self.load_im(im_B_ref)
            depth_A = self.load_depth(depth_A_ref)
            depth_B = self.load_depth(depth_B_ref)

            # Recompute camera intrinsic matrix due to the resize
            W_A, H_A = im_A.width, im_A.height
            W_B, H_B = im_B.width, im_B.height

            detections2D_A = self.detections[idx1]
            detections2D_B = self.detections[idx2]
            
            K = 10000
            tracks3D_A = torch.zeros(K,3)
            tracks3D_B = torch.zeros(K,3)
            tracks3D_A[:len(detections2D_A)] = torch.tensor(self.tracks3D[detections2D_A[:K,-1].astype(np.int32)])
            tracks3D_B[:len(detections2D_B)] = torch.tensor(self.tracks3D[detections2D_B[:K,-1].astype(np.int32)])
            
            #projs_A, _ = self.tracks_to_detections(tracks3D_A, T1, K1, W_A, H_A)
            #tracks3D_B = torch.zeros(K,2)

            K1 = self.scale_intrinsic(K1, W_A, H_A)
            K2 = self.scale_intrinsic(K2, W_B, H_B)
            
            # Process images
            im_A, im_B = self.im_transform_ops((im_A, im_B))
            depth_A, depth_B = self.depth_transform_ops(
                (depth_A[None, None], depth_B[None, None])
            )
            [im_A, depth_A], t_A = self.rand_shake(im_A, depth_A)
            [im_B, depth_B], t_B = self.rand_shake(im_B, depth_B)

            detections_A = -torch.ones(K,2)
            detections_B = -torch.ones(K,2)
            detections_A[:len(self.detections[idx1])] = self.scale_detections(torch.tensor(detections2D_A[:K,:2]), W_A, H_A) + t_A
            detections_B[:len(self.detections[idx2])] = self.scale_detections(torch.tensor(detections2D_B[:K,:2]), W_B, H_B) + t_B

            
            K1[:2, 2] += t_A
            K2[:2, 2] += t_B
                    
            if self.use_horizontal_flip_aug:
                if np.random.rand() > 0.5:
                    im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K1, K2)
                    detections_A[:,0] = W-detections_A
                    detections_B[:,0] = W-detections_B
                    
            if DeDoDe.DEBUG_MODE:
                tensor_to_pil(im_A[0], unnormalize=True).save(
                                f"vis/im_A.jpg")
                tensor_to_pil(im_B[0], unnormalize=True).save(
                                f"vis/im_B.jpg")
            if self.grayscale:
                im_A = im_A.mean(dim=-3,keepdim=True)
                im_B = im_B.mean(dim=-3,keepdim=True)
            data_dict = {
                "im_A": im_A,
                "im_A_identifier": self.image_paths[idx1].split("/")[-1].split(".jpg")[0],
                "im_B": im_B,
                "im_B_identifier": self.image_paths[idx2].split("/")[-1].split(".jpg")[0],
                "im_A_depth": depth_A[0, 0],
                "im_B_depth": depth_B[0, 0],
                "pose_A": T1,
                "pose_B": T2,
                "detections_A": detections_A,
                "detections_B": detections_B,
                "tracks3D_A": tracks3D_A,
                "tracks3D_B": tracks3D_B,
                "K1": K1,
                "K2": K2,
                "T_1to2": T_1to2,
                "im_A_path": im_A_ref,
                "im_B_path": im_B_ref,
            }
        except Exception as e:
            print(e)
            print(f"Failed to load image pair {self.pairs[pair_idx]}")
            print("Loading a random pair in scene instead")
            rand_ind = np.random.choice(range(len(self)))
            return self[rand_ind]
        return data_dict


class MegadepthBuilder:
    def __init__(self, data_root="data/megadepth", loftr_ignore=True, imc21_ignore = True) -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        # LoFTR did the D2-net preprocessing differently than we did and got more ignore scenes, can optionially ignore those
        self.loftr_ignore_scenes = set(['0121.npy', '0133.npy', '0168.npy', '0178.npy', '0229.npy', '0349.npy', '0412.npy', '0430.npy', '0443.npy', '1001.npy', '5014.npy', '5015.npy', '5016.npy'])
        self.imc21_scenes = set(['0008.npy', '0019.npy', '0021.npy', '0024.npy', '0025.npy', '0032.npy', '0063.npy', '1589.npy'])
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]
        self.loftr_ignore = loftr_ignore
        self.imc21_ignore = imc21_ignore

    def build_scenes(self, split="train", min_overlap=0.0, scene_names = None, **kwargs):
        if split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        elif split == "custom":
            scene_names = scene_names
        else:
            raise ValueError(f"Split {split} not available")
        scenes = []
        for scene_name in tqdm(scene_names):
            if self.loftr_ignore and scene_name in self.loftr_ignore_scenes:
                continue
            if self.imc21_ignore and scene_name in self.imc21_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            scene_info_detections = np.load(
                os.path.join(self.scene_info_root, "detections", f"detections_{scene_name}"), allow_pickle=True
            ).item()
            scene_info_detections3D = np.load(
                os.path.join(self.scene_info_root, "detections3D", f"detections3D_{scene_name}"), allow_pickle=True
            )

            scenes.append(
                MegadepthScene(
                    self.data_root, scene_info, scene_info_detections = scene_info_detections, scene_info_detections3D = scene_info_detections3D, min_overlap=min_overlap,scene_name = scene_name, **kwargs
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws
