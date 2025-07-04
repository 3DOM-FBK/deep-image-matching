import time
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from src.deep_image_matching.thirdparty.MiHo.src import ncc as ncc
from pycolmap import Database, Reconstruction


def convert_image(im):

    transform_gray = transforms.Compose([
        transforms.Grayscale(),
        transforms.PILToTensor() 
        ]) 

    #transform = transforms.PILToTensor() 
    #img1 = transform(im1).type(torch.float16).to(device)
    #img2 = transform(im2).type(torch.float16).to(device)

    img = transform_gray(im).type(torch.float16).to(device)
        
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_db", help="Path to the input database")
    parser.add_argument("output_db", help="Path to the output database")
    parser.add_argument("model", help="Path to an existing COLMAP model")
    parser.add_argument("images", help="Path to images folder")
    args = parser.parse_args()

    db = Database(args.input_db)
    out_db = Database(args.output_db)
    images = db.read_all_images()
    images_dir = Path(args.images)
    reconstruction = Reconstruction(args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w = 15
    ref_image=['left'] ################################################################## mettere right
    angle=[-30, -15, 0, 15, 30]
    scale = [[10/14, 1], [10/12, 1], [1, 1], [1, 12/10], [1, 14/10]]

    point3Ds = []
    for point3D_id in reconstruction.points3D:
        point3Ds.append(point3D_id)

    for point3D_id in point3Ds:
        point3D = reconstruction.points3D[point3D_id]
        #print(f"3D Point {point3D_id} coordinates: {point3D.xyz}")
        for t, track in enumerate(point3D.track.elements):
            if t == 0:
                image_id_ref = track.image_id
                image_name_ref = db.read_image(image_id_ref).name
                point2D_id_ref = track.point2D_idx
                keypoints = db.read_keypoints(image_id_ref)
                keypoint_ref = keypoints[point2D_id_ref]
                keypoint_ref = torch.from_numpy(np.array([keypoint_ref])).to(device)
                #print(keypoint_ref);quit()
                im_ref = convert_image(Image.open(images_dir / image_name_ref))
            else:
                image_id_target = track.image_id
                image_name_target = db.read_image(image_id_target).name
                point2D_id_target = track.point2D_idx
                keypoints = db.read_keypoints(image_id_target)
                keypoint_target = keypoints[point2D_id_target]
                keypoint_target = torch.from_numpy(np.array([keypoint_target])).to(device)
                im_target = convert_image(Image.open(images_dir / image_name_target))

                l = 1
                Hs = torch.eye(3, device=device).repeat(l*2, 1).reshape(l, 2, 3, 3)
                t0 = time.time()
                res = ncc.refinement_norm_corr_alternate(im1=im_ref, im2=im_target, pt1=keypoint_ref, pt2=keypoint_target, Hs=Hs, w=w, ref_image=['both'], angle=angle, scale=scale, subpix=True, img_patches=True, im1_disp=im_ref, im2_disp=im_target)
                t1 = time.time()
                print(f"Elapsed = {t1 - t0} (NCC refinement)")
                print(keypoint_ref)
                print(keypoint_target)
                print(res)
                quit()



