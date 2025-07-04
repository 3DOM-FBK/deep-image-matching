import h5py
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from deep_image_matching.io.h5_to_db import COLMAPDatabase, image_ids_to_pair_id
from typing import Optional
import os, h5py, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ExifTags
import argparse

# Credit to: https://github.com/ducha-aiki/imc2023-kornia-starter-pack/blob/main/loftr-pycolmap-3dreconstruction.ipynb

def get_focal(image_path, err_on_default=False):
    image         = Image.open(image_path)
    max_size      = max(image.size)
    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break
        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")
        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size
    return focal

def create_camera(db, image_path, camera_model):
    image         = Image.open(image_path)
    width, height = image.size
    focal = get_focal(image_path)
    if camera_model == 'simple-pinhole':
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == 'pinhole':
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'opencv':
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
    return db.add_camera(model, width, height, param_arr)


def add_keypoints(db, h5_path, image_path, camera_model, single_camera = True):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')
    fname_to_id = {}
    db.clean_keypoints()
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]
        fname_with_ext = filename
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')
        images = db.get_images()
        image_id, camera_id = images[filename]
        fname_to_id[filename] = image_id
        db.add_keypoints(image_id, keypoints)
    return fname_to_id

def add_matches(db, h5_path, fname_to_id):
    db.clean_matches()
    db.clean_two_view_geometries()
    match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')
    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2
    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]
                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                    continue
                matches = group[key_2][()]
                #db.add_matches(id_1, id_2, matches)
                db.add_two_view_geometry(id_1, id_2, matches)
                added.add(pair_id)
                pbar.update(1)

def get_unique_idxs(A, dim=1):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

def import_into_colmap(img_dir,
                       feature_dir ='.featureout',
                       database_path = 'colmap.db',
                       ):

    db = COLMAPDatabase.connect(database_path)
    #db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, 'simple-radial', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()
    return

def LoftrRomaToMultiview(
        input_dir: Path,
        output_dir: Path,
        image_dir: Path, 
        img_ext: Path,
        mask_dir: Optional[Path] = None,
        ) -> None:
    if mask_dir:
        image_names = [p.name for p in image_dir.glob(f"*{img_ext}")]
        mask_paths = {p: os.path.join(mask_dir, p.replace(img_ext,'.png')) for p in image_names}
        mask_arr = {}
        for p in image_names:
            if not os.path.exists(mask_paths[p]):
                FileNotFoundError(f"Mask for {p} not found, expected path: {mask_paths[p]}")
            mask_arr[p] = np.array(Image.open(mask_paths[p])) > 0  # ensure binary
    with h5py.File(fr'{input_dir}/features.h5', mode='r') as h5_feats, \
        h5py.File(fr'{input_dir}/matches.h5', mode='r') as h5_matches, \
        h5py.File(fr'{input_dir}/matches_loftr.h5', mode='w') as h5_out:
        for img1 in h5_matches.keys():
            print(img1)
            kpts1 = h5_feats[img1]['keypoints'][...]
            group_match = h5_matches[img1]
            group_out = h5_out.require_group(img1)
            for img2 in group_match.keys():
                print(f"--- {img2}")
                kpts2 = h5_feats[img2]['keypoints'][...]
                matches = group_match[img2][...]
                h5_out[img1][img2] = np.hstack((kpts1[matches[:,0],:], kpts2[matches[:,1],:]))
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(fr'{input_dir}/matches_loftr.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                if mask_dir:
                    matches = matches[
                        mask_arr[k1][matches[:, 1].astype(int), matches[:, 0].astype(int)] &
                        mask_arr[k2][matches[:, 3].astype(int), matches[:, 2].astype(int)]
                    ]
                total_kpts[k1]#???
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match
    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:,0] = unique_match_idxs[k1][m2[:,0]]
            m2[:,1] = unique_match_idxs[k2][m2[:,1]]
            mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                    unique_kpts[k2][  m2[:,1]],
                                   ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(fr'{output_dir}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
    with h5py.File(fr'{output_dir}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match

    #try:
    #    os.remove(f"{output_dir}/database.db")
    #except:
    #    pass

    import_into_colmap(
        image_dir, 
        feature_dir=f"{output_dir}", 
        database_path=f"{output_dir}/database.db")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LOFTR and Roma matchers to multi-view. Assign same index to keypoints in different images with distance < 1 px')
    parser.add_argument('-i', '--input_dir', type=str, help='Path to directory containing databases features.h5 and matches.h5')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-d', '--image_dir', type=str, help='Image directory')
    parser.add_argument('-e', '--img_ext', type=str, default='.jpg', help='Image extension')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    image_dir = args.image_dir
    img_ext = args.img_ext

    LoftrRomaToMultiview(
        input_dir,
        output_dir,
        image_dir, 
        img_ext,
        )
