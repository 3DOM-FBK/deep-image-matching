#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import threading
import argparse
import json
import os
import warnings
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm

from .. import logger
from ..utils.database import COLMAPDatabase, image_ids_to_pair_id

def loadJSON(sfm_data):
  with open(sfm_data) as file:
    sfm_data = json.load(file)
  view_ids = {view['value']['ptr_wrapper']['data']['filename']:view['key'] for view in sfm_data['views']}
  image_paths = [os.path.join(sfm_data['root_path'], view['value']['ptr_wrapper']['data']['filename']) for view in sfm_data['views']]
  return view_ids, image_paths

def saveFeaturesOpenMVG(matches_folder, basename, keypoints):
  with open(os.path.join(matches_folder, f'{basename}.feat'), 'w') as feat:
    for x, y in keypoints:
      feat.write(f'{x} {y} 1.0 0.0\n')

def saveDescriptorsOpenMVG(matches_folder, basename, descriptors):
  with open(os.path.join(matches_folder, f'{basename}.desc'), 'wb') as desc:
    desc.write(len(descriptors).to_bytes(8, byteorder='little'))
    desc.write(((descriptors.numpy() + 1) * 0.5 * 255).round(0).astype(np.ubyte).tobytes())

def saveMatchesOpenMVG(matches, out_folder):
  with open(out_folder / 'matches.putative.bin', 'wb') as bin:
    bin.write((1).to_bytes(1, byteorder='little'))
    bin.write(len(matches).to_bytes(8, byteorder='little'))
    for index1, index2, idxs in matches:
      bin.write(index1.tobytes())
      bin.write(index2.tobytes())
      bin.write(len(idxs).to_bytes(8, byteorder='little'))
      bin.write(idxs.tobytes())

def add_keypoints(h5_path, image_path, matches_dir):
    keypoint_f = h5py.File(str(h5_path), "r")
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename]["keypoints"].__array__()

        path = os.path.join(image_path, filename)
        if not os.path.isfile(path):
            raise IOError(f"Invalid image path {path}")
        if len(keypoints.shape) >= 2:
            threading.Thread(target=lambda: saveFeaturesOpenMVG(matches_dir, filename, keypoints)).start()
            #threading.Thread(target=lambda: saveDescriptorsOpenMVG(matches_dir, filename, features.descriptors)).start()
    return

def add_matches(h5_path, sfm_data, matches_dir):
    view_ids, image_paths = loadJSON(sfm_data)
    putative_matches = []

    match_file = h5py.File(str(h5_path), "r")
    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = view_ids[key_1]
                id_2 = view_ids[key_2]
                if (key_1, key_2) in added:
                    warnings.warn(f"Pair ({key_1}, {key_2}) already added!")
                    continue
                matches = group[key_2][()]
                putative_matches.append([np.int32(id_1), np.int32(id_2), matches.astype(np.int32)])
                added.add((key_1, key_2))
                pbar.update(1)
    match_file.close()
    saveMatchesOpenMVG(putative_matches, matches_dir)

def export_to_openmvg(
    img_dir,
    feature_path: Path,
    match_path: Path,
    openmvg_out_path: Path,
    openmvg_sfm_bin: Path,
):
    if openmvg_out_path.exists():
        logger.warning(f"OpenMVG output folder {openmvg_out_path} already exists - deleting it")
        os.rmdir(openmvg_out_path)
    os.makedirs(openmvg_out_path)

    camera_file_params = openmvg_sfm_bin / "sensor_width_database" / "sensor_width_camera_database.txt"
    matches_dir = openmvg_out_path / "matches"

    pIntrisics = subprocess.Popen( [os.path.join(openmvg_sfm_bin, "openMVG_main_SfMInit_ImageListing"),  "-i", img_dir, "-o", matches_dir, "-d", camera_file_params] )
    pIntrisics.wait()

    add_keypoints(feature_path, img_dir, matches_dir)
    add_matches(match_path, openmvg_out_path / "matches" / "sfm_data.json", matches_dir)
    print('quit() vedi fine di h5_to_openmvg')
    print('testare i risultati, bisogna salvare i matches come matches verificati se no openmvg non funziona')
    quit()

    return