import argparse
import numpy as np

import os


base_path = "data/megadepth"
# Remove the trailing / if need be.
if base_path[-1] in ['/', '\\']:
    base_path = base_path[: - 1]


base_depth_path = os.path.join(
    base_path, 'phoenix/S6/zl548/MegaDepth_v1'
)
base_undistorted_sfm_path = os.path.join(
    base_path, 'Undistorted_SfM'
)

scene_ids = os.listdir(base_undistorted_sfm_path)
for scene_id in scene_ids:
    if os.path.exists(f"{base_path}/prep_scene_info/detections/detections_{scene_id}.npy"):
        print(f"skipping {scene_id} as it exists")
        continue
    undistorted_sparse_path = os.path.join(
        base_undistorted_sfm_path, scene_id, 'sparse-txt'
    )
    if not os.path.exists(undistorted_sparse_path):
        print("sparse path doesnt exist")
        continue

    depths_path = os.path.join(
        base_depth_path, scene_id, 'dense0', 'depths'
    )
    if not os.path.exists(depths_path):
        print("depths doesnt exist")
        
        continue

    images_path = os.path.join(
        base_undistorted_sfm_path, scene_id, 'images'
    )
    if not os.path.exists(images_path):
        print("images path doesnt exist")
        continue

    # Process cameras.txt
    if not os.path.exists(os.path.join(undistorted_sparse_path, 'cameras.txt')):
        print("no cameras")
        continue
    with open(os.path.join(undistorted_sparse_path, 'cameras.txt'), 'r') as f:
        raw = f.readlines()[3 :]  # skip the header

    camera_intrinsics = {}
    for camera in raw:
        camera = camera.split(' ')
        camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2 :]]

    # Process points3D.txt
    with open(os.path.join(undistorted_sparse_path, 'points3D.txt'), 'r') as f:
        raw = f.readlines()[3 :]  # skip the header

    points3D = {}
    for point3D in raw:
        point3D = point3D.split(' ')
        points3D[int(point3D[0])] = np.array([
            float(point3D[1]), float(point3D[2]), float(point3D[3])
        ])
        
    # Process images.txt
    with open(os.path.join(undistorted_sparse_path, 'images.txt'), 'r') as f:
        raw = f.readlines()[4 :]  # skip the header

    image_id_to_idx = {}
    image_names = []
    raw_pose = []
    camera = []
    points3D_id_to_2D = []
    n_points3D = []
    id_to_detections = {}
    for idx, (image, points) in enumerate(zip(raw[:: 2], raw[1 :: 2])):
        image = image.split(' ')
        points = points.split(' ')

        image_id_to_idx[int(image[0])] = idx

        image_name = image[-1].strip('\n')
        image_names.append(image_name)

        raw_pose.append([float(elem) for elem in image[1 : -2]])
        camera.append(int(image[-2]))
        points_np = np.array(points).astype(np.float32).reshape(len(points)//3, 3)
        visible_points = points_np[points_np[:,2] != -1]
        id_to_detections[idx] = visible_points
    np.save(f"{base_path}/prep_scene_info/detections/detections_{scene_id}.npy",
            id_to_detections)
    print(f"{scene_id} done")