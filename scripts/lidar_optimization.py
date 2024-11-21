import tqdm
import random
import argparse
import pycolmap
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description='Process point cloud and colmap model folder')
    parser.add_argument('point_cloud', type=str, help='Path to the point cloud file')
    parser.add_argument('colmap_folder', type=str, help='Path to the folder containing the colmap model')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')
    parser.add_argument('camera_pos', type=str, help='Path to camera positions')

    args = parser.parse_args()
    point_cloud_path = args.point_cloud
    colmap_folder_path = args.colmap_folder
    output_folder = args.output_folder
    camera_pos_path = args.camera_pos

    constraints = {}
    reconstruction = pycolmap.Reconstruction(colmap_folder_path)
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    points_ply = np.asarray(point_cloud.points)
    kdtree = cKDTree(points_ply)

    #camera_pos = {}
    #with open(camera_pos_path, 'r') as f:
    #    lines = f.readlines()
    #    for line in lines:
    #        line = line.strip()
    #        image, x, y, z, _ = line.split('\t', 4)
    #        for img_id, img in reconstruction.images.items():
    #            if img.name == image:
    #                camera_pos[img_id] = np.array([float(x), float(y), float(z)])
    #                break
#
    #scale, rotation, translation = pycolmap.align_reconstruction(reconstruction, camera_pos);quit()

    point3Ds = []
    for point3D_id in reconstruction.points3D:
        point3Ds.append(point3D_id)
    
    random_point3D_ids = random.sample(point3Ds, 200)

    for point3D_id in random_point3D_ids:
        point3D = reconstruction.points3D[point3D_id]
        #print(f"3D Point {point3D_id} coordinates: {point3D.xyz}")

        projections = []
        for image in reconstruction.images.values():
            # For each image, check if it observes the 3D point
            for feature_idx, point2D in enumerate(image.points2D):
                if point2D.has_point3D and point2D.point3D_id == point3D_id:
                    # Add the projection information: image ID and 2D point coordinates
                    projections.append((image.image_id, point2D.xy))
                    #print(f"Image {image.image_id}: Projected 2D point (x, y) = {point2D.xy}")
    
        constraints[point3D_id] = {
            '3D_tiepoint': point3D.xyz,
            'projections': projections,
            'lidar_point': None,
        }
    
    # Export to txt file
    with open(f'{output_folder}/constraints.txt', 'w') as f:
        for point3D_id in random_point3D_ids:
            for image_id, point2D in constraints[point3D_id]['projections']:
                image = reconstruction.images[image_id]
                f.write(f"{point3D_id},{image.name[:-4]},{point2D[0]},{point2D[1]}\n")


    query_points = np.empty((0, 3))
    ids = []
    for point3D_id in random_point3D_ids:
        x = constraints[point3D_id]['3D_tiepoint'][0]
        y = constraints[point3D_id]['3D_tiepoint'][1]
        z = constraints[point3D_id]['3D_tiepoint'][2]
        query_points = np.vstack((query_points, np.array([float(x), float(y), float(z)])))
        ids.append(point3D_id)
    
    distances, indices = kdtree.query(query_points)

    for i, (point3D_id, index) in enumerate(zip(ids, indices)):
        nearest_point = points_ply[index]
        constraints[point3D_id]['lidar_point'] = nearest_point

    with open(f'{output_folder}/points.txt', 'w') as f:
        for point3D_id in random_point3D_ids:
            x_photogram = float(constraints[point3D_id]['3D_tiepoint'][0])
            y_photogram = float(constraints[point3D_id]['3D_tiepoint'][1])
            z_photogram = float(constraints[point3D_id]['3D_tiepoint'][2])
            x_lidar = float(constraints[point3D_id]['lidar_point'][0])
            y_lidar = float(constraints[point3D_id]['lidar_point'][1])
            z_lidar = float(constraints[point3D_id]['lidar_point'][2])
            norm = ((x_photogram-x_lidar)**2+(y_photogram-y_lidar)**2+(z_photogram-z_lidar)**2)**0.5
            print(norm)
            if norm < 0.20:
                f.write(f"{point3D_id},{x_lidar},{y_lidar},{z_lidar}\n")


if __name__ == '__main__':
    main()