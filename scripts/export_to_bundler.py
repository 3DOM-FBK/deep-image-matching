import pycolmap
import argparse
import networkx as nx

from pathlib import Path


def build_tracks(
    colmap_database_path: Path,
    ):
    
    database = pycolmap.Database(str(colmap_database_path))
    all_pairs, all_matches = database.read_all_matches()


    G = nx.Graph()
    for pairs, matches in zip(all_pairs, all_matches):
        image_id1, image_id2 = database.pair_id_to_image_pair(pairs)
        #if geom.config != pycolmap.TwoViewGeometryConfig.SUCCESS:
        #    continue
        for (i1, i2) in matches:
            G.add_edge((image_id1, i1), (image_id2, i2))

    tracks = list(nx.connected_components(G))
    
    return database, tracks


def export_to_bundler(
    database,
    tracks,
    output_dir: Path,
    ):

    images = database.read_all_images()
    num_images = len(images)
    num_points = len(tracks)

    cameras = database.read_all_cameras()
    
    out_file = open(output_dir / "bundler.out", 'w')
    image_out_file = open(output_dir / "bundler.out.list.txt", 'w')

    out_file.write(f"# Bundle file v0.3\n")
    out_file.write(f"{num_images} {num_points}\n")

    # Camera Parameter Blocks
    for i,image in enumerate(images):
        image_id = image.image_id
        camera_id = image.camera_id
        name = image.name
        image_out_file.write(f"{name}\n")

        #out_file.write(f"# Camera {i}\n")
        out_file.write("1000 0 0\n")    # f k1 k2
        out_file.write("1.0 0.0 0.0\n") # R11 R12 R13
        out_file.write("0.0 1.0 0.0\n") # R21 R22 R23
        out_file.write("0.0 0.0 1.0\n") # R31 R32 R33
        out_file.write("0 0 0\n")       # t1 t2 t3

    # 3D Point Blocks
    for t,track in enumerate(tracks):
        #out_file.write(f"# Point {t}\n")
        track_length = len(track)
        out_file.write("0.0 0.0 0.0\n")  # X Y Z
        out_file.write("250 250 250\n")  # R G B
        out_file.write(f"{track_length} ")
        for (image_id, keypoint_idx) in track:
            image = database.read_image(image_id)
            camera_id = image.camera_id
            camera = database.read_camera(camera_id)
            width = camera.width
            height = camera.height

            keypoints = database.read_keypoints(image_id)
            keypoint = keypoints[keypoint_idx]
            x, y = keypoint
            
            x = x - width/2          
            y = height/2 - y
            out_file.write(f"{image_id-1} {keypoint_idx} {x} {y} ")
        out_file.write("\n")

    out_file.close()
    image_out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export COLMAP database model to Bundler format")
    parser.add_argument("-d", "--colmap_database_path", type=Path, help="Path to the COLMAP database file", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, help="Path to the output dir", required=True)
    args = parser.parse_args()  

    database, tracks = build_tracks(args.colmap_database_path)
    export_to_bundler(database, tracks, args.output_dir)