import argparse
import glob
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from deep_image_matching.utils.database import (
    COLMAPDatabase,
    blob_to_array,
    pair_id_to_image_ids,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge all COLMAP databases of the input folder."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Folder containing COLMAP databases to join",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output folder (if None, use the input folder)",
        default=None,
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    return args


def MergeColmapDatabases(db_path1: Path, db_path2: Path, db_joined: Path) -> None:
    print(f"\nMerging {db_path1.name} and {db_path2.name} into {db_joined.name}")

    db1 = COLMAPDatabase.connect(db_path1)
    db2 = COLMAPDatabase.connect(db_path2)
    db3 = COLMAPDatabase.connect(db_joined)
    db3.create_tables()

    # Load cameras in merged database
    rows = db2.execute("SELECT * FROM cameras")
    for row in rows:
        camera_id, model, width, height, params, prior = row
        params = blob_to_array(params, np.float64)
        camera_id1 = db3.add_camera(model, width, height, params, prior)

    # Read existing kpts from database 2
    keypoints2 = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db2.execute("SELECT image_id, data FROM keypoints")
    )

    matches2 = {}
    for pair_id, r, c, data in db2.execute(
        "SELECT pair_id, rows, cols, data FROM matches"
    ):
        if data is not None:
            pair_id = pair_id_to_image_ids(pair_id)
            matches2[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(
                data, np.uint32, (-1, 2)
            )

    two_views_matches2 = {}
    for pair_id, r, c, data in db2.execute(
        "SELECT pair_id, rows, cols, data FROM two_view_geometries"
    ):
        if data is not None:
            pair_id = pair_id_to_image_ids(pair_id)
            two_views_matches2[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(
                data, np.uint32, (-1, 2)
            )

    # Store all imgs
    img_list = list(keypoints2.keys())

    # Add images in merged database
    imgs2 = dict(
        (image_id, (name, camera_id))
        for image_id, name, camera_id in db2.execute(
            "SELECT image_id, name, camera_id FROM images"
        )
    )

    for image_id in list(imgs2.keys()):
        db3.add_image(imgs2[image_id][0], imgs2[image_id][1])

    # Read existing kpts from database 1
    keypoints1 = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))  # (-1, 6)
        for image_id, data in db1.execute("SELECT image_id, data FROM keypoints")
    )

    print("keypoints1 shape", np.shape(keypoints1[img_list[0]]), "First 5 rows:")
    print(keypoints1[img_list[0]][:5, :])
    print("keypoints2 shape", np.shape(keypoints2[img_list[0]]), "First 5 rows:")
    print(keypoints2[img_list[0]][:5, :])

    # Read existing matches from database 1
    matches1 = {}
    for pair_id, r, c, data in db1.execute(
        "SELECT pair_id, rows, cols, data FROM matches"
    ):
        if data is not None:
            pair_id = pair_id_to_image_ids(pair_id)
            matches1[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(
                data, np.uint32, (-1, 2)
            )

    two_views_matches1 = {}
    for pair_id, r, c, data in db1.execute(
        "SELECT pair_id, rows, cols, data FROM two_view_geometries"
    ):
        if data is not None:
            pair_id = pair_id_to_image_ids(pair_id)
            two_views_matches1[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(
                data, np.uint32, (-1, 2)
            )

    n_kpts_dict = {}

    # Merge kpts
    for im_id in img_list:
        n_kpts = np.shape(keypoints1[im_id])[0]
        keypoints = np.vstack((keypoints1[im_id][:, :2], keypoints2[im_id][:, :2]))
        # keypoints = np.vstack((keypoints1[im_id], keypoints2[im_id]))
        db3.add_keypoints(im_id, keypoints)
        n_kpts_dict[im_id] = n_kpts

    # Merge raw matches
    all_matches = {}
    for pair in matches1:
        all_matches[pair] = matches1[pair]

    for pair in list(matches2.keys())[:]:
        im1 = int(pair[0])
        im2 = int(pair[1])
        matches2_im1 = matches2[pair][:, 0] + n_kpts_dict[im1]
        matches2_im2 = matches2[pair][:, 1] + n_kpts_dict[im2]
        matches2_im1 = matches2_im1.reshape((-1, 1))
        matches2_im2 = matches2_im2.reshape((-1, 1))
        joinedmatches2 = np.hstack((matches2_im1, matches2_im2))
        if pair not in matches1.keys():
            all_matches[pair] = joinedmatches2
        elif pair in matches1.keys():
            all_matches[pair] = np.vstack((matches1[pair], joinedmatches2))

    for pair in all_matches:
        im1 = int(pair[0])
        im2 = int(pair[1])
        db3.add_matches(im1, im2, all_matches[pair])

    # Merge verified matches
    all_matches = {}
    for pair in two_views_matches1:
        all_matches[pair] = two_views_matches1[pair]

    for pair in list(two_views_matches2.keys())[:]:
        im1 = int(pair[0])
        im2 = int(pair[1])
        matches2_im1 = two_views_matches2[pair][:, 0] + n_kpts_dict[im1]
        matches2_im2 = two_views_matches2[pair][:, 1] + n_kpts_dict[im2]
        matches2_im1 = matches2_im1.reshape((-1, 1))
        matches2_im2 = matches2_im2.reshape((-1, 1))
        joinedmatches2 = np.hstack((matches2_im1, matches2_im2))
        if pair not in two_views_matches1.keys():
            all_matches[pair] = joinedmatches2
        elif pair in two_views_matches1.keys():
            all_matches[pair] = np.vstack((two_views_matches1[pair], joinedmatches2))

    for pair in all_matches:
        im1 = int(pair[0])
        im2 = int(pair[1])
        db3.add_two_view_geometry(im1, im2, all_matches[pair])

    db3.commit()
    db1.close()
    db2.close()
    db3.close()


def main():
    args = parse_args()

    databases_dir = Path(args.input)
    db_out = Path(args.output) / "joined.db"
    db_temp = db_out.parent / "temp.db"
    if db_out.exists():
        os.remove(db_out)
    if db_temp.exists():
        os.remove(db_temp)
    db_files = glob.glob(f"{databases_dir}/*.db")

    print("\nMERGING DATABASES ..")
    print(f"Found {len(db_files)} COLMAP databases in {databases_dir}.")
    print("db_out - joined database", db_out)
    print("db_temp - temporary file", db_temp)
    print("\n")

    for db_file in tqdm(db_files[1:], total=len(db_files), mininterval=0):
        if not db_temp.exists():
            db0 = Path(db_files[0])
            db1 = Path(db_files[1])
            MergeColmapDatabases(db0, db1, db_temp)
        else:
            db0 = deepcopy(db_temp)
            db1 = Path(db_file)
            print(db0, db1)
            MergeColmapDatabases(db0, db1, db_out)
            os.remove(db_temp)
            os.rename(db_out, db_temp)

    os.rename(db_temp, db_out)


if __name__ == "__main__":
    main()
