import os
from pathlib import Path

import numpy as np

from ..thirdparty.database import COLMAPDatabase
from .h5_to_db import get_focal


def ExportToColmap(
    images: list,
    width: int,
    height: int,
    keypoints: dict,
    correspondences: dict,
    output_dir: Path,
) -> None:
    # tmp by Francesco to get focal length from exif
    img0 = images.img_paths[0]
    images = [im.name for im in images.img_paths]

    # Generate colmap database with camera model and images
    db_path = output_dir / "db.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    # By Luca
    model, params = (
        0,
        np.array((1024.0, 512.0, 384.0)),
    )
    # Add camera model
    focal = get_focal(img0)
    model, params = (
        0,
        np.array([focal, width / 2, height / 2]),
    )

    img_dict = {}
    camera_id = db.add_camera(model, width, height, params)
    for img_name in images:
        image_id = db.add_image(img_name, camera_id)
        img_dict[img_name] = image_id

    # Export keypoints
    for img in keypoints:
        with open(output_dir / f"{img}.txt", "w") as file:
            kpts = keypoints[img]
            file.write(f"{kpts.shape[0]} 128\n")
            for row in range(kpts.shape[0]):
                file.write(f"{kpts[row, 0]} {kpts[row, 1]} 0.00 0.00\n")
        img_id = img_dict[img]
        db.add_keypoints(img_id, kpts)

    # Export tie points
    with open(output_dir / "colmap_matches.txt", "w") as file:
        for pair in correspondences:
            im0 = pair[0].name
            im1 = pair[1].name
            file.write(f"{im0} {im1}\n")
            pair_tie_points = correspondences[pair]
            for row in range(pair_tie_points.shape[0]):
                file.write(
                    "{} {}\n".format(pair_tie_points[row, 0], pair_tie_points[row, 1])
                )
            file.write("\n")
            # db.add_matches(img_dict[im0], img_dict[im1], pair_tie_points)
            db.add_two_view_geometry(img_dict[im0], img_dict[im1], pair_tie_points)
            db.commit()
