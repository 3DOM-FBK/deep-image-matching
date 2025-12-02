import os
import cv2
import argparse
import pycolmap
import numpy as np

from pathlib import Path


FOCAL = 1000

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Extract RootSIFT features with pycolmap"
    )
    parser.add_argument(
        "work_dir", help="Path to the folder containing input folder 'images'.", type=Path,
    )
    args = parser.parse_args()
    work_dir = args.work_dir

    return work_dir

if __name__ == "__main__":
    
    work_dir = ParseArguments()
    print(f"Working dir: {work_dir}")

    #database_path = work_dir / "database.db"
    #if os.path.exists(database_path):
    #    database_path.unlink()
#
    #db = pycolmap.Database(database_path)

    image_dir = work_dir / "images"
    images = os.listdir(image_dir)

    for i,img in enumerate(images):
        image_cv = cv2.imread(str(image_dir / img), cv2.IMREAD_GRAYSCALE)
        height, width = image_cv.shape

        #camera = pycolmap.Camera(
        #    camera_id=1,
        #    model=pycolmap.CameraModelId.SIMPLE_RADIAL,
        #    #focal_length=1,
        #    width=2048,
        #    height=1000,
        #    params=[6586.0, 1024.0, 500.0, 0.0],
        #)
        #db.write_camera(camera, use_camera_id=False)


        #image = pycolmap.Image(
        #    name=img,
        #    points2D=pycolmap.Point2DList(np.empty((0, 2), dtype=np.float64)),
        #    #cam_from_world=pycolmap.Rigid3d(rotation=pycolmap.Rotation3d([0, 0, 0, 1]), translation=[0, 0, 0]),
        #    camera_id=i+1,
        #    image_id=i,
        #    )
#
        #db.write_image(image)

        print(dir(pycolmap))
        options = pycolmap.SiftExtractionOptions()
        options.max_image_size = 3200
        options.use_gpu = True
        options.max_num_features = 8192
        options.peak_threshold = 0.00666 # 0.00666

        print(dir(options))

        sift = pycolmap.Sift(options)
        kpts = sift.extract(image_cv)
        print(kpts[0].shape)
        print(kpts[1].shape)
        quit()