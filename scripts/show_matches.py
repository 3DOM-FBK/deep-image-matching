import argparse
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import rasterio

from deep_image_matching.utils.database import (
    COLMAPDatabase,
    blob_to_array,
    pair_id_to_image_ids,
)


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def ExportMatches(
    database_path: Path,
    min_num_matches: int = 1,
) -> None:
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    pairs = []

    cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
    for row in cursor:
        pair_id = row[0]
        n_matches = row[1]
        id_img1, id_img2 = pair_id_to_image_ids(pair_id)
        id_img1, id_img2 = int(id_img1), int(id_img2)
        # img1 = images[id_img1]
        # img2 = images[id_img2]
        if n_matches >= min_num_matches:
            pairs.append((id_img1, id_img2))

    connection.close()

    return pairs


def generate_pairs(imgs_dir, method=["bruteforce", "custom"], database_path=Path("./")):
    pairs = []
    if method == "custom":
        n_images = len(os.listdir(imgs_dir))
        for i in range(n_images - 1):
            if i % 2 == 0:
                pairs.append((i + 1, i + 2))
    elif method == "bruteforce":
        pairs = ExportMatches(database_path)
    return pairs


class ShowPairMatches:
    def __init__(
        self,
        database_path: Path,
        imgs_dict: dict,
        imgs_dir: Path,
        out_file: Path,
        max_size: int,
    ):
        self.db_path = database_path
        self.imgs_dict = imgs_dict
        self.imgs_dir = imgs_dir
        self.out_file = out_file
        self.max_out_img_size = max_size
        self.imgs = {}
        self.keypoints = {}
        self.matches = {}
        self.two_views_matches = {}

        if self.db_path.suffix == ".db":
            self.db_type = "colmap"
        else:
            print("Error. Not supported database extension. Quit")
            quit()

    def LoadDatabase(self):
        print("Loading database..")
        if self.db_type == "colmap":
            db = COLMAPDatabase.connect(self.db_path)
            self.imgs = dict(
                (image_id, name)
                for image_id, name in db.execute("SELECT image_id, name FROM images")
            )
            self.keypoints = dict(
                (image_id, blob_to_array(data, np.float32, (-1, 2)))
                for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
            )
            for pair_id, r, c, data in db.execute(
                "SELECT pair_id, rows, cols, data FROM matches"
            ):
                if data is not None:
                    pair_id = pair_id_to_image_ids(pair_id)
                    self.matches[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(
                        data, np.uint32, (-1, 2)
                    )
            for pair_id, r, c, data in db.execute(
                "SELECT pair_id, rows, cols, data FROM two_view_geometries"
            ):
                if data is not None:
                    pair_id = pair_id_to_image_ids(pair_id)
                    self.two_views_matches[(int(pair_id[0]), int(pair_id[1]))] = (
                        blob_to_array(data, np.uint32, (-1, 2))
                    )

    def ShowMatches(self, plot_config: dict):
        if self.db_type == "colmap":
            self.ShowColmapMatches(plot_config)

    def ShowColmapMatches(self, plot_config: dict):
        print("Showing matches..")
        if self.imgs_dict["type"] == "ids":
            id0 = int(self.imgs_dict["data"][0])
            id1 = int(self.imgs_dict["data"][1])

        elif self.imgs_dict["type"] == "names":
            inverted_dict = {v: k for k, v in self.imgs.items()}
            im0 = self.imgs_dict["data"][0]
            id0 = inverted_dict[im0]
            im1 = self.imgs_dict["data"][1]
            id1 = inverted_dict[im1]

        keypoints0 = self.keypoints[id0][:, :2]
        keypoints1 = self.keypoints[id1][:, :2]
        print(f"Img {id0}: kpts shape = {keypoints0.shape}")
        print(f"Img {id1}: kpts shape = {keypoints1.shape}")
        try:
            print("raw matches shape", np.shape(self.matches[(id0, id1)]))
        except:
            pass
        try:
            print(
                "verified matches shape", np.shape(self.two_views_matches[(id0, id1)])
            )
        except:
            self.two_views_matches[(id0, id1)] = []

        img0_path = self.imgs_dir / self.imgs[id0]
        img1_path = self.imgs_dir / self.imgs[id1]
        print("img0_path", img0_path)
        print("img1_path", img1_path)

        self.GeneratePlot(
            img0_path,
            img1_path,
            keypoints0,
            keypoints1,
            self.matches[(id0, id1)], #"self.two_views_matches[(id0, id1)]," or "self.matches[(id0, id1)],"
            plot_config,
        )

    def GeneratePlot(
        self,
        img0_path: Path,
        img1_path: Path,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        matches: np.ndarray,
        plot_config: dict,
    ):
        show_keypoints = plot_config["show_keypoints"]
        radius = plot_config["radius"]
        thickness = plot_config["thickness"]
        space_between_images = plot_config["space_between_images"]

        # Load images using rasterio
        def load_image_with_rasterio(img_path):
            with rasterio.open(str(img_path)) as src:
                img_data = src.read()
                # Convert from (bands, rows, cols) to (rows, cols, bands)
                img = np.transpose(img_data, (1, 2, 0))
                
                # Handle different number of bands
                if img.shape[2] == 1:
                    # Single band - convert to 3-channel grayscale
                    img = np.repeat(img, 3, axis=2)
                elif img.shape[2] > 3:
                    # More than 3 bands - take first 3 (typically RGB)
                    img = img[:, :, :3]
                
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    # Normalize to 0-255 range if values are in different range
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                
                ## Convert RGB to BGR for OpenCV compatibility (if 3 channels)
                #if img.shape[2] == 3:
                #    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                return img
        
        img0 = load_image_with_rasterio(img0_path)
        img1 = load_image_with_rasterio(img1_path)

        # Filter out invalid keypoints (NaN, inf) and convert to integers
        valid_mask0 = np.isfinite(kpts0).all(axis=1)
        valid_mask1 = np.isfinite(kpts1).all(axis=1)
        
        kpts0_valid = kpts0[valid_mask0]
        kpts1_valid = kpts1[valid_mask1]
        
        kpts0_int = np.round(kpts0_valid).astype(int)
        kpts1_int = np.round(kpts1_valid).astype(int)
        
        print(f"Valid keypoints - Img0: {len(kpts0_int)}/{len(kpts0)}, Img1: {len(kpts1_int)}/{len(kpts1)}")

        # Create a new image to draw matches
        img_matches = np.zeros(
            (
                max(img0.shape[0], img1.shape[0]),
                img0.shape[1] + img1.shape[1] + space_between_images,
                3,
            ),
            dtype=np.uint8,
        )
        img_matches[: img0.shape[0], : img0.shape[1]] = img0
        img_matches[: img1.shape[0], img0.shape[1] + space_between_images :] = img1
        img_matches[
            : img1.shape[0], img0.shape[1] : img0.shape[1] + space_between_images
        ] = (255, 255, 255)

        if show_keypoints:
            # Show valid keypoints within image bounds
            for kpt in kpts0_int:
                if 0 <= kpt[0] < img0.shape[1] and 0 <= kpt[1] < img0.shape[0]:
                    kpt_tuple = tuple(kpt)
                    cv2.circle(img_matches, kpt_tuple, radius, (0, 0, 255), thickness)

            for kpt in kpts1_int:
                kpt_shifted = kpt + np.array([img0.shape[1] + space_between_images, 0])
                if (0 <= kpt[0] < img1.shape[1] and 0 <= kpt[1] < img1.shape[0] and
                    0 <= kpt_shifted[0] < img_matches.shape[1] and 0 <= kpt_shifted[1] < img_matches.shape[0]):
                    kpt_tuple = tuple(kpt_shifted)
                    cv2.circle(img_matches, kpt_tuple, radius, (0, 0, 255), thickness)

        # Filter matches to only include those with valid keypoints and within image bounds
        valid_matches = []
        
        # Create mapping from original indices to filtered indices
        valid_idx0_map = {}
        valid_idx1_map = {}
        
        for i, is_valid in enumerate(valid_mask0):
            if is_valid:
                valid_idx0_map[i] = len(valid_idx0_map)
                
        for i, is_valid in enumerate(valid_mask1):
            if is_valid:
                valid_idx1_map[i] = len(valid_idx1_map)
        
        for match in matches:
            orig_idx0, orig_idx1 = match[0], match[1]
            
            # Check if both original indices had valid keypoints
            if orig_idx0 in valid_idx0_map and orig_idx1 in valid_idx1_map:
                new_idx0 = valid_idx0_map[orig_idx0]
                new_idx1 = valid_idx1_map[orig_idx1]
                
                # Check if indices are within filtered arrays
                if new_idx0 < len(kpts0_int) and new_idx1 < len(kpts1_int):
                    kpt0 = kpts0_int[new_idx0]
                    kpt1 = kpts1_int[new_idx1]
                    
                    # Check if keypoints are within image bounds
                    if (0 <= kpt0[0] < img0.shape[1] and 0 <= kpt0[1] < img0.shape[0] and
                        0 <= kpt1[0] < img1.shape[1] and 0 <= kpt1[1] < img1.shape[0]):
                        valid_matches.append((new_idx0, new_idx1))
        
        print(f"Valid matches: {len(valid_matches)}/{len(matches)}")
        
        # Draw lines and circles for valid matches
        for idx0, idx1 in valid_matches:
            pt1 = tuple(kpts0_int[idx0])
            pt2 = tuple(kpts1_int[idx1] + np.array([img0.shape[1] + space_between_images, 0]))

            # Draw a line connecting the keypoints
            cv2.line(img_matches, pt1, pt2, (0, 255, 0), thickness)

            # Draw circles around keypoints
            cv2.circle(img_matches, pt1, radius, (255, 0, 0), thickness)
            cv2.circle(img_matches, pt2, radius, (255, 0, 0), thickness)

        img_matches_resized = self.resize_image(img_matches, self.max_out_img_size)

        ## Show the image with matches
        # cv2.imshow(f"Verified matches   {img0_path.name} - {img1_path.name}", img_matches_resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(str(self.out_file), img_matches_resized)

    def resize_image(self, img, max_side_length):
        height, width = img.shape[:2]

        if max(height, width) > max_side_length:
            scale_factor = max_side_length / max(height, width)
            img_resized = cv2.resize(
                img, (int(width * scale_factor), int(height * scale_factor))
            )
            return img_resized
        else:
            return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show matches from COLMAP database.   'python ./show_matches.py -d assets/output/database.db -i 'DSC_6466.JPG DSC_6468.JPG' -t names -o . -f assets/example_cyprus/' or 'python ./show_matches.py -d assets/output/database.db -i '1 2' -t ids -o . -f assets/example_cyprus/'"
    )

    parser.add_argument(
        "-d", "--database", type=str, help="Path to COLMAP database", required=True
    )
    parser.add_argument(
        "-f", "--imgsdir", type=str, help="Path to images directory", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to output folder", required=True
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export matches for all pairs",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        help="Images IDs or names. E.g.: 'img1.jpg img2.jpg' or '37 98'. Max two images.",
        required=False,
    )
    parser.add_argument(
        "-t", "--type", type=str, choices=["names", "ids"], required=False
    )
    parser.add_argument(
        "-m",
        "--max_size",
        type=int,
        help="Max size of the output image showing matches",
        required=False,
        default=1500,
    )
    args = parser.parse_args()

    return args


def main():
    plot_config = {
        "show_keypoints": True,
        "radius": 5,
        "thickness": 2,
        "space_between_images": 0,
    }

    args = parse_args()
    database_path = Path(args.database)
    out_dir = Path(args.output)
    imgs_dir = Path(args.imgsdir)
    max_size = args.max_size

    print(f"database path: {database_path}")
    print(f"output dir: {out_dir}")

    if args.all == False:
        i1, i2 = args.images.split()
        out_file = out_dir / f"{i1}_{i2}.png"
        imgs = {
            "type": args.type,
            "data": (i1, i2),
        }
        print("images: ", imgs)

        show_pair_matches = ShowPairMatches(
            database_path=database_path,
            imgs_dict=imgs,
            imgs_dir=imgs_dir,
            out_file=out_file,
            max_size=max_size,
        )

        show_pair_matches.LoadDatabase()
        show_pair_matches.ShowMatches(plot_config)

    else:
        pairs = generate_pairs(
            imgs_dir, method="bruteforce", database_path=database_path
        )
        print(pairs)
        for pair in pairs:
            i1, i2 = pair[0], pair[1]
            out_file = out_dir / f"{i1}_{i2}.png"
            imgs = {
                "type": "ids",
                "data": (i1, i2),
            }
            print("images: ", imgs)

            show_pair_matches = ShowPairMatches(
                database_path=database_path,
                imgs_dict=imgs,
                imgs_dir=imgs_dir,
                out_file=out_file,
                max_size=max_size,
            )

            show_pair_matches.LoadDatabase()
            # show_pair_matches.ShowMatches(plot_config);quit()

            try:
                show_pair_matches.ShowMatches(plot_config)
            except:
                print("No verified matches found")


if __name__ == "__main__":
    main()
