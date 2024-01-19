import argparse
from pathlib import Path

import cv2
import numpy as np
from deep_image_matching.utils.database import (
    COLMAPDatabase,
    blob_to_array,
    pair_id_to_image_ids,
)


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
                    self.two_views_matches[
                        (int(pair_id[0]), int(pair_id[1]))
                    ] = blob_to_array(data, np.uint32, (-1, 2))

    def ShowMatches(self):
        if self.db_type == "colmap":
            self.ShowColmapMatches()

    def ShowColmapMatches(self):
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

        keypoints0 = self.keypoints[id0]
        keypoints1 = self.keypoints[id1]
        print(f"Img {id0}: kpts shape = {keypoints0.shape}")
        print(f"Img {id1}: kpts shape = {keypoints1.shape}")
        print("raw matches shape", np.shape(self.matches[(id0, id1)]))
        print("verified matches shape", np.shape(self.two_views_matches[(id0, id1)]))

        img0_path = self.imgs_dir / self.imgs[id0]
        img1_path = self.imgs_dir / self.imgs[id1]
        print("img0_path", img0_path)
        print("img1_path", img1_path)

        self.GeneratePlot(
            img0_path,
            img1_path,
            keypoints0,
            keypoints1,
            self.two_views_matches[(id0, id1)],
        )

    def GeneratePlot(
        self,
        img0_path: Path,
        img1_path: Path,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        matches: np.ndarray,
    ):
        # Load images
        img0 = cv2.imread(str(img0_path))
        img1 = cv2.imread(str(img1_path))

        # Convert keypoints to integers
        kpts0_int = np.round(kpts0).astype(int)
        kpts1_int = np.round(kpts1).astype(int)

        # Create a new image to draw matches
        img_matches = np.zeros(
            (max(img0.shape[0], img1.shape[0]), img0.shape[1] + img1.shape[1], 3),
            dtype=np.uint8,
        )
        img_matches[: img0.shape[0], : img0.shape[1]] = img0
        img_matches[: img1.shape[0], img0.shape[1] :] = img1

        # Show keypoints
        for kpt in kpts0_int:
            kpt = tuple(kpt)
            cv2.circle(img_matches, kpt, 3, (0, 0, 255), -1)

        for kpt in kpts1_int:
            kpt = tuple(kpt + np.array([img0.shape[1], 0]))
            cv2.circle(img_matches, kpt, 3, (0, 0, 255), -1)

        # Draw lines and circles for matches
        for match in matches:
            pt1 = tuple(kpts0_int[match[0]])
            pt2 = tuple(np.array(kpts1_int[match[1]]) + np.array([img0.shape[1], 0]))

            # Draw a line connecting the keypoints
            cv2.line(img_matches, pt1, pt2, (0, 255, 0), 2)

            # Draw circles around keypoints
            cv2.circle(img_matches, pt1, 5, (255, 0, 0), -1)
            cv2.circle(img_matches, pt2, 5, (255, 0, 0), -1)

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
        "-i",
        "--images",
        type=str,
        help="Images IDs or names. E.g.: 'img1.jpg img2.jpg' or '37 98'. Max two images.",
        required=True,
    )
    parser.add_argument(
        "-t", "--type", type=str, choices=["names", "ids"], required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to output folder", required=True
    )
    parser.add_argument(
        "-m", "--max_size", type=int, help="Max size of the output image showing matches", required=False, default=1500,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    database_path = Path(args.database)
    out_file = Path(args.output)
    imgs_dir = Path(args.imgsdir)
    max_size = args.max_size
    print(f"database path: {database_path}")
    print(f"output dir: {out_file}")

    i1, i2 = args.images.split()
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
    show_pair_matches.ShowMatches()


if __name__ == "__main__":
    main()
