import argparse
import os
from pathlib import Path

import cv2
import exifread

ORIENTATION_MAP = {
    "Horizontal (normal)": 0,
    "Rotated 180": 180,
    "Rotated 90 CW": 90,
    "Rotated 90 CCW": 270,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and output folder")
    parser.add_argument(
        "images", type=str, help="path to the input folder containing images"
    )
    parser.add_argument("output_folder", type=str, help="path to the output folder")
    args = parser.parse_args()

    input_folder = Path(args.images)
    output_folder = Path(args.output_folder)

    with open(str(output_folder / "rotations.txt"), "w") as output_file:
        for img in os.listdir(input_folder):
            with open(str(input_folder / img), "rb") as image_file:
                tags = exifread.process_file(image_file)
                orientation_tag = "Image Orientation"

                if orientation_tag in tags:
                    orientation_description = str(tags[orientation_tag])
                    orientation_degrees = ORIENTATION_MAP.get(orientation_description)
                    output_file.write(f"{img} {orientation_degrees}\n")
                    image = cv2.imread(str(input_folder / img))
                    cv2.imwrite(str(output_folder / img), image)
