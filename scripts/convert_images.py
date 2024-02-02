import os
import cv2
import argparse
import numpy as np

from PIL import Image
from deep_image_matching import logger


def load_img(
    path_to_img: str,
):
    img = cv2.imread(path_to_img, cv2.IMREAD_ANYDEPTH)
    if img.all() == None:
        logger.warning(
            "It was not possible to load the image with opencv module. Trying with pillow .."
        )
        img = Image.open(path_to_img)
        logger.info(f"Image mode: {img.mode}")
        img = np.array(img)
        logger.info(f"Image mode: {img.dtype}")
    else:
        logger.info(f"Image mode: {img.dtype}")

    return img


def convert_images(input_folder, output_folder, target_format, normalize, method):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [
        f
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
        and not f.startswith(".")
        and f.lower().endswith(
            (
                ".png",
                ".PNG",
                ".jpg",
                ".JPG",
                ".jpeg",
                ".JPEG",
                ".bmp",
                ".BMP",
                ".tif",
                ".TIF",
                ".tiff",
                ".TIFF",
                ".gif",
                ".GIF",
            )
        )
    ]
    if len(image_files) == 0:
        logger.error(
            "No images in the folder. Check that the path to the folder is correct or that the format is supported."
        )
        quit()

    # Get min and max values of all images
    MIN = []
    MAX = []

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = load_img(image_path)

        MIN.append(np.min(img))
        MAX.append(np.max(img))

    abs_min = np.min(MIN)
    abs_max = np.max(MAX)

    for i, image_file in enumerate(image_files):
        # Read the image
        logger.info(f"[IMAGE{i}]")
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(
            output_folder, os.path.splitext(image_file)[0] + "." + target_format
        )
        img = load_img(image_path)
        min_original, max_original = np.min(img), np.max(img)
        logger.info(f"Range: min_original {min_original}  max_original {max_original}")

        if not normalize:
            if img.dtype == np.uint8:
                print("Input image 8-bit - no conversions")
                cv2.imwrite(output_path, img)
            elif img.dtype == np.uint16:
                print("Input image 16-bit - converted to 8-bit")
                img = img.astype(np.float32)
                img = img * 255 / 65535
                img = img.astype(np.uint8)
                cv2.imwrite(output_path, img)

        elif normalize and method == "image_set":
            print(
                "Normalizing respect max and min value of the image set - output in 8-bit"
            )
            img = (img - abs_min) * (255 / (abs_max - abs_min))
            img = img.astype(np.uint8)
            cv2.imwrite(output_path, img)

        elif normalize and method == "single_image":
            print(
                "Normalizing respect max and min value of each image - output in 8-bit"
            )
            min = np.min(img)
            max = np.max(img)
            img = (img - min) * (255 / (max - min))
            img = img.astype(np.uint8)
            cv2.imwrite(output_path, img)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert images in a folder to a target image format using OpenCV."
    )
    parser.add_argument(
        "input_folder", help="Path to the folder containing input images."
    )
    parser.add_argument(
        "output_folder", help="Path to the folder where converted images will be saved."
    )
    parser.add_argument(
        "target_format", help="Target image format (e.g., 'png', 'jpg', 'jpeg')."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize pixel values if set to cover the whole range 0-255.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["single_image", "image_set"],
        help="Normalize respect max and min of each single image or respect max and min of the entire image set.",
    )
    args = parser.parse_args()

    # Call the function to convert images
    convert_images(
        args.input_folder,
        args.output_folder,
        args.target_format,
        args.normalize,
        args.method,
    )
