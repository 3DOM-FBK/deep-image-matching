import os
import cv2
import argparse
import numpy as np
import rasterio
from PIL import Image


def load_img(
    path_to_img: str,
    reader: str,
):
    """
    Load an image from the given path using the specified reader.

    Args:
        path_to_img (str): Path to the image file.
        reader (str): The image reader to use. Supported options are "rasterio", "opencv", and "pillow".

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    if reader == "rasterio":
        with rasterio.open(path_to_img) as dataset:
            img = dataset.read(1)

    elif reader == "opencv":
        img = cv2.imread(path_to_img, cv2.IMREAD_ANYDEPTH)

    elif reader == "pillow":
        img = Image.open(path_to_img)
        img = img.convert("L")
        img = np.array(img)

    return img


def convert_images(input_folder, output_folder, target_format, normalize, method, reader):
    """
    Convert images in a folder to a target image format using OpenCV.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where converted images will be saved.
        target_format (str): Target image format (e.g., 'png', 'jpg', 'jpeg', 'tiff').
        normalize (bool): Flag indicating whether to normalize pixel values to cover the whole range 0-255.
        method (str): Normalization method. Supported options are "single_image" and "image_set".
        reader (str): Library to read the images. Supported options are "opencv", "pillow", and "rasterio".
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
                ".pgm",
            )
        )
    ]
    if len(image_files) == 0:
        print(
            "No images in the folder. Check that the path to the folder is correct or that the format is supported."
        )
        quit()

    # Get min and max values of all images
    MIN = []
    MAX = []

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = load_img(image_path, reader)

        MIN.append(np.min(img))
        MAX.append(np.max(img))

    abs_min = np.min(MIN)
    abs_max = np.max(MAX)

    for i, image_file in enumerate(image_files):
        print(f"[IMAGE{i}]")
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(
            output_folder, os.path.splitext(image_file)[0] + "." + target_format
        )
        img = load_img(image_path, reader)
        min_original, max_original = np.min(img), np.max(img)
        print(f"Range: min_original {min_original}  max_original {max_original}")

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
        "target_format", help="Target image format (e.g., 'png', 'jpg', 'jpeg', 'tiff')."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize pixel values if set to cover the whole range 0-255.",
    )
    parser.add_argument(
        "--method",
        choices=["single_image", "image_set"],
        help="Normalize respect max and min of each single image or respect max and min of the entire image set.",
    )
    parser.add_argument(
        "--reader",
        choices=["opencv", "pillow", "rasterio"],
        default="pillow",
        help="Library to read the images. Default is 'pillow'.",
    )
    args = parser.parse_args()

    convert_images(
        args.input_folder,
        args.output_folder,
        args.target_format,
        args.normalize,
        args.method,
        args.reader,
    )
