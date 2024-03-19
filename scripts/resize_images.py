import os
import argparse
from PIL import Image


def resize_images(input_folder, output_folder, new_resolution):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png", ".gif")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open image and resize
            with Image.open(input_path) as img:
                resized_img = img.resize(new_resolution, Image.Resampling.LANCZOS)

                # Save the resized image to the output folder
                resized_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Resize images in a folder.")
    parser.add_argument("input_folder", help="Path to the input image folder")
    parser.add_argument(
        "output_folder", help="Path to the output folder for resized images"
    )
    parser.add_argument("width", type=int, help="New width for the images")
    parser.add_argument("height", type=int, help="New height for the images")

    args = parser.parse_args()

    new_resolution = (args.width, args.height)

    resize_images(args.input_folder, args.output_folder, new_resolution)
    print("Image resizing complete!")


if __name__ == "__main__":
    main()
