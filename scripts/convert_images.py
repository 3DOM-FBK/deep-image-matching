import os
import cv2
import argparse
import numpy as np

def convert_images(input_folder, output_folder, target_format, normalize):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif'))]

    for i,image_file in enumerate(image_files):
        # Read the image
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        min_original, max_original = np.min(img), np.max(img)
        if normalize:
            img = (img-min_original) * (255/max_original)
            img = img.astype(np.uint8)
            min, max = np.min(img), np.max(img)

        # Construct the output path with the new format
        output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.' + target_format)

        # Save the image in the target format
        cv2.imwrite(output_path, img)

        print(f"[IMAGE{i}]: {image_path} --> {output_path}")
        if normalize:
            print(f"[NORMALIZE]: min_original:{min_original} max_original:{max_original}  -->  min:{min} max:{max}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert images in a folder to a target image format using OpenCV.")
    parser.add_argument("input_folder", help="Path to the folder containing input images.")
    parser.add_argument("output_folder", help="Path to the folder where converted images will be saved.")
    parser.add_argument("target_format", help="Target image format (e.g., 'png', 'jpg', 'jpeg').")
    parser.add_argument("--normalize", action="store_true", help="Normalize pixel values if set to cover the whole range 0-255.")
    args = parser.parse_args()

    # Call the function to convert images
    convert_images(args.input_folder, args.output_folder, args.target_format, args.normalize)
