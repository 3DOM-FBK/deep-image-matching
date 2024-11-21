import argparse
import os
import cv2

def copy_images_no_exif(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img)
                print(f"Copied {filename} to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Copy images without EXIF data")
    parser.add_argument('input_folder', type=str, help="Folder with input images")
    parser.add_argument('output_folder', type=str, help="Folder to save output images")
    args = parser.parse_args()

    copy_images_no_exif(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()