import argparse

import cv2
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show an image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("lib", type=str, help="pillow or opencv")
    args = parser.parse_args()

    if args.lib == "opencv":
        image = cv2.imread(args.image_path)
        if image is None:
            print("Failed to load the image")
        else:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.lib == "pillow":
        pil_image = Image.open(args.image_path)
        pil_image.show()
