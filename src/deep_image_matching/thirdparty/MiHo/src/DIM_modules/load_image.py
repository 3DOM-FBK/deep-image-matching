import cv2
import numpy as np

def load_image(
        path_to_img: str,
        grayscale: bool,
        as_float: bool,
) -> np.ndarray:
    image = cv2.imread(path_to_img)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if as_float:
        image = image.astype(np.float32)
    return image   
