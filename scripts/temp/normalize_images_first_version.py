import os
import cv2
import argparse
import numpy as np

from pathlib import Path

def img_gradient(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_gradient

def normalize_image(image_path, output_path, patch_size=300, method="minmax"):
    # Read the image in grayscale mode
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")

    if method == "gradient":
        gradient_magnitude = img_gradient(image)
        cv2.imwrite(str(output_path), gradient_magnitude)
        return

    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Create an empty array to store the normalized image
    normalized_image = np.zeros_like(image, dtype=np.float32)
    
    # Define the half patch size
    half_patch_size = patch_size // 2
    
    # Iterate over each pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Define the boundaries of the patch
            r_min = max(0, i - half_patch_size)
            r_max = min(rows, i + half_patch_size + 1)
            c_min = max(0, j - half_patch_size)
            c_max = min(cols, j + half_patch_size + 1)
            
            # Extract the patch
            patch = image[r_min:r_max, c_min:c_max]
            
            # Calculate the mean and standard deviation of the patch
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            #masked_patch = np.ma.masked_outside(patch, patch_mean - 2 * patch_std, patch_mean + 2 * patch_std)
            #robust_mean = np.mean(masked_patch)
            #robust_std = np.std(masked_patch)
            min_value = np.min(patch)
            max_value = np.max(patch)
            
            ## Normalize the pixel with respect to the patch
            #if patch_std > 0:
            #    normalized_image[i, j] = (image[i, j] - patch_mean) / patch_std
            #else:
            #    normalized_image[i, j] = 0  # If the standard deviation is 0, set the normalized value to 0

            #norm_value = int((max_value-min_value)*image[i,j]/255)

            if method == "minmax":
                try:
                    norm_value = (image[i,j]-min_value)*255/(max_value-min_value)
                    if norm_value > 0 and norm_value < 256:
                        normalized_image[i, j] = norm_value
                    else:
                        normalized_image[i, j] = 0
                except:
                    normalized_image[i, j] = 0
            elif method == "robust":
                norm_value = (image[i,j]-patch_mean-patch_std)*255/(2*patch_std)
                if norm_value > 0 and norm_value < 256:
                    normalized_image[i, j] = norm_value
                else:
                    normalized_image[i, j] = 0
    
    # Scale the normalized image to the range [0, 255]
    #normalized_image = cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert the normalized image to uint8 type
    normalized_image = normalized_image.astype(np.uint8)
    
    # Save the normalized image
    cv2.imwrite(str(output_path), normalized_image)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize an image")
    parser.add_argument("images", type=Path, help="Path to the input image")
    parser.add_argument("output", type=Path, help="Path to save the normalized image")
    parser.add_argument("--patch_size", type=int, default=30, help="Size of the patch (default: 30)")
    args = parser.parse_args()
    
    images = os.listdir(args.images)
    print(images)

    for img in images:
        normalize_image(args.images / img, args.output / img, args.patch_size)