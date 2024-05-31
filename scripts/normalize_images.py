import os
import cv2
import argparse
import numpy as np

from pathlib import Path


class ImageNormalizer:
    """
    A class for normalizing images using different methods.
    """

    def __init__(
            self, 
            reduce_noise: bool = True,
            noise_kernel: tuple = (5, 5),
            ):
        """
        Initialize the ImageNormalizer object.

        Args:
            reduce_noise (bool): Whether to apply noise reduction. Default is True.
            noise_kernel (tuple): The size of the noise reduction kernel. Default is (5, 5).
        """
        self.reduce_noise = reduce_noise
        self.noise_kernel = noise_kernel

    def img_gradient(
            self,
            image_path: Path,
            output_path: Path,
            ksize: int = 3,
            ):
        """
        Export the gradient of the image.

        Args:
            image_path (Path): The path to the input image.
            output_path (Path): The path to save the normalized image.
            ksize (int): The size of the Sobel kernel. Default is 3.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")
        if self.reduce_noise:
            image = cv2.GaussianBlur(image, self.noise_kernel, 0)
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        normalized_gradient = cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
        )
        cv2.imwrite(str(output_path), normalized_gradient)
    
    def normalize_kernel(
            self,
            image_path: Path,
            output_path: Path,
            patch_size: int = 30,
            method: str = "minmax",
            ):
        """
        Normalize an image using kernel method.

        Args:
            image_path (Path): The path to the input image.
            output_path (Path): The path to save the normalized image.
            patch_size (int): The size of the patch. Default is 30.
            method (str): The normalization method to use. Default is "minmax".
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")
        if self.reduce_noise:
            image = cv2.GaussianBlur(image, self.noise_kernel, 0)

        rows, cols = image.shape
        normalized_image = np.zeros_like(image, dtype=np.float32)
        half_patch_size = patch_size // 2

        for i in range(rows):
            for j in range(cols):
                # Define the boundaries of the patch
                r_min = max(0, i - half_patch_size)
                r_max = min(rows, i + half_patch_size + 1)
                c_min = max(0, j - half_patch_size)
                c_max = min(cols, j + half_patch_size + 1)

                # Extract the patch
                patch = image[r_min:r_max, c_min:c_max]

                if method == "minmax":
                    min_value = np.min(patch)
                    max_value = np.max(patch)
                    norm_value = (image[i, j] - min_value) * 255 / (max_value - min_value)
                    if norm_value > 0 and norm_value < 256:
                        normalized_image[i, j] = norm_value
                    else:
                        normalized_image[i, j] = 0

                elif method == "meanstd":
                    patch_mean = np.mean(patch)
                    patch_std = np.std(patch)
                    # masked_patch = np.ma.masked_outside(patch, patch_mean - 2 * patch_std, patch_mean + 2 * patch_std)
                    # robust_mean = np.mean(masked_patch)
                    # robust_std = np.std(masked_patch)
                    norm_value = (
                        (image[i, j] - patch_mean - patch_std) * 255 / (2 * patch_std)
                    )
                    if norm_value > 0 and norm_value < 256:
                        normalized_image[i, j] = norm_value
                    else:
                        normalized_image[i, j] = 0

        normalized_image = normalized_image.astype(np.uint8)
        cv2.imwrite(str(output_path), normalized_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize an image")
    parser.add_argument("images", type=Path, help="Path to image folder to normalize")
    parser.add_argument(
        "output", type=Path, help="Path to output folder to save normalized images"
    )
    parser.add_argument(
        "--patch_size", type=int, default=30, help="Size of the patch (default: 30)"
    )
    parser.add_argument(
        "--reduce_noise", type=bool, default=True, help="Apply noise reduction (default: True)"
    )
    parser.add_argument(
        "--noise_kernel", type=tuple, default=(5, 5), help="Apply noise reduction (default: True)"
    )
    parser.add_argument(
        "--method", type=str, choices=["norm_kernel_minmax", "norm_kernel_meanstd", "gradient"], default=(5, 5), help="Apply noise reduction (default: True)"
    )
    args = parser.parse_args()

    images = os.listdir(args.images)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    image_normalizer = ImageNormalizer(
        reduce_noise = args.reduce_noise,
        noise_kernel = args.noise_kernel,
        )

    for img in images:
        if args.method == "norm_kernel_minmax":
            image_normalizer.normalize_kernel(
                args.images / img, 
                args.output / img, 
                patch_size=args.patch_size,
                method="minmax",
                )
    
        elif args.method == "norm_kernel_meanstd":
            image_normalizer.normalize_kernel(
                args.images / img, 
                args.output / img, 
                patch_size=args.patch_size,
                method="meanstd",
                )
        
        elif args.method == "gradient":
            image_normalizer.img_gradient(
                args.images / img, 
                args.output / img, 
                ksize=3,
                )