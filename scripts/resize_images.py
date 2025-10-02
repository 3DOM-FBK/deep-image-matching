import argparse
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import zoom

def resize_images(input_folder, output_folder, new_resolution):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", 
                             ".gif", ".GIF", ".tif", ".TIF", ".tiff", ".TIFF")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Open image with rasterio
                with rasterio.open(input_path) as src:
                    # Get original dimensions and data
                    original_height, original_width = src.height, src.width
                    original_transform = src.transform
                    original_crs = src.crs
                    original_profile = src.profile.copy()
                    
                    print(f"Processing {filename}: {original_width}x{original_height} -> {new_resolution[0]}x{new_resolution[1]}")
                    
                    # Read the image data
                    data = src.read()
                    
                    # For images without geospatial information, use simple numpy resize
                    if original_crs is None or original_transform == rasterio.Affine.identity():
                        print(f"  No geospatial info found, using simple resize for {filename}")
                        
                        # Calculate zoom factors
                        zoom_y = new_resolution[1] / original_height
                        zoom_x = new_resolution[0] / original_width
                        
                        # Resize each band
                        resized_data = np.zeros((src.count, new_resolution[1], new_resolution[0]), 
                                              dtype=src.dtypes[0])
                        
                        for band_idx in range(src.count):
                            resized_data[band_idx] = zoom(data[band_idx], (zoom_y, zoom_x), order=3)  # order=3 is cubic (like Lanczos)
                        
                        # Update profile for simple image
                        new_profile = original_profile
                        new_profile.update({
                            'height': new_resolution[1],
                            'width': new_resolution[0],
                            'transform': rasterio.Affine.identity(),
                            'crs': None
                        })
                    
                    else:
                        # Handle georeferenced images with proper reprojection
                        print(f"  Georeferenced image, using proper reprojection for {filename}")
                        
                        # Calculate scale factors
                        scale_x = new_resolution[0] / original_width
                        scale_y = new_resolution[1] / original_height
                        
                        # Create new transform for the resized image
                        new_transform = rasterio.Affine(
                            original_transform.a / scale_x,  # pixel width
                            original_transform.b,            # rotation
                            original_transform.c,            # x offset
                            original_transform.d,            # rotation
                            original_transform.e / scale_y,  # pixel height (negative)
                            original_transform.f             # y offset
                        )
                        
                        # Update profile for output
                        new_profile = original_profile
                        new_profile.update({
                            'height': new_resolution[1],
                            'width': new_resolution[0],
                            'transform': new_transform
                        })
                        
                        # Create output array
                        resized_data = np.zeros((src.count, new_resolution[1], new_resolution[0]), 
                                              dtype=src.dtypes[0])
                        
                        # Reproject/resample each band
                        for band_idx in range(src.count):
                            reproject(
                                source=data[band_idx],
                                destination=resized_data[band_idx],
                                src_transform=original_transform,
                                src_crs=original_crs,
                                dst_transform=new_transform,
                                dst_crs=original_crs,
                                resampling=Resampling.lanczos
                            )
                    
                    # Write the resized image
                    with rasterio.open(output_path, 'w', **new_profile) as dst:
                        dst.write(resized_data)
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue


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
