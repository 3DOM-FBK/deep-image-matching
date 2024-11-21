import sqlite3
import argparse
import pycolmap
import random
import rasterio
import numpy as np
from typing import List, Tuple

from tqdm import tqdm
from pathlib import Path
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import zoom

def ProjectionsOfTriangulatedTiePoints(
        model_path: Path,
    ) -> None:

    image_projections = {}
    reconstruction = pycolmap.Reconstruction(model_path)
 
    for image in tqdm(reconstruction.images.values(), desc="Find projections that have a corresponding 3D tie points"):
        camera_id = image.camera_id
        camera = reconstruction.cameras[camera_id]
        width = camera.width
        height = camera.height

        projections = []
        for feature_idx, point2D in enumerate(image.points2D):
            #if point2D.has_point3D:
            if point2D.point3D_id < 1000000:
                #print(point2D.point3D_id);quit()
                #print(point2D.has_point3D.__getattribute__)
                #print(type(point2D.has_point3D))
                #quit()
                x, y = point2D.xy
                x, y = x, (height-y)
                projections.append((image.image_id, (x,y), width, height))
        image_projections[image] = projections

    return image_projections

def ComupteMetrics(
        output_dir: Path,
        image_name: str,
        projections: List[Tuple],
        scale_max_value: int = 10,
        resolution: int = 30,
        upsample_factor: int = 10,
        ) -> None:  
    firs_prj = projections[0]
    width, height = firs_prj[2], firs_prj[3]
    x_points = np.array([prj[1][0] for prj in projections])
    y_points = np.array([prj[1][1] for prj in projections])
    xmin, ymin, xmax, ymax = 0, 0, width, height  # Rectangle bounds (local system)
    resolution = resolution  # Grid cell size in the same units as your coordinates

    # Create a grid for point density
    cols = int((xmax - xmin) / resolution)
    rows = int((ymax - ymin) / resolution)
    density_grid = np.zeros((rows, cols), dtype=np.float32)

    # Assign points to grid cells
    for x, y in zip(x_points, y_points):
        col = int((x - xmin) / resolution)
        row = int((ymax - y) / resolution)  # Reverse y-axis for raster orientation
        if 0 <= row < rows and 0 <= col < cols:  # Ensure points are within bounds
            density_grid[row, col] += 1

    # Normalize density values for better visualization (optional)
    density_grid_normalized = ((density_grid / scale_max_value) != 0).astype(int)
    density_grid_normalized = (density_grid_normalized * 255).astype(np.uint8)  # Scale to [0, 255]

    # Define raster transformation for local reference
    transform = from_origin(xmin, ymax, resolution, resolution)

    zero_cells = np.count_nonzero(density_grid == 0)
    print(f"Image: {image_name}")
    print(f"Number of cells with zero density: {zero_cells}")
    print(f"Total number of cells: {rows * cols}")
    covered_area = ((rows * cols)-zero_cells) / (rows * cols) * 100
    print(f"Percentage of cells with non zero density: {covered_area}%")

    density_grid_normalized = zoom(
        density_grid_normalized, 
        zoom=upsample_factor, 
        order=1  # Cubic interpolation (you can use order=1 for linear interpolation)
    )

    # Save the density map as a JPEG
    with rasterio.open(
        f'{output_dir}/density_{image_name}.jpg',
        "w",
        driver="JPEG",  # Use JPEG driver
        height=density_grid_normalized.shape[0],
        width=density_grid_normalized.shape[1],
        count=1,
        dtype=density_grid_normalized.dtype,
        transform=transform
    ) as dst:
        dst.write(density_grid_normalized, 1)
    
    return image_name, covered_area

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(
        description="Export pairs in a txt file from a COLMAP database."
    )
    parser.add_argument("-d", "--database", type=Path, required=True, help="Path to COLMAP database.")
    parser.add_argument("-m", "--min_n_matches", type=int, required=True, help="Min number of matches that a pair should have after geometric verification.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to output folder.")
    parser.add_argument("-l", "--model", type=Path, required=True, help="Path to model folder.")
    args = parser.parse_args()

    image_projections = ProjectionsOfTriangulatedTiePoints(model_path=args.model)

    results = {}

    for image in tqdm(image_projections, desc="Compute metrics"):
        #print(f"Image: {image.name} - Number of projections: {len(image_projections[image])}")
        image_name, covered_area = ComupteMetrics(
                                        output_dir=Path(args.output),
                                        image_name=image.name,
                                        projections=image_projections[image],
                                        scale_max_value=1,
                                        resolution=15,
                                        upsample_factor=10,
                                                       )
        results[image_name] = covered_area
    
    radiometric = []
    rgb = []
    for key, value in results.items():
        if "radiometric" in key:
            radiometric.append(value)
        else:
            rgb.append(value)
    print(f"Radiometric: {sum(radiometric)/len(radiometric)}")
    print(f"RGB: {sum(rgb)/len(rgb)}")



