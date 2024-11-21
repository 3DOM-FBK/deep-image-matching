import sqlite3
import argparse
import pycolmap
import random
import numpy as np
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def ExportMatches(
    database_path: Path, 
    min_num_matches: int, 
    output_path: Path,
    width: int,
    height: int,
) -> None:
    
    if not output_path.exists():
        print("Output folder does not exist")
        quit()

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    images = {}
    pairs = []

    cursor.execute("SELECT image_id, name FROM images")
    for row in cursor:
        image_id = row[0]
        name = row[1]
        images[image_id] = name

    cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
    for row in cursor:
        pair_id = row[0]
        n_matches = row[1]
        id_img1, id_img2 = pair_id_to_image_ids(pair_id)
        id_img1, id_img2 = int(id_img1), int(id_img2)
        img1 = images[id_img1]
        img2 = images[id_img2]

        if n_matches >= min_num_matches:
            pairs.append((img1, img2))
    
    with open(output_path / "pairs.txt", "w") as f:
        n_pairs = len(pairs)
        n_brute = 0
        N = len(images.keys())
        for x in range(1, N):
            n_brute += N-x
        print(f"n_pairs/n_brute: {n_pairs} / {n_brute}")
        
        for pair in pairs:
            f.write(f"{pair[0]} {pair[1]}\n")

    connection.close()

def EvaluateKeypointDensity(
        database_path: Path,
        min_num_matches: Path,
        output_path: Path,
        model_path: Path,
    ) -> None:

    reconstruction = pycolmap.Reconstruction(model_path)

    point3Ds = []
    projections_3d_tiepoints = {}
    for point3D_id in reconstruction.points3D:
        point3Ds.append(point3D_id)
    
    #random_point3D_ids = random.sample(point3Ds, 200)

    #for point3D_id in random_point3D_ids:
    for point3D_id in point3Ds:
        point3D = reconstruction.points3D[point3D_id]
        #print(f"3D Point {point3D_id} coordinates: {point3D.xyz}")

        projections = []
        for image in reconstruction.images.values():
            # For each image, check if it observes the 3D point
            for feature_idx, point2D in enumerate(image.points2D):
                if point2D.has_point3D and point2D.point3D_id == point3D_id:
                    # Add the projection information: image ID and 2D point coordinates
                    projections.append((image.image_id, point2D.xy))
                    #print(f"Image {image.image_id}: Projected 2D point (x, y) = {point2D.xy}")

        projections_3d_tiepoints[point3D_id] = {
            '3D_tiepoint': point3D.xyz,
            'projections': projections,
        }

    return projections_3d_tiepoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export pairs in a txt file from a COLMAP database."
    )
    parser.add_argument("-d", "--database", type=Path, required=True, help="Path to COLMAP database.")
    parser.add_argument("-m", "--min_n_matches", type=int, required=True, help="Min number of matches that a pair should have after geometric verification.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to output folder.")
    parser.add_argument("-l", "--model", type=Path, required=True, help="Path to model folder.")
    args = parser.parse_args()

    #ExportMatches(
    #    database_path=args.database,
    #    min_num_matches=args.min_n_matches,
    #    output_path=args.output,
    #    width=args.width,
    #    height=args.height,
    #)

    projections_3d_tiepoints = EvaluateKeypointDensity(
                                    database_path=args.database,
                                    min_num_matches=args.min_n_matches,
                                    output_path=args.output,
                                    model_path=args.model,
                                )
    print(projections_3d_tiepoints)

    
    x_values = np.array([])
    y_values = np.array([])
    for point in projections_3d_tiepoints:
        projections = projections_3d_tiepoints[point]['projections']
        x_values = np.concatenate((x_values, np.array([prj[1][0] for prj in projections])))
        y_values = np.concatenate((y_values, np.array([prj[1][1] for prj in projections])))

    
    xmin, ymin, xmax, ymax = 0, 0, 1067, 800  # Rectangle bounds (local system)
    resolution = 30  # Grid cell size in the same units as your coordinates

    # Generate example points
    x_points = np.random.uniform(xmin, xmax, 1000)  # Example x-coordinates
    y_points = np.random.uniform(ymin, ymax, 1000)  # Example y-coordinates
    x_points = x_values
    y_points = y_values

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
    density_grid_normalized = density_grid / density_grid.max()  # Scale to [0, 1]
    density_grid_normalized = (density_grid_normalized * 255).astype(np.uint8)  # Scale to [0, 255]

    # Define raster transformation for local reference
    transform = from_origin(xmin, ymax, resolution, resolution)

    # Save the density map as a JPEG
    with rasterio.open(
        '/home/threedom/Desktop/buttare/density.jpg',
        "w",
        driver="JPEG",  # Use JPEG driver
        height=density_grid_normalized.shape[0],
        width=density_grid_normalized.shape[1],
        count=1,
        dtype=density_grid_normalized.dtype,
        transform=transform
    ) as dst:
        dst.write(density_grid_normalized, 1)




