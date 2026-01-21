"""
Script to convert a TIFF file to a GeoTIFF using a transformation matrix from a text file.
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine


def read_transformation_matrix(matrix_file: Path) -> np.ndarray:
    """
    Read transformation matrix from a text file.
    
    Args:
        matrix_file: Path to the text file containing the transformation matrix
        
    Returns:
        4x4 transformation matrix as numpy array
    """
    matrix = np.loadtxt(matrix_file)
    
    # Ensure it's a 4x4 matrix
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {matrix.shape}")
    
    return matrix


def matrix_to_affine(matrix: np.ndarray) -> Affine:
    """
    Convert a 4x4 transformation matrix to rasterio Affine transform.
    
    The transformation matrix format is:
    [a  b  0  x_offset]
    [d  e  0  y_offset]
    [0  0  0  z_offset]
    [0  0  0  1       ]
    
    Rasterio Affine uses: (a, b, x_offset, d, e, y_offset)
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        Affine transformation for rasterio
    """
    # Extract the relevant components from the transformation matrix
    a = matrix[0, 0]  # pixel width
    b = matrix[0, 1]  # row rotation
    x_offset = matrix[0, 3]  # x coordinate of upper-left corner
    d = matrix[1, 0]  # column rotation
    e = matrix[1, 1]  # pixel height (negative value)
    y_offset = matrix[1, 3]  # y coordinate of upper-left corner
    print(a, b, x_offset, d, e, y_offset)
    #return Affine(a, b, x_offset, d, e, y_offset)
    return Affine(e, d, x_offset, b, a, y_offset)


def convert_to_geotiff(
    input_tiff: Path,
    output_geotiff: Path,
    transform_matrix: np.ndarray,
    crs: str
) -> None:
    """
    Convert a TIFF file to a GeoTIFF using the provided transformation matrix.
    
    Args:
        input_tiff: Path to the input TIFF file
        output_geotiff: Path to the output GeoTIFF file
        transform_matrix: 4x4 transformation matrix
        crs: Coordinate Reference System (e.g., 'EPSG:4326', 'EPSG:32632')
    """
    # Normalize CRS format - if just a number is provided, prepend "EPSG:"
    if crs.isdigit():
        crs = f"EPSG:{crs}"
    elif not crs.upper().startswith("EPSG:") and crs.isdigit():
        crs = f"EPSG:{crs}"
    
    # Convert matrix to Affine transform
    affine_transform = matrix_to_affine(transform_matrix)
    print(f"Affine Transform: {affine_transform}")
    
    # Read the input TIFF
    with rasterio.open(input_tiff) as src:
        # Read the image data
        data = src.read()
        
        # Get metadata
        metadata = src.meta.copy()
        
        # Update metadata with geospatial information
        metadata.update({
            'crs': crs,
            'transform': affine_transform
        })
        
        # Write the GeoTIFF
        with rasterio.open(output_geotiff, 'w', **metadata) as dst:
            dst.write(data)
    
    print(f"Successfully created GeoTIFF: {output_geotiff}")
    print(f"CRS: {crs}")
    print(f"Transform: {affine_transform}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TIFF file to a GeoTIFF using a transformation matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python write_tif_file.py input.tif output.tif transform.txt --crs EPSG:32632
    
Transformation matrix format (4x4):
    a   b   0   x_offset
    d   e   0   y_offset
    0   0   0   z_offset
    0   0   0   1
        """
    )
    
    parser.add_argument(
        'input_tiff',
        type=str,
        help='Path to the input TIFF file'
    )
    
    parser.add_argument(
        'output_geotiff',
        type=str,
        help='Path to the output GeoTIFF file'
    )
    
    parser.add_argument(
        'transform_matrix',
        type=str,
        help='Path to the text file containing the 4x4 transformation matrix'
    )
    
    parser.add_argument(
        '--crs',
        type=str,
        required=True,
        help='Coordinate Reference System (e.g., EPSG:4326, EPSG:32632, 6857, or just 3857)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_tiff = Path(args.input_tiff)
    output_geotiff = Path(args.output_geotiff)
    transform_file = Path(args.transform_matrix)
    
    # Validate input files exist
    if not input_tiff.exists():
        raise FileNotFoundError(f"Input TIFF file not found: {input_tiff}")
    
    if not transform_file.exists():
        raise FileNotFoundError(f"Transform matrix file not found: {transform_file}")
    
    # Read transformation matrix
    print(f"Reading transformation matrix from: {transform_file}")
    matrix = read_transformation_matrix(transform_file)
    print(f"Transformation matrix:\n{matrix}")
    
    # Convert to GeoTIFF
    convert_to_geotiff(input_tiff, output_geotiff, matrix, args.crs)


if __name__ == '__main__':
    main()
