"""
Raster Export & GIS Formatting Subsystem
Utility functions to manage GeoTIFF transformations and CRS alignments.
Ensures all outputs from ML and Simulator follow the same Coordinate Reference System.
"""

import rasterio
import numpy as np
import os

def export_georeferenced_raster(data, meta, filename, output_dir="outputs"):
    """
    Exports a 2D numpy array into a GeoTIFF with the specified metadata.
    Automatically handles data type casting and compression.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Update metadata to match the current data shape and type
    meta.update({
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "compress": "lzw"
    })
    
    output_path = os.path.join(output_dir, filename)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data, 1)
        
    print(f"Raster Exported: {output_path} | CRS: {meta.get('crs')} | Shape: {data.shape}")
    return output_path

if __name__ == "__main__":
    print("Raster export utility initialized.")
