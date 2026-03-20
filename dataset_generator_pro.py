"""
Generator for realistic Geospatial Data (Mock Real Data)
Produces GeoTIFF files that would normally be downloaded from SRTM, ERA5, and NASA FIRMS.
This ensures the rest of the pipeline (ML + Simulation) works with Geo-referenced data.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

# Create data directory
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Constants for Uttarakhand Subset (Pauri/Nainital region)
LAT_MAX, LAT_MIN = 30.15, 29.75
LON_MIN, LON_MAX = 78.50, 79.50
# Flexible Grid Resolution (Context: Large Scale = 1481 x 3703)
# For demo, keeping it manageable, but logic scales.
ROWS, COLS = 120, 120 
# ROWS, COLS = 1481, 3703 # Uncomment for Stress Test

def create_geotiff(filename, data, transform, crs="EPSG:4326"):
    """Saves a numpy array as a georeferenced GeoTIFF"""
    with rasterio.open(
        os.path.join(DATA_DIR, filename),
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    print(f"Created: {filename} ({data.shape[0]}x{data.shape[1]})")

def generate_mock_real_data():
    res_lat = (LAT_MAX - LAT_MIN) / ROWS
    res_lon = (LON_MAX - LON_MIN) / COLS
    transform = from_origin(LON_MIN, LAT_MAX, res_lon, res_lat)
    
    # 1. Elevation (DEM) - Fractal terrain
    x = np.linspace(0, 5, COLS)
    y = np.linspace(0, 5, ROWS)
    X, Y = np.meshgrid(x, y)
    dem = (np.sin(X) * np.cos(Y) * 500 + 2000 + np.random.normal(0, 20, (ROWS, COLS))).astype(np.float32)
    create_geotiff("dem.tif", dem, transform)
    
    # 2. Vegetation Density (Simulated Sentinel-2 NDVI)
    veg = (np.random.beta(5, 2, (ROWS, COLS)) * 100).astype(np.float32)
    create_geotiff("vegetation.tif", veg, transform)
    
    # 3. Weather - Temperature (Simulated ERA5)
    temp = (25 + 5 * np.sin(X/2) + np.random.normal(0, 1, (ROWS, COLS))).astype(np.float32)
    create_geotiff("temperature.tif", temp, transform)
    
    # 4. Weather - Humidity
    humidity = (30 + 10 * np.cos(Y/3) + np.random.normal(0, 2, (ROWS, COLS))).astype(np.float32)
    create_geotiff("humidity.tif", humidity, transform)
    
    # 5. Weather - Wind Speed
    wind = (np.random.gamma(2, 2, (ROWS, COLS)) * 5).astype(np.float32)
    create_geotiff("wind_speed.tif", wind, transform)

    # 6. Fire Labels (Mock NASA FIRMS VIIRS)
    # Put some fires in high probability zones (high temp, high veg, low humidity)
    fire_prob = (temp/40.0 * veg/100.0 * (1 - humidity/100.0))
    fire_labels = (fire_prob > np.percentile(fire_prob, 95)).astype(np.uint8)
    create_geotiff("fire_labels.tif", fire_labels, transform)

if __name__ == "__main__":
    generate_mock_real_data()
