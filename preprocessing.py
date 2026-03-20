"""
Geospatial Data Preprocessing Module
Extracts features (SLOPE, ASPECT) from DEM and stacks all GIS layers into ML-ready inputs.
Uses rasterio for GeoTIFF handling and local NumPy-based raster math.
"""

import numpy as np
import rasterio
import os
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
FEATURE_FILES = [
    "dem.tif",
    "vegetation.tif",
    "temperature.tif",
    "humidity.tif",
    "wind_speed.tif"
]
LABEL_FILE = "fire_labels.tif"

def calculate_slope_aspect_numpy(dem, res):
    """
    Computes slope and aspect using central difference method on DEM.
    Slope in degrees (0-90), Aspect in degrees (0-360, relative to North).
    """
    dy, dx = np.gradient(dem, res)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180/np.pi)
    aspect = np.arctan2(-dy, dx) * (180/np.pi)
    aspect = (aspect + 360) % 360  # Normalize to 0-360
    return slope, aspect

def load_and_stack_features():
    """
    Loads all GeoTIFFs, scales features, and returns (X_stack, y_labels, metadata)
    """
    feature_data = []
    meta = None
    
    # Load primary features
    for file in FEATURE_FILES:
        path = os.path.join(DATA_DIR, file)
        with rasterio.open(path) as src:
            data = src.read(1)
            feature_data.append(data.flatten())
            if meta is None:
                meta = src.meta.copy()

    # Derived Features (Slope & Aspect)
    with rasterio.open(os.path.join(DATA_DIR, "dem.tif")) as src:
        dem = src.read(1)
        res_lat = abs(src.transform[4])
        res_lon = abs(src.transform[0])
        # Use simple mean for slope calculation or separate factors if needed
        slope, aspect = calculate_slope_aspect_numpy(dem, (res_lat + res_lon)/2)
        feature_data.append(slope.flatten())
        feature_data.append(aspect.flatten())

    # Stack Features
    X = np.stack(feature_data, axis=1) # (N_pixels, N_features)
    
    # Load Labels
    with rasterio.open(os.path.join(DATA_DIR, LABEL_FILE)) as src:
        y = src.read(1).flatten()
        
    return X, y, meta, (dem.shape[0], dem.shape[1])

def preprocess_for_ml():
    """Returns normalized X and y for Random Forest training"""
    X, y, meta, shape = load_and_stack_features()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, meta, shape

if __name__ == "__main__":
    X, y, scaler, meta, shape = preprocess_for_ml()
    print(f"Preprocessed Feature Matrix: {X.shape}")
    print(f"Fire Labels Found: {np.sum(y)}")
