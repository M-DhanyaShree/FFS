"""
Machine Learning Subsystem (AI Prediction)
Trains a RandomForestClassifier on pixel features to predict Fire Probability.
Handles training, evaluation, and outputting probability GeoTIFFs.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import rasterio
import os
import preprocessing

MODEL_FILE = "fire_model.joblib"
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

MODEL_FILE = "fire_model.joblib"
OUTPUT_DIR = "outputs"
DATA_FLAG = "data/ready.flag"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def train_and_export_prediction():
    """
    Optimized ML Pipeline for Large-Scale Data (5M+ samples)
    Includes: Sampling, Parallelism, Batch Prediction, and Timing.
    """
    start_total = time.time()
    
    # 1. Avoid recomputing data
    if not os.path.exists(DATA_FLAG):
        print("Initial data setup required...")
        import dataset_generator_pro
        dataset_generator_pro.generate_mock_real_data()
        with open(DATA_FLAG, 'w') as f: f.write('ready')

    print("--- OPTIMIZED ML PIPELINE START ---")
    
    # 2. Loading & Preprocessing
    t0 = time.time()
    X, y, scaler, meta, shape = preprocessing.preprocess_for_ml()
    print(f"Loaded {X.shape[0]} samples. Time: {time.time()-t0:.2f}s")

    # 3. Smart Sampling (Keep 20% for training)
    t0 = time.time()
    sample_size = int(0.2 * len(X))
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_train_sample = X[indices]
    y_train_sample = y[indices]
    print(f"Sampled {sample_size} training points. Time: {time.time()-t0:.2f}s")

    # 4. Fast Forest Training (Parallel + Fewer Estimators)
    t0 = time.time()
    # Use ExtraTreesClassifier for faster training if needed, or stick to RF with n_jobs=-1
    rf = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10, 
        n_jobs=-1, 
        random_state=42,
        class_weight='balanced' # Improved for rare fire events
    )
    rf.fit(X_train_sample, y_train_sample)
    print(f"Training Complete. Time: {time.time()-t0:.2f}s")

    # 5. Chunked Prediction (Memory-Safe)
    t0 = time.time()
    batch_size = 200000
    all_probs = []
    print(f"Predicting in {len(X)//batch_size + 1} batches...")
    for i in range(0, len(X), batch_size):
        batch_X = X[i : i + batch_size]
        all_probs.append(rf.predict_proba(batch_X)[:, 1])
    
    prob_map = np.concatenate(all_probs).reshape(shape)
    print(f"Prediction Complete. Time: {time.time()-t0:.2f}s")

    # 6. Export Results
    export_raster("fire_probability.tif", prob_map.astype(np.float32), meta)
    
    # Save Model
    joblib.dump((rf, scaler), MODEL_FILE)
    
    print(f"--- PIPELINE FINISHED: {time.time()-start_total:.2f}s ---")

def export_raster(filename, data, meta):
    """Utility to export GeoTIFF with correct metadata"""
    meta.update(dtype=data.dtype, count=1)
    with rasterio.open(os.path.join(OUTPUT_DIR, filename), 'w', **meta) as dst:
        dst.write(data, 1)
    print(f"Exported Raster: {filename}")

if __name__ == "__main__":
    import dataset_generator_pro
    dataset_generator_pro.generate_mock_real_data() # Ensure fresh data
    train_and_export_prediction()
