"""
Streamlit Control Panel (Frontend Dashboard)
Enables interactive fire simulation and prediction for the Uttarakhand region.
Allows configuring wind speed, direction, and ignition points via UI.
Visualizes Fire Probability (ML) and Fire Spread (Simulation) in real-time.
"""

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
import os

# Imports from local modules
import train_model
import fire_simulator_pro
import dataset_generator_pro

st.set_page_config(page_title="Uttarakhand Forest Fire AI Predictor", layout="wide")

st.title("🔥 AI-Powered Forest Fire Prediction & Simulation (Uttarakhand)")
st.markdown("Developed for hacked-based real-world geospatial applications using Open-Source data.")

# Sidebar - Parameters
st.sidebar.header("Simulation Settings")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 25)
wind_dir = st.sidebar.slider("Wind Direction (° from N)", 0, 360, 45)
duration = st.sidebar.slider("Simulation Duration (Hrs)", 1, 48, 12)

st.sidebar.header("Data Controls")
if st.sidebar.button("Regenerate Mock Geospatial Data"):
    dataset_generator_pro.generate_mock_real_data()
    st.sidebar.success("New data generated!")

if st.sidebar.button("Train AI Prediction Model"):
    with st.spinner("Training RandomForestClassifier on pixel data..."):
        train_model.train_and_export_prediction()
        st.sidebar.success("Model trained & Probability Map generated!")

# Main Panel
col1, col2 = st.columns(2)

# Load existing outputs if available
prob_file = "outputs/fire_probability.tif"
if os.path.exists(prob_file):
    with col1:
        st.subheader("1. AI Fire Probability Prediction (ML)")
        with rasterio.open(prob_file) as src:
            prob_data = src.read(1)
            plt.figure(figsize=(10, 8))
            plt.imshow(prob_data, cmap='hot')
            plt.colorbar(label="Probability")
            plt.title("ML Forest Fire Risk Map")
            st.pyplot(plt)
            st.info("Map generated using pixel-wise features (elevation, slope, vegetation, weather).")

# Simulation Trigger
st.subheader("2. Run Interactive Fire Spread Simulation")
st.write("Click 'Start Simulation' to see how the fire spreads from a central ignition point based on current wind settings.")

if st.button("🚀 Start Fire Simulation"):
    with st.spinner("Running Dijkstra Simulation with Anisotropic Weights..."):
        sim = fire_simulator_pro.UttarakhandFireSimulatorPro()
        sim.wind_direction = wind_dir
        # Run simulation with 1 central point
        ignition = (sim.rows // 2, sim.cols // 2)
        sim.simulate_dijkstra([ignition], duration_hrs=duration)
        sim.export_snapshots([1, 6, 12, 24])
        
        # Display the result (e.g., 12hr map)
        snap_file = f"outputs/fire_spread_12hr.tif"
        if os.path.exists(snap_file):
            with col2:
                st.subheader(f"Fire Spread Snapshot ({duration} Hrs)")
                with rasterio.open(snap_file) as src:
                    spread_data = src.read(1)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(spread_data, cmap='Reds', alpha=0.8)
                    plt.title(f"Burned Area After {duration} Hours")
                    st.pyplot(plt)
                    st.success("Simulation Complete! Check 'outputs/' for GeoTIFF files.")

st.markdown("---")
st.subheader("📁 Output Files (Geo-referenced Raster)")
if os.path.exists("outputs"):
    files = os.listdir("outputs")
    st.write(f"Generated {len(files)} files in `outputs/` directory.")
    for f in files:
        if f.endswith(".tif"):
            st.text(f"✅ {f} (Ready for QGIS / ArcGIS)")
