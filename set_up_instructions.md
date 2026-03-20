# 🔥 Uttarakhand Forest Fire AI Predictor Setup Instructions

This guide will walk you through setting up and running the **Real-Data-Driven Forest Fire Simulation and Prediction System** for the Uttarakhand region.

## 🛠️ Prerequisites
- Python 3.8 to 3.12 (standard installation)
- `pip` (the Python package manager)

## 📥 Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/M-DhanyaShree/FFS.git
cd FFS
```

## 📦 Step 2: Install Libraries
Install all required GIS and AI dependencies from the optimized requirements file:
```bash
pip install -r requirements.txt
```
*Note: If `rasterio` fails to install on Windows, try `pip install pipwin && pipwin install rasterio`.*

## ⚙️ Step 3: Run the Full Data Pipeline
Run the following sequence to generate mock geospatial data, train the AI model, and execute the simulation:
```bash
python dataset_generator_pro.py && python train_model.py && python fire_simulator_pro.py && python animation.py
```
*Wait for the progress logs; you'll see timing information for the 20% optimized training cycle.*

## 🖥️ Step 4: Launch the Dashboard
Run the interactive Streamlit UI to visualize predictions and adjust wind parameters:
```bash
python -m streamlit run app.py
```

## 📂 Project Structure
- `dataset_generator_pro.py`: Generates the GeoTIFF data required for the GIS layers.
- `preprocessing.py`: Handles slope, aspect, and feature alignment.
- `train_model.py`: Optimized Random Forest trainer for fire probability mapping.
- `fire_simulator_pro.py`: Dijkstra-based fire spread simulator (anisotropic).
- `app.py`: Streamlit frontend dashboard.
- `outputs/`: Location for generated TIF maps and the firing spread animation GIF.

---
**Disclaimer**: This project uses simulated GeoTIFFs formatted as SRTM/Sentinel data for demo purposes. In a professional setting, we use the `preprocessing.py` logic to ingest direct satellite imagery.
