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
from matplotlib.colors import ListedColormap
import os
import joblib
from scipy import ndimage
import io
import tempfile
from matplotlib import animation

# Imports from local modules
import train_model
import fire_simulator_pro
import dataset_generator_pro
import preprocessing

st.set_page_config(page_title="Forest Fire AI Dashboard", layout="wide", page_icon="🔥")

st.title("🔥 AI-Powered Forest Fire Decision Dashboard")
st.markdown("Interactive Real-Time Simulation and Hotspot Detection for Uttarakhand. Optimize resource deployment with instant AI predictions.")

# --- Cached Functions --- #
@st.cache_data
def load_base_probability_map():
    prob_file = "outputs/fire_probability.tif"
    if not os.path.exists(prob_file): return None, None, None
    with rasterio.open(prob_file) as src:
        prob_data = src.read(1)
        transform = src.transform
        shape = prob_data.shape
    return prob_data, transform, shape

@st.cache_data
def load_features():
    if not os.path.exists("data/dem.tif"): return None
    X, y, meta, shape = preprocessing.load_and_stack_features()
    feats = X.reshape(shape[0], shape[1], -1)
    feats_dict = {
        "dem": feats[:, :, 0], "veg": feats[:, :, 1], "temp": feats[:, :, 2],
        "hum": feats[:, :, 3], "wind": feats[:, :, 4], "slope": feats[:, :, 5]
    }
    return feats_dict

def calculate_adjusted_probability(base_prob, wind_speed, temp, humidity):
    """Real-time parameter sensitivity. Avoids retraining ML for what-if scenarios."""
    wind_factor = 1.0 + (wind_speed / 50.0) 
    humidity_factor = max(0.1, 1.0 - (humidity / 100.0))
    temp_factor = max(0.5, temp / 30.0)
    adj = base_prob * wind_factor * humidity_factor * temp_factor
    return np.clip(adj, 0.0, 1.0)

def detect_hotspots(adjusted_prob, transform):
    """Actionable Hotspot Detection: clusters areas with prob > 0.7"""
    high_risk = adjusted_prob > 0.7
    labeled, num_clusters = ndimage.label(high_risk)
    if num_clusters == 0: return []

    unique, counts = np.unique(labeled, return_counts=True)
    sizes = dict(zip(unique, counts))
    if 0 in sizes: del sizes[0] # remove background
    
    top_clusters = sorted(sizes.keys(), key=lambda k: sizes[k], reverse=True)[:5]
    hotspots = []
    
    for label in top_clusters:
        coords = np.argwhere(labeled == label)
        if len(coords) == 0: continue
        r_center, c_center = coords.mean(axis=0)
        max_prob = np.max(adjusted_prob[labeled == label])
        lon, lat = transform * (c_center, r_center) # using transform matrix
        
        risk = "Extreme" if max_prob >= 0.9 else "High"
        action = "Immediate containment & evacuation" if sizes[label] > 50 else "Deploy rapid response & fire lines"
        
        hotspots.append({"row": int(r_center), "col": int(c_center), "lat": round(lat, 4), "lon": round(lon, 4), "risk": risk, "action": action, "size": sizes[label]})
    return hotspots

def generate_animation_gif(arrival_times, shape, max_hrs):
    """Generates an animated GIF in memory representing time-based fire spread."""
    
    map_data = np.full(shape, np.nan)
    for (r, c), arrival in arrival_times.items():
        if arrival <= max_hrs:
            map_data[r, c] = arrival
            
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    ims = []
    step = max(1, max_hrs // 10)
    for hr in range(1, max_hrs + 1, step):
        frame = np.copy(map_data)
        frame[frame > hr] = np.nan
        im = ax.imshow(frame, animated=True, cmap='Reds', vmin=0, vmax=max_hrs)
        title = ax.text(0.5, 1.05, f"Spread at {hr} Hrs", ha="center", transform=ax.transAxes, fontsize=12)
        ims.append([im, title])
        
    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True)
    
    # Save animation to temporary file, then read into BytesIO
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        ani.save(tmp_path, writer='pillow')
        with open(tmp_path, 'rb') as f:
            gif_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    plt.close(fig)
    return gif_bytes

# --- App Layout --- #

st.sidebar.header("🎛️ Real-Time Parameters")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 25)
wind_dir = st.sidebar.selectbox("Wind Direction", [0, 45, 90, 135, 180, 225, 270, 315], index=1)
temp = st.sidebar.slider("Temperature (°C)", 10, 50, 30)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 40)
duration = st.sidebar.slider("Simulation Duration (Hrs)", 1, 48, 12)

st.sidebar.markdown("---")
st.sidebar.header("Data Controls")
if st.sidebar.button("Regenerate Mock Geospatial Data"):
    with st.spinner("Generating layers..."):
        dataset_generator_pro.generate_mock_real_data()
        st.cache_data.clear()
        st.sidebar.success("New data generated! Pls retrain AI model.")

if st.sidebar.button("Train AI Prediction Model"):
    with st.spinner("Training ML Model..."):
        train_model.train_and_export_prediction()
        st.cache_data.clear()
        st.sidebar.success("Model trained & Probability Map cached!")

prob_data, transform, shape = load_base_probability_map()
feats_dict = load_features()

# Auto-generate data and train model if missing
if prob_data is None or feats_dict is None:
    with st.spinner("🔧 Initializing system - generating data and training model..."):
        st.sidebar.info("Auto-generating baseline data...")
        dataset_generator_pro.generate_mock_real_data()
        
        st.sidebar.info("Auto-training ML model...")
        train_model.train_and_export_prediction()
        
        st.cache_data.clear()
        prob_data, transform, shape = load_base_probability_map()
        feats_dict = load_features()
        st.sidebar.success("✅ System initialized! Ready to go.")

# Real-Time Computation
adjusted_prob = calculate_adjusted_probability(prob_data, wind_speed, temp, humidity)
hotspots = detect_hotspots(adjusted_prob, transform)

# Main Application Tabs
main_tab1, main_tab2 = st.tabs(["� Risk Analysis & Hotspots", "🚀 Fire Simulation"])

# ============= TAB 1: Risk Analysis & Hotspots =============
with main_tab1:
    # SECTION 1: Real-Time Fire Risk Maps
    st.markdown("### 📊 Real-Time Fire Risk Assessment")
    st.divider()
    
    risk_sub_tabs = st.tabs(["🔥 Risk Classification", "📈 Probability Heatmap", "🌍 Geospatial Layers"])
    
    with risk_sub_tabs[0]:
        col_risk1, col_risk2 = st.columns([2, 1])
        with col_risk1:
            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = ListedColormap(['#a8e6cf', '#ffd3b6', '#ff8b94', '#ff2e63'])
            # Classify pixels
            risk_map = np.zeros_like(adjusted_prob)
            risk_map[adjusted_prob > 0.3] = 1
            risk_map[adjusted_prob > 0.7] = 2
            risk_map[adjusted_prob > 0.9] = 3
            
            im = ax.imshow(risk_map, cmap=cmap)
            ax.set_title("Risk Classification Map", fontsize=13, fontweight='bold')
            ax.axis("off")
            st.pyplot(fig, width='stretch')
        
        with col_risk2:
            st.markdown("**Risk Levels:**")
            st.write("🟢 **Low** (0-30%)")
            st.write("🟠 **Moderate** (30-70%)")
            st.write("🔴 **High** (70-90%)")
            st.write("🔴 **Extreme** (90-100%)")
            
            # Show statistics
            low_pct = np.sum(adjusted_prob <= 0.3) / adjusted_prob.size * 100
            mod_pct = np.sum((adjusted_prob > 0.3) & (adjusted_prob <= 0.7)) / adjusted_prob.size * 100
            high_pct = np.sum((adjusted_prob > 0.7) & (adjusted_prob <= 0.9)) / adjusted_prob.size * 100
            ext_pct = np.sum(adjusted_prob > 0.9) / adjusted_prob.size * 100
            
            st.metric("Low Risk %", f"{low_pct:.1f}%")
            st.metric("Moderate %", f"{mod_pct:.1f}%")
            st.metric("High %", f"{high_pct:.1f}%")
            st.metric("Extreme %", f"{ext_pct:.1f}%")
    
    with risk_sub_tabs[1]:
        col_heat1, col_heat2 = st.columns([2, 1])
        with col_heat1:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(adjusted_prob, cmap='hot', vmin=0, vmax=1)
            cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("Fire Probability", fontsize=10)
            ax.set_title("Fire Probability Heatmap (Weather-Adjusted)", fontsize=13, fontweight='bold')
            ax.axis("off")
            st.pyplot(fig, width='stretch')
        
        with col_heat2:
            avg_prob = np.mean(adjusted_prob)
            max_prob = np.max(adjusted_prob)
            min_prob = np.min(adjusted_prob)
            
            st.metric("Average Probability", f"{avg_prob:.3f}")
            st.metric("Maximum Probability", f"{max_prob:.3f}")
            st.metric("Minimum Probability", f"{min_prob:.3f}")
            
            st.markdown("**Parameters:**")
            st.write(f"🌬️ Wind: {wind_speed} km/h")
            st.write(f"🌡️ Temp: {temp}°C")
            st.write(f"💧 Humidity: {humidity}%")
    
    with risk_sub_tabs[2]:
        layer_select = st.selectbox("Select Geospatial Layer", ["Vegetation Index", "Terrain Slope", "Digital Elevation Model"], key="layer_select")
        col_geo1, col_geo2 = st.columns([2, 1])
        
        with col_geo1:
            fig, ax = plt.subplots(figsize=(8, 6))
            if layer_select == "Vegetation Index": 
                im = ax.imshow(feats_dict["veg"], cmap='YlGn')
                title = "Vegetation Density"
                unit = "Index"
            elif layer_select == "Terrain Slope": 
                im = ax.imshow(feats_dict["slope"], cmap='copper')
                title = "Terrain Slope"
                unit = "Degrees"
            else: 
                im = ax.imshow(feats_dict["dem"], cmap='terrain')
                title = "Digital Elevation Model"
                unit = "Meters"
            
            cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label(unit, fontsize=10)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis("off")
            st.pyplot(fig, width='stretch')
        
        with col_geo2:
            if layer_select == "Vegetation Index":
                st.write("🌳 **Vegetation Index**")
                st.write("High vegetation density = fuel availability")
            elif layer_select == "Terrain Slope":
                st.write("⛰️ **Terrain Slope**")
                st.write("Steeper slopes = faster fire spread")
            else:
                st.write("🏔️ **Elevation**")
                st.write("Affects climate & vegetation type")

# ============= TAB 2: Fire Simulation & Animation =============
with main_tab2:
    st.markdown("### 🎯 Actionable Hotspots for Quick Simulation")
    st.divider()
    
    # HOTSPOTS SECTION
    if hotspots:
        st.success(f"✅ **{len(hotspots)} Critical Hotspots Ready**")
        st.write("*Click any hotspot below to select it as ignition point, then run simulation*")
        st.markdown("")
        
        # Display hotspots in a clean grid
        for idx in range(0, len(hotspots), 2):
            col_hs1, col_hs2 = st.columns(2)
            
            for col_idx, col in enumerate([col_hs1, col_hs2]):
                if idx + col_idx < len(hotspots):
                    hs = hotspots[idx + col_idx]
                    with col:
                        with st.container(border=True):
                            st.markdown(f"### 🔥 Hotspot {idx + col_idx + 1}")
                            
                            # Risk level badge
                            risk_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡"}
                            st.markdown(f"**Risk Level:** {risk_colors.get(hs['risk'], '⭕')} {hs['risk']}")
                            
                            col_metric1, col_metric2 = st.columns(2)
                            with col_metric1:
                                st.metric("Latitude", f"{hs['lat']:.4f}")
                                st.metric("Size", f"{hs['size']} px")
                            with col_metric2:
                                st.metric("Longitude", f"{hs['lon']:.4f}")
                            
                            st.markdown(f"**📍 Action:** {hs['action']}")
                            
                            if st.button(f"Select Hotspot {idx + col_idx + 1}", key=f"sim_btn_{idx + col_idx}", width='stretch', type="secondary"):
                                st.session_state.ignite_point = (hs["row"], hs["col"])
                                st.rerun()
    else:
        st.info("✨ No extreme hotspots detected. Use manual coordinates below.")
    
    st.markdown("---")
    st.markdown("### � Interactive Fire Spread Simulation")
    st.divider()
    
    col_controls, col_results = st.columns([1, 1.5], gap="large")
    
    with col_controls:
        st.markdown("#### 🎯 Simulation Setup")
        
        with st.container(border=True):
            st.markdown("**Select Ignition Point**")
            
            default_r = shape[0] // 2
            default_c = shape[1] // 2
            if 'ignite_point' in st.session_state:
                default_r, default_c = st.session_state.ignite_point
            
            # Option 1: Enter coordinates
            input_latlon = st.text_input("Lat, Lon Coordinates", "0.00, 0.00", help="Format: latitude, longitude")
            if input_latlon != "0.00, 0.00":
                try:
                    parts = input_latlon.split(",")
                    lat, lon = float(parts[0].strip()), float(parts[1].strip())
                    inv_transform = ~transform
                    c, r = inv_transform * (lon, lat)
                    default_r, default_c = int(r), int(c)
                except:
                    st.error("Invalid format. Use: lat, lon")
            
            st.write(f"**Pixel Location:** Row {default_r}, Col {default_c}")
            

            if 'ignite_point' in st.session_state:
                st.success("✓ Hotspot selected from Risk Analysis tab")
        
        with st.container(border=True):
            st.markdown("**Environmental Conditions** (from sidebar)")
            st.write(f"🌬️ Wind Speed: **{wind_speed} km/h**")
            st.write(f"🧭 Wind Direction: **{wind_dir}°**")
            st.write(f"🌡️ Temperature: **{temp}°C**")
            st.write(f"💧 Humidity: **{humidity}%**")
            st.write(f"⏱️ Duration: **{duration} hours**")
        
        st.markdown("---")
        if st.button("▶️ Run Simulation", type="primary", width='stretch', key="run_sim"):
            with st.spinner("🔥 Running Dijkstra fire spread algorithm..."):
                sim = fire_simulator_pro.UttarakhandFireSimulatorPro()
                sim.wind_direction = wind_dir
                sim.wind_speed[:] = wind_speed
                sim.temp[:] = temp
                sim.hum[:] = humidity
                
                sim.simulate_dijkstra([(default_r, default_c)], duration_hrs=duration)
                st.session_state.sim_result = sim.burned_cells
                st.success("✅ Simulation complete!")
    
    with col_results:
        st.markdown("#### 📊 Simulation Results")
        
        if 'sim_result' in st.session_state:
            # Animation section
            with st.container(border=True):
                st.markdown("**🔥 Fire Spread Animation**")
                gif_bytes = generate_animation_gif(st.session_state.sim_result, shape, duration)
                st.image(gif_bytes, width='stretch')
            
            st.markdown("")
            
            # Time-to-reach map section
            with st.container(border=True):
                st.markdown("**⏱️ Evacuation Time Map**")
                st.write("Shows how many hours it takes fire to reach each location")
                
                final_map = np.full(shape, np.nan)
                for (r, c), t in st.session_state.sim_result.items():
                    if t <= duration: 
                        final_map[r, c] = t
                
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(final_map, cmap='magma_r', interpolation='bilinear')
                cbar = fig.colorbar(im, ax=ax, label="Hours to Reach Fire")
                ax.set_title("Evacuation / Response Time (Hours)", fontsize=12, fontweight='bold')
                ax.axis("off")
                st.pyplot(fig, width='stretch')
        else:
            with st.container(border=True):
                st.markdown("**⏳ Awaiting Simulation**")
                st.info("� Set ignition point using coordinates or select from Hotspots, then click 'Run Simulation' to view results")

