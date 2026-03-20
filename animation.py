"""
Fire Spread Animation Utility
Generates an animated GIF of the fire spread simulation using snapshots.
Uses matplotlib.animation and ImageIO.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import rasterio

OUTPUT_DIR = "outputs"

def generate_fire_animation():
    """Reads all fire_spread_*.tif files and creates an animation"""
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("fire_spread_") and f.endswith(".tif")])
    if not files:
        print("No fire spread snapshots found in 'outputs/'. Run the simulator first.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ims = []
    
    for f in files:
        with rasterio.open(os.path.join(OUTPUT_DIR, f)) as src:
            data = src.read(1)
            im = ax.imshow(data, animated=True, cmap='Reds', alpha=0.8)
            txt = ax.text(0.5, 1.05, f"Simulation: {f.replace('fire_spread_', '').replace('.tif', '')}", 
                        ha="center", transform=ax.transAxes, fontsize=12)
            ims.append([im, txt])

    print(f"Combining {len(files)} snapshots into animation...")
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
    
    ani_path = os.path.join(OUTPUT_DIR, "fire_spread_animation.gif")
    ani.save(ani_path, writer='pillow')
    print(f"Animation saved to: {ani_path}")
    plt.close()

if __name__ == "__main__":
    import fire_simulator_pro
    # 1. Ensure snapshots exist
    sim = fire_simulator_pro.UttarakhandFireSimulatorPro()
    sim.simulate_dijkstra([(sim.rows//2, sim.cols//2)], duration_hrs=24)
    sim.export_snapshots([1, 2, 4, 6, 8, 12, 18, 24]) # More steps for smooth animation
    
    # 2. Animate
    generate_fire_animation()
