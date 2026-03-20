"""
Upgraded Fire Simulator Module (Advanced Dijkstra-based Anisotropic Spread)
This version incorporates:
- Vector-based Wind Influence (directional weights).
- Vector-based Slope Influence (faster uphill, slower downhill).
- Geo-referenced outputs (Raster Snapshots).
"""

import numpy as np
import heapq
import rasterio
import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import matplotlib.animation as animation
import preprocessing

@dataclass(order=True)
class FireCell:
    priority: float
    position: Tuple[int, int] = field(compare=False)
    ignition_time: float = field(compare=False)

class UttarakhandFireSimulatorPro:
    def __init__(self, data_folder='data'):
        # 1. Load Preprocessed Data
        self.X, self.y, self.meta, self.shape = preprocessing.load_and_stack_features()
        self.rows, self.cols = self.shape
        
        # 2. Reshape features back to 2D for spatial indexing
        # Features are: [dem, veg, temp, hum, wind, slope, aspect]
        feats = self.X.reshape(self.rows, self.cols, -1)
        self.dem = feats[:, :, 0]
        self.veg = feats[:, :, 1]
        self.temp = feats[:, :, 2]
        self.hum = feats[:, :, 3]
        self.wind_speed = feats[:, :, 4]
        self.slope = feats[:, :, 5]
        self.aspect = feats[:, :, 6]
        
        # 3. Global parameters
        self.wind_direction = 45.0  # (Degrees from North, e.g., 45 = NE)
        self.burned_cells = {}
        
    def _calculate_directional_weight(self, r1, c1, r2, c2):
        """
        Calculates edge weight (spread time) between adjacent cells factoring vectors.
        Higher weight = Slower spread.
        """
        # Feature base Factors
        veg_factor = 1.0 / (max(0.1, self.veg[r2, c2] / 100.0))
        temp_factor = 1.0 / (max(0.1, (self.temp[r2, c2]-15) / 30.0))
        hum_factor = 1.0 + (self.hum[r2, c2] / 100.0)
        
        # 1. Slope Vector Factor: Faster uphill, slower downhill
        elev_diff = self.dem[r2, c2] - self.dem[r1, c1]
        # Slope factor (exponential increase uphill, decrease downhill)
        slope_vec_factor = np.exp(-0.02 * elev_diff) 
        
        # 2. Wind Vector Factor: Vector alignment between wind dir and spread dir
        # Calculate angle of spread from (r1, c1) to (r2, c2)
        dy, dx = -(r2 - r1), (c2 - c1) # Flip Y for image coords
        spread_angle = np.degrees(np.arctan2(dy, dx))
        spread_angle = (90 - spread_angle) % 360  # Align to North-based azimuth
        
        # Angle difference between wind and spread dir
        angle_diff = abs(spread_angle - self.wind_direction)
        if angle_diff > 180: angle_diff = 360 - angle_diff
        
        # Wind factor (faster if spread dir matches wind dir)
        wind_vec_factor = 1.0 / (1.0 + (self.wind_speed[r1, c1] / 10.0) * np.cos(np.radians(angle_diff)))
        
        # Combined Weight (Total Time to cross cell)
        time_to_cross = 1.0 * veg_factor * temp_factor * hum_factor * slope_vec_factor * wind_vec_factor
        return max(0.01, time_to_cross)

    def simulate_dijkstra(self, ignition_points: List[Tuple[int, int]], duration_hrs: int):
        """
        Standard Dijkstra with vector-weighted edges.
        """
        pq = []
        arrival_times = {}
        
        for p in ignition_points:
            heapq.heappush(pq, FireCell(0.0, p, 0.0))
            arrival_times[p] = 0.0
            
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        
        while pq:
            curr = heapq.heappop(pq)
            r, c = curr.position
            curr_time = curr.priority
            
            if curr_time > duration_hrs: break
            
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    weight = self._calculate_directional_weight(r, c, nr, nc)
                    arrival = curr_time + weight
                    
                    if (nr, nc) not in arrival_times or arrival < arrival_times[(nr, nc)]:
                        arrival_times[(nr, nc)] = arrival
                        heapq.heappush(pq, FireCell(arrival, (nr, nc), arrival))
                        
        self.burned_cells = arrival_times
        return arrival_times

    def export_snapshots(self, snapshots_hrs: List[int], output_dir='outputs'):
        """Generates GeoTIFFs for specific time intervals"""
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        for hr in snapshots_hrs:
            map_data = np.zeros(self.shape, dtype=np.uint8)
            for (r, c), arrival in self.burned_cells.items():
                if arrival <= hr:
                    map_data[r, c] = 1
            
            filename = f"fire_spread_{hr}hr.tif"
            self.meta.update(dtype=np.uint8, count=1)
            with rasterio.open(os.path.join(output_dir, filename), 'w', **self.meta) as dst:
                dst.write(map_data, 1)
            print(f"Exported Snapshot: {filename}")

if __name__ == "__main__":
    import train_model
    # 1. Ensure model is trained & data exists
    if not os.path.exists("outputs/fire_probability.tif"):
        train_model.train_and_export_prediction()

    # 2. Run simulation
    sim = UttarakhandFireSimulatorPro()
    # Starter points (central high risk areas)
    start_point = (sim.rows // 2, sim.cols // 2)
    print(f"Ignition at: {start_point}")
    
    sim.simulate_dijkstra([start_point], duration_hrs=12)
    sim.export_snapshots([1, 3, 6, 12])
    print("Simulation Complete!")
