import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import deque
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict
import json

@dataclass(order=True)
class FireCell:
    """Priority queue item for fire spread simulation"""
    priority: float
    position: Tuple[int, int] = field(compare=False)
    ignition_time: float = field(compare=False)

class UttarakhandForestFireSimulator:
    """Realistic forest fire simulator for Uttarakhand region"""
    
    LAT_MIN, LAT_MAX = 29.75, 30.15  
    LON_MIN, LON_MAX = 78.50, 79.50  
    
    FIRE_LOCATIONS_2021 = {
        'Pauri_Srinagar': (29.916, 78.777),
        'Nainital_Town': (29.380, 79.450),
        'Almora_District': (29.597, 79.659),
        'Ramnagar': (29.392, 79.127),
        'Haldwani': (29.217, 79.511),
        'Kotdwar': (29.746, 78.524),
        'Lansdowne': (29.837, 78.680),
        'Ranikhet': (29.642, 79.432)
    }
    
    LANDMARKS = {
        'Dehradun': (30.316, 78.032),
        'Pauri': (30.155, 78.780),
        'Nainital': (29.380, 79.450),
        'Almora': (29.597, 79.659),
        'Rudraprayag': (30.284, 78.980),
        'Tehri': (30.391, 78.478)
    }
    
    def __init__(self, grid_size: Tuple[int, int] = (120, 120)):
        self.rows, self.cols = grid_size
        
        self.lat_per_cell = (self.LAT_MAX - self.LAT_MIN) / self.rows
        self.lon_per_cell = (self.LON_MAX - self.LON_MIN) / self.cols
        
        self.terrain_graph = {}
        self.fire_probability_map = np.zeros((self.rows, self.cols))
        self.vegetation_map = np.zeros((self.rows, self.cols))
        self.elevation_map = np.zeros((self.rows, self.cols))
        self.wind_speed_map = np.zeros((self.rows, self.cols))
        self.temperature_map = np.zeros((self.rows, self.cols))
        self.humidity_map = np.zeros((self.rows, self.cols))
        
        # Location mapping
        self.location_grid_map = {}
        self.fire_spread_history = []
        self.burned_cells = set()
        
    def latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert latitude/longitude to grid coordinates"""
        row = int((self.LAT_MAX - lat) / self.lat_per_cell)
        col = int((lon - self.LON_MIN) / self.lon_per_cell)
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))
        return (row, col)
    
    def grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to latitude/longitude"""
        lat = self.LAT_MAX - (row * self.lat_per_cell)
        lon = self.LON_MIN + (col * self.lon_per_cell)
        return (lat, lon)
    
    def load_data_from_file(self, filename: str):
        """Load real-world forest data from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.vegetation_map = np.array(data['vegetation_density'])
        self.elevation_map = np.array(data['elevation'])
        self.wind_speed_map = np.array(data['wind_speed'])
        self.temperature_map = np.array(data['temperature'])
        self.humidity_map = np.array(data['humidity'])
        
        self.rows, self.cols = self.vegetation_map.shape
        
        # Map real locations to grid
        for loc_name, (lat, lon) in self.FIRE_LOCATIONS_2021.items():
            grid_pos = self.latlon_to_grid(lat, lon)
            self.location_grid_map[loc_name] = grid_pos
        
        for loc_name, (lat, lon) in self.LANDMARKS.items():
            grid_pos = self.latlon_to_grid(lat, lon)
            self.location_grid_map[loc_name] = grid_pos
        
        self._build_terrain_graph()
        self._calculate_fire_probability()
        
    def _build_terrain_graph(self):
        """Build adjacency list for terrain (Graph DS)"""
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(self.rows):
            for j in range(self.cols):
                neighbors = []
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        weight = self._calculate_spread_factor(i, j, ni, nj)
                        neighbors.append(((ni, nj), weight))
                self.terrain_graph[(i, j)] = neighbors
    
    def _calculate_spread_factor(self, i1: int, j1: int, i2: int, j2: int) -> float:
        """Calculate fire spread rate between two cells"""
        veg_factor = (self.vegetation_map[i2, j2] / 100.0) * 1.5
        elevation_diff = self.elevation_map[i2, j2] - self.elevation_map[i1, j1]
        slope_factor = 1.0 + (elevation_diff / 100.0) * 0.3
        wind_factor = 1.0 + (self.wind_speed_map[i1, j1] / 50.0) * 0.5
        humidity_factor = 1.0 - (self.humidity_map[i1, j1] / 100.0) * 0.4
        temp_factor = 1.0 + ((self.temperature_map[i1, j1] - 20) / 50.0) * 0.3
        
        spread_rate = veg_factor * slope_factor * wind_factor * humidity_factor * temp_factor
        return max(0.1, spread_rate)
    
    def _calculate_fire_probability(self):
        """Calculate fire ignition probability for each cell"""
        veg_norm = self.vegetation_map / 100.0
        temp_norm = np.clip((self.temperature_map - 20) / 30.0, 0, 1)
        humidity_norm = 1.0 - (self.humidity_map / 100.0)
        wind_norm = self.wind_speed_map / 50.0
        
        self.fire_probability_map = (
            0.35 * veg_norm +
            0.25 * temp_norm +
            0.25 * humidity_norm +
            0.15 * wind_norm
        ) * 100.0
    
    def get_real_ignition_points(self) -> Dict[str, Tuple[int, int]]:
        """Get real fire ignition points from 2021 incidents"""
        ignition_points = {}
        for loc_name in ['Pauri_Srinagar', 'Nainital_Town', 'Almora_District']:
            if loc_name in self.location_grid_map:
                ignition_points[loc_name] = self.location_grid_map[loc_name]
        return ignition_points
    
    def simulate_fire_spread_dijkstra(self, ignition_points: Dict[str, Tuple[int, int]], 
                                      duration_hours: int) -> List[np.ndarray]:
        """Simulate fire spread using Dijkstra's algorithm with Priority Queue"""
        snapshots = []
        current_fire = np.zeros((self.rows, self.cols))
        
        pq = []
        arrival_time = {}
        
        # Initialize with ignition points
        for loc_name, point in ignition_points.items():
            heapq.heappush(pq, FireCell(0.0, point, 0.0))
            arrival_time[point] = 0.0
            current_fire[point] = 1
            print(f"  Ignition at {loc_name}: Grid({point[0]}, {point[1]})")
        
        snapshots.append(current_fire.copy())
        
        # Process fire spread
        while pq:
            cell = heapq.heappop(pq)
            i, j = cell.position
            current_time = cell.priority
            
            if current_time > duration_hours:
                break
            
            for (ni, nj), weight in self.terrain_graph[(i, j)]:
                spread_time = current_time + (1.0 / weight)
                
                if (ni, nj) not in arrival_time or spread_time < arrival_time[(ni, nj)]:
                    arrival_time[(ni, nj)] = spread_time
                    heapq.heappush(pq, FireCell(spread_time, (ni, nj), spread_time))
        
        # Create hourly snapshots
        for hour in range(1, duration_hours + 1):
            hour_fire = np.zeros((self.rows, self.cols))
            for (i, j), time in arrival_time.items():
                if time <= hour:
                    hour_fire[i, j] = 1
            snapshots.append(hour_fire)
        
        self.burned_cells = set(arrival_time.keys())
        return snapshots
    
    def plot_probability_map(self, save_path: str = 'fire_probability_map.png'):
        """Plot realistic fire probability map with actual locations"""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Color scheme
        colors = ['#00ff00', '#90ee90', '#ffff00', '#ffa500', '#ff4500', '#8b0000']
        cmap = ListedColormap(colors)
        
        im = ax.imshow(self.fire_probability_map, cmap=cmap, vmin=0, vmax=100,
                      extent=[self.LON_MIN, self.LON_MAX, self.LAT_MIN, self.LAT_MAX],
                      aspect='auto', origin='upper')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[10, 30, 50, 65, 75, 90], 
                           fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extreme'])
        cbar.set_label('Fire Risk Level', rotation=270, labelpad=25, fontsize=11, fontweight='bold')
        
        # Plot landmarks and fire locations
        for loc_name, (lat, lon) in self.LANDMARKS.items():
            if self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX:
                ax.plot(lon, lat, 'k^', markersize=8, markeredgewidth=1.5, 
                       markerfacecolor='white', markeredgecolor='black')
                ax.text(lon, lat + 0.02, loc_name, fontsize=9, fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor='black', alpha=0.8))
        
        # Mark 2021 fire locations
        for loc_name, (lat, lon) in self.FIRE_LOCATIONS_2021.items():
            if self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX:
                ax.plot(lon, lat, 'r*', markersize=15, markeredgewidth=1.5,
                       markeredgecolor='darkred')
        
        # Add legend for symbols
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='white',
                  markeredgecolor='black', markersize=8, label='Major Towns'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                  markeredgecolor='darkred', markersize=12, label='2021 Fire Locations')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                 framealpha=0.9, edgecolor='black')
        
        # Grid and labels
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
        ax.set_title('Uttarakhand Forest Fire Probability Map\n' +
                    'Pauri Garhwal - Nainital - Almora Region (April 2021 Conditions)',
                    fontsize=14, fontweight='bold', pad=15)
        
        # North arrow
        ax.annotate('N', xy=(0.97, 0.95), xycoords='axes fraction',
                   fontsize=18, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
        ax.annotate('↑', xy=(0.97, 0.92), xycoords='axes fraction',
                   fontsize=22, ha='center')
        
        # Scale bar (approximate)
        scale_km = 10  # 10 km
        lon_per_km = (self.LON_MAX - self.LON_MIN) / 100  # rough estimate
        scale_lon = scale_km * lon_per_km
        scale_x = self.LON_MIN + 0.05
        scale_y = self.LAT_MIN + 0.02
        ax.plot([scale_x, scale_x + scale_lon], [scale_y, scale_y], 'k-', linewidth=3)
        ax.text(scale_x + scale_lon/2, scale_y + 0.01, f'{scale_km} km',
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Probability map saved to {save_path}")
        plt.close()
    
    def plot_spread_simulation(self, snapshots: List[np.ndarray],
                               ignition_points: Dict[str, Tuple[int, int]],
                               save_path: str = 'fire_spread_simulation.png'):
        """Plot fire spread over time with real locations"""
        n_snapshots = min(6, len(snapshots) - 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Base colors
        base_colors = ['#00ff00', '#90ee90', '#ffff00']
        base_cmap = ListedColormap(base_colors)
        
        for idx in range(n_snapshots):
            ax = axes[idx]
            snapshot = snapshots[idx + 1]
            
            # Base probability map
            prob_normalized = self.fire_probability_map / 100.0
            ax.imshow(prob_normalized, cmap=base_cmap, alpha=0.6, vmin=0, vmax=1,
                     extent=[self.LON_MIN, self.LON_MAX, self.LAT_MIN, self.LAT_MAX],
                     aspect='auto', origin='upper')
            
            # Fire overlay
            fire_overlay = np.ma.masked_where(snapshot == 0, snapshot)
            ax.imshow(fire_overlay, cmap='Reds', alpha=0.8, vmin=0, vmax=1,
                     extent=[self.LON_MIN, self.LON_MAX, self.LAT_MIN, self.LAT_MAX],
                     aspect='auto', origin='upper')
            
            # Mark ignition points
            for loc_name, (row, col) in ignition_points.items():
                lat, lon = self.grid_to_latlon(row, col)
                circle = plt.Circle((lon, lat), 0.015, color='black',
                                  fill=False, linewidth=2.5)
                ax.add_patch(circle)
                # Label ignition point
                ax.text(lon, lat - 0.03, loc_name.replace('_', ' '),
                       fontsize=7, ha='center', va='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                               alpha=0.7, edgecolor='black'))
            
            # Add some landmarks
            for loc_name in ['Pauri', 'Nainital', 'Almora']:
                if loc_name in self.location_grid_map:
                    row, col = self.location_grid_map[loc_name]
                    lat, lon = self.grid_to_latlon(row, col)
                    if self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX:
                        ax.plot(lon, lat, 'k^', markersize=5, markerfacecolor='white',
                               markeredgecolor='black', markeredgewidth=1)
            
            ax.set_title(f'Hour-{idx+1:02d}', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                                edgecolor='black', linewidth=2))
            ax.set_xlabel('Longitude (°E)', fontsize=9)
            ax.set_ylabel('Latitude (°N)', fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.tick_params(labelsize=8)
        
        plt.suptitle('Forest Fire Spread Simulation - Uttarakhand Region\n' +
                    'Real-time propagation from 2021 fire incident locations (April 19, 2021)',
                    fontsize=15, fontweight='bold', y=0.995)
        
        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Active Fire/Burned Area'),
            Patch(facecolor='yellow', alpha=0.6, label='High Fire Probability'),
            Patch(facecolor='green', alpha=0.6, label='Low Fire Probability'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                  markeredgecolor='black', markersize=8, markeredgewidth=2,
                  label='Fire Ignition Point'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='white',
                  markeredgecolor='black', markersize=6, label='Major Town')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                  fontsize=9, frameon=True, fancybox=True, shadow=True,
                  bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.99])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Spread simulation saved to {save_path}")
        plt.close()
    
    def generate_report(self, ignition_points: Dict[str, Tuple[int, int]]):
        """Generate detailed simulation report"""
        print("\n" + "="*70)
        print("UTTARAKHAND FOREST FIRE SIMULATION REPORT")
        print("="*70)
        print(f"Study Area: Pauri Garhwal - Nainital - Almora Districts")
        print(f"Date Reference: April 19, 2021 (Peak Fire Season)")
        print(f"Geographical Bounds:")
        print(f"  Latitude: {self.LAT_MIN}°N to {self.LAT_MAX}°N")
        print(f"  Longitude: {self.LON_MIN}°E to {self.LON_MAX}°E")
        print(f"\nGrid Resolution: {self.rows} x {self.cols}")
        print(f"Total Cells: {self.rows * self.cols}")
        print(f"Cells Burned: {len(self.burned_cells)}")
        print(f"Burn Percentage: {len(self.burned_cells)/(self.rows*self.cols)*100:.2f}%")
        
        print(f"\nFire Ignition Points (2021 Actual Locations):")
        for loc_name, (row, col) in ignition_points.items():
            lat, lon = self.grid_to_latlon(row, col)
            print(f"  • {loc_name.replace('_', ' ')}: {lat:.3f}°N, {lon:.3f}°E")
        
        print(f"\nHigh Risk Analysis:")
        high_risk_count = np.sum(self.fire_probability_map > 70)
        print(f"  Cells with >70% risk: {high_risk_count}")
        print(f"  High Risk Percentage: {high_risk_count/(self.rows*self.cols)*100:.2f}%")
        
        print(f"\nData Structures & Algorithms Used:")
        print("  ✓ Graph (Adjacency List) - Terrain connectivity")
        print("  ✓ Priority Queue (Min-Heap) - Dijkstra's algorithm")
        print("  ✓ Hash Set - O(1) burned cell tracking")
        print("  ✓ Dictionary - Location-to-grid mapping")
        print(f"  ✓ Time Complexity: O((V + E) log V)")
        print(f"  ✓ Space Complexity: O(V)")
        print("="*70 + "\n")


def main():
    print("="*70)
    print("UTTARAKHAND FOREST FIRE PREDICTION & SPREAD SIMULATION")
    print("Based on Real 2021 Forest Fire Incidents")
    print("="*70 + "\n")
    
    # Initialize simulator
    print("Initializing Uttarakhand forest fire simulator...")
    simulator = UttarakhandForestFireSimulator(grid_size=(120, 120))
    print("✓ Simulator initialized\n")
    
    # Load data
    print("Loading real-world forest data...")
    simulator.load_data_from_file('dataset.txt')
    print("✓ Data loaded successfully")
    print(f"  Mapped {len(simulator.location_grid_map)} real locations to grid\n")
    
    # Generate probability map
    print("Generating fire probability map...")
    simulator.plot_probability_map()
    
    # Get real ignition points from 2021 fires
    print("\nIdentifying fire ignition points from 2021 incidents...")
    ignition_points = simulator.get_real_ignition_points()
    print(f"✓ Using {len(ignition_points)} real fire locations:")
    
    # Simulate fire spread
    print("\nSimulating fire spread using Dijkstra's algorithm...")
    duration_hours = 6
    snapshots = simulator.simulate_fire_spread_dijkstra(ignition_points, duration_hours)
    print(f"✓ Simulation complete for {duration_hours} hours\n")
    
    # Plot spread
    print("Generating spread visualization...")
    simulator.plot_spread_simulation(snapshots, ignition_points)
    
    # Generate report
    simulator.generate_report(ignition_points)
    
    print("✓ Simulation complete! Check output files:")
    print("  - fire_probability_map.png (with real locations marked)")
    print("  - fire_spread_simulation.png (showing fire progression)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()