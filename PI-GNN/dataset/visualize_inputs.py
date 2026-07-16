import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import sys

# Ensure Python path finds PI-GNN module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.process_adcirc import load_adcirc_mesh, parse_fort22, holland_wind_model

def visualize():
    print("=== PI-GNN Domain & Physics Forcing Visualization ===")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_io'))
    f14 = os.path.join(base_dir, 'fort.14')
    f22 = os.path.join(base_dir, 'fort.22')
    
    print("Loading Mesh, Boundaries, and Track...")
    nodes, elements, open_boundary_nodes = load_adcirc_mesh(f14)
    track_data = parse_fort22(f22)
    
    x, y, depth = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    triangulation = tri.Triangulation(x, y, elements)
    
    track_lons = [pt['lon'] for pt in track_data]
    track_lats = [pt['lat'] for pt in track_data]
    
    # PLOT 1: Domain
    plt.figure(figsize=(12, 10))
    plt.title("PI-GNN Domain: Bathymetry, Boundaries & Cyclone Track", fontsize=16)
    contour = plt.tricontourf(triangulation, depth, levels=50, cmap='ocean_r')
    plt.colorbar(contour, label='Depth (m)')
    plt.triplot(triangulation, color='black', alpha=0.1, linewidth=0.1)
    
    if len(open_boundary_nodes) > 0:
        plt.scatter(x[open_boundary_nodes], y[open_boundary_nodes], color='red', s=10, label='Open Ocean Boundary')
        
    plt.plot(track_lons, track_lats, color='orange', linewidth=2, linestyle='--', label='Cyclone Track')
    plt.scatter(track_lons, track_lats, color='yellow', edgecolor='black', s=30, zorder=5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='lower right')
    plt.savefig('domain_visualization.png')
    print("Saved 'domain_visualization.png'")
    
    # PLOT 2: Holland Physics
    print("Generating Analytical Holland Wind/Pressure Fields...")
    step = 18
    storm = track_data[step]
    
    p_field, u_field, v_field = holland_wind_model(x, y, storm['lon'], storm['lat'], storm['vmax'], storm['pc'])
    wind_mag = np.sqrt(u_field**2 + v_field**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    c1 = ax1.tricontourf(triangulation, wind_mag, levels=30, cmap='YlOrRd')
    ax1.plot(track_lons, track_lats, color='black', linestyle='--', alpha=0.5)
    ax1.scatter([storm['lon']], [storm['lat']], color='red', marker='X', s=200, label="Storm Eye")
    ax1.set_title(f"Holland Wind Magnitude (m/s)\nVmax: {storm['vmax']} kts")
    fig.colorbar(c1, ax=ax1, label='Wind Speed (m/s)')
    ax1.legend()
    
    c2 = ax2.tricontourf(triangulation, p_field, levels=30, cmap='coolwarm')
    ax2.plot(track_lons, track_lats, color='black', linestyle='--', alpha=0.5)
    ax2.scatter([storm['lon']], [storm['lat']], color='red', marker='X', s=200)
    ax2.set_title(f"Holland Atmospheric Pressure (mb)\nPc: {storm['pc']} mb")
    fig.colorbar(c2, ax=ax2, label='Pressure (mb)')
    
    plt.savefig('holland_physics_visualization.png')
    print("Saved 'holland_physics_visualization.png'")

if __name__ == "__main__":
    visualize()
