import matplotlib.pyplot as plt
import numpy as np

def plot_hydrograph(time_array, adcirc_elevation, gnn_elevation, node_id, save_path=None):
    """
    Plots the time-series water elevation comparison at a specific node.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_array, adcirc_elevation, label='ADCIRC (Truth)', color='blue', linewidth=2)
    plt.plot(time_array, gnn_elevation, label='PI-GNN (Predicted)', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Water Surface Elevation at Node {node_id}")
    plt.xlabel("Time Steps")
    plt.ylabel("Elevation (m)")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
