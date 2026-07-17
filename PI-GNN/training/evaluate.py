import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.st_gnn import AutoregressiveSurrogate
from dataset.process_adcirc import create_full_simulation_dataset

def evaluate():
    print("=== PI-GNN Engineering Evaluation ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_io'))
    f14 = os.path.join(base_dir, 'fort.14')
    f22 = os.path.join(base_dir, 'fort.22')
    f63 = os.path.join(base_dir, 'fort.63.nc')
    
    print("1. Re-loading the test data and trained model...")
    forcing_sequence, edge_index, true_zetas, open_boundary_nodes, boundary_tides = create_full_simulation_dataset(f14, f22, f63)
    
    num_nodes = forcing_sequence.size(1)
    # Model uses 5 features (Depth, P, U, V, Manning)
    model = AutoregressiveSurrogate(num_nodes=num_nodes, num_forcing_features=5).to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'pi_gnn_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("2. Running the Full Autoregressive Simulation...")
    forcing_sequence = forcing_sequence.to(device)
    edge_index = edge_index.to(device)
    boundary_tides = boundary_tides.to(device)
    
    with torch.no_grad():
        simulated_zetas, _, _, _ = model(forcing_sequence, edge_index, open_boundary_nodes, boundary_tides)
        
    preds_array = simulated_zetas.cpu().numpy().squeeze()
    truth_array = true_zetas.numpy().squeeze()
    
    print("3. Generating Engineering Visualizations...")
    
    # 3A: Hydrographs
    test_nodes = [1000, 5000, 15000] 
    time_axis = np.arange(preds_array.shape[0])
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Storm Surge Hydrographs: ADCIRC vs PI-GNN', fontsize=16)
    for i, node_id in enumerate(test_nodes):
        if node_id < num_nodes:
            axs[i].plot(time_axis, truth_array[:, node_id], label='ADCIRC', color='black', linewidth=2)
            axs[i].plot(time_axis, preds_array[:, node_id], label='PI-GNN (Self-Simulated)', color='red', linestyle='--', linewidth=2)
            axs[i].set_ylabel('Elevation (m)')
            axs[i].set_title(f'Coastal Node ID: {node_id}')
            axs[i].legend()
            axs[i].grid(True)
    axs[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig('hydrographs.png')
    print("Saved 'hydrographs.png'")
    
    # 3B: Peak Surge Scatter
    peak_truth = np.max(truth_array, axis=0)
    peak_preds = np.max(preds_array, axis=0)
    valid_idx = (peak_truth > 0.1)
    peak_truth_valid = peak_truth[valid_idx]
    peak_preds_valid = peak_preds[valid_idx]
    
    if len(peak_truth_valid) > 0:
        r2 = r2_score(peak_truth_valid, peak_preds_valid)
        rmse = np.sqrt(mean_squared_error(peak_truth_valid, peak_preds_valid))
        plt.figure(figsize=(8, 8))
        plt.scatter(peak_truth_valid, peak_preds_valid, alpha=0.3, color='blue', s=2)
        plt.plot([0, np.max(peak_truth_valid)], [0, np.max(peak_truth_valid)], 'r--', label='Perfect Agreement (1:1)')
        plt.title(f"Peak Surge Correlation\n$R^2$: {r2:.3f} | RMSE: {rmse:.3f} m")
        plt.xlabel("ADCIRC Peak Surge (m)")
        plt.ylabel("PI-GNN Peak Surge (m)")
        plt.legend()
        plt.grid(True)
        plt.savefig('peak_scatter.png')
        print("Saved 'peak_scatter.png'")
        
    # 3C: Spatial Error Map
    timestep = 50
    if timestep < preds_array.shape[0]:
        spatial_error = np.abs(truth_array[timestep, :] - preds_array[timestep, :])
        
        # Load elements for triangulation
        with open(f14, 'r') as f:
            f.readline()
            ne, nn = map(int, f.readline().split())
            x, y = np.zeros(nn), np.zeros(nn)
            elements = np.zeros((ne, 3), dtype=int)
            for i in range(nn):
                parts = f.readline().split()
                x[i], y[i] = float(parts[1]), float(parts[2])
            for i in range(ne):
                parts = f.readline().split()
                elements[i,0], elements[i,1], elements[i,2] = int(parts[2])-1, int(parts[3])-1, int(parts[4])-1
        
        triangulation = tri.Triangulation(x, y, elements)
        plt.figure(figsize=(12, 10))
        plt.title(f"Absolute Spatial Error (m) at t={timestep}", fontsize=16)
        contour = plt.tricontourf(triangulation, spatial_error, levels=30, cmap='Reds')
        plt.colorbar(contour, label='Absolute Error |ADCIRC - GNN| (m)')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig('spatial_error.png')
        print("Saved 'spatial_error.png'")
    
    # 3D: Boundary Node Visualization
    if len(open_boundary_nodes) > 0:
        import random
        rand_boundary_node = random.choice(open_boundary_nodes)
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, truth_array[:, rand_boundary_node], label='ADCIRC Boundary (True Tide)', color='black', linewidth=2)
        plt.plot(time_axis, preds_array[:, rand_boundary_node], label='PI-GNN Boundary Prediction', color='blue', linestyle='--', linewidth=2)
        plt.title(f"Boundary Node {rand_boundary_node} Water Level")
        plt.ylabel("Elevation (m)")
        plt.xlabel("Time Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig('boundary_node.png')
        print("Saved 'boundary_node.png'")

if __name__ == "__main__":
    evaluate()
