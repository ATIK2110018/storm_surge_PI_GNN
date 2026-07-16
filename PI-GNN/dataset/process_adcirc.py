import os
import numpy as np
import torch
import netCDF4 as nc
from torch_geometric.data import Data

def load_adcirc_mesh(fort14_path):
    """
    Parses the ADCIRC fort.14 (mesh) file.
    Returns node coordinates (x, y, depth) and element connectivity.
    """
    with open(fort14_path, 'r') as f:
        f.readline() # Header
        ne, nn = map(int, f.readline().split())
        
        nodes = np.zeros((nn, 3))
        elements = np.zeros((ne, 3), dtype=int)
        
        for i in range(nn):
            parts = f.readline().split()
            nodes[i, 0] = float(parts[1]) # x
            nodes[i, 1] = float(parts[2]) # y
            nodes[i, 2] = float(parts[3]) # depth
            
        for i in range(ne):
            parts = f.readline().split()
            elements[i, 0] = int(parts[2]) - 1
            elements[i, 1] = int(parts[3]) - 1
            elements[i, 2] = int(parts[4]) - 1
            
        # Parse Open Boundaries
        open_boundary_nodes = []
        try:
            nope_line = f.readline().split()
            nope = int(nope_line[0]) # Number of open boundaries
            neta_line = f.readline().split()
            neta = int(neta_line[0]) # Total number of open boundary nodes
            
            for _ in range(nope):
                seg_info = f.readline().split()
                num_nodes_in_seg = int(seg_info[0])
                for _ in range(num_nodes_in_seg):
                    node_id = int(f.readline().strip()) - 1
                    open_boundary_nodes.append(node_id)
        except:
            print("Warning: Could not parse open boundaries from fort.14. Returning empty list.")
            
    return nodes, elements, np.array(open_boundary_nodes)

def create_graph_edges(elements):
    """
    Creates PyG edge_index format.
    """
    edges = []
    for el in elements:
        edges.extend([
            [el[0], el[1]], [el[1], el[0]],
            [el[1], el[2]], [el[2], el[1]],
            [el[2], el[0]], [el[0], el[2]]
        ])
    edges = np.unique(edges, axis=0)
    return torch.tensor(edges.T, dtype=torch.long)

def load_dynamic_data(fort63_path, fort73_path, fort74_path):
    """
    Loads water elevation, atmospheric pressure, and wind velocities.
    """
    print("Loading Water Elevation (fort.63.nc)...")
    ds63 = nc.Dataset(fort63_path)
    zeta = ds63.variables['zeta'][:]
    ds63.close()
    
    print("Loading Atmospheric Pressure (fort.73.nc)...")
    try:
        ds73 = nc.Dataset(fort73_path)
        pressure = ds73.variables['pressure'][:]
        ds73.close()
    except:
        print("Warning: fort.73.nc not found or invalid. Using zeros for pressure.")
        pressure = np.zeros_like(zeta)

    print("Loading Wind Velocity (fort.74.nc)...")
    try:
        ds74 = nc.Dataset(fort74_path)
        windx = ds74.variables['windx'][:]
        windy = ds74.variables['windy'][:]
        ds74.close()
    except:
        print("Warning: fort.74.nc not found or invalid. Using zeros for wind.")
        windx = np.zeros_like(zeta)
        windy = np.zeros_like(zeta)

    # Convert masked arrays to regular arrays, filling missing/dry nodes with 0
    zeta = np.ma.filled(zeta, 0.0)
    pressure = np.ma.filled(pressure, 0.0)
    windx = np.ma.filled(windx, 0.0)
    windy = np.ma.filled(windy, 0.0)
    
    return (
        torch.tensor(zeta, dtype=torch.float32), 
        torch.tensor(pressure, dtype=torch.float32), 
        torch.tensor(windx, dtype=torch.float32), 
        torch.tensor(windy, dtype=torch.float32)
    )

def create_sequence_dataset(f14, f63, f73, f74, window_size=6, horizon=1, max_time_steps=None):
    """
    Creates a temporal sequence dataset.
    Returns: dataset, open_boundary_nodes
    """
    print("Parsing mesh and boundaries...")
    nodes, elements, open_boundary_nodes = load_adcirc_mesh(f14)
    edge_index = create_graph_edges(elements)
    
    print(f"Found {len(open_boundary_nodes)} Open Boundary Nodes!")
    
    zeta, pressure, windx, windy = load_dynamic_data(f63, f73, f74)
    time_steps, num_nodes = zeta.shape
    
    if max_time_steps is not None and max_time_steps < time_steps:
        print(f"Limiting to first {max_time_steps} time steps for testing...")
        time_steps = max_time_steps
        zeta = zeta[:time_steps]
        pressure = pressure[:time_steps]
        windx = windx[:time_steps]
        windy = windy[:time_steps]
    
    # Static features: Depth (normalized)
    depth = torch.tensor(nodes[:, 2], dtype=torch.float32).unsqueeze(1)
    depth = (depth - depth.mean()) / (depth.std() + 1e-8)
    
    dataset = []
    
    print(f"Building temporal sequences (Window={window_size}, Horizon={horizon})...")
    # Sliding window over time
    for t in range(time_steps - window_size - horizon + 1):
        # Input features for the window: [Nodes, Features, Time]
        # Features: [Depth, Zeta, Pressure, WindX, WindY] -> 5 features
        
        node_features = []
        for step in range(t, t + window_size):
            feat_t = torch.stack([
                depth.squeeze(),
                zeta[step],
                pressure[step],
                windx[step],
                windy[step]
            ], dim=1)
            node_features.append(feat_t)
            
        # Shape: [num_nodes, num_features * window_size]
        # Flattening the temporal dimension into the feature dimension for basic GNN, 
        # or keeping it structured for a Recurrent GNN. 
        # Here we stack them: shape [num_nodes, window_size, features]
        x = torch.stack(node_features, dim=1) 
        
        # Target: Water elevation at t + window_size + horizon - 1
        y = zeta[t + window_size + horizon - 1].unsqueeze(1)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
        
    print(f"Total sequences created: {len(dataset)}")
    return dataset, open_boundary_nodes
