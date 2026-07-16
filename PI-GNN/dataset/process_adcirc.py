import os
import numpy as np
import torch
import netCDF4 as nc
import datetime
from torch_geometric.data import Data

def load_adcirc_mesh(fort14_path):
    """Parses fort.14 for node coordinates, elements, and open boundaries."""
    print("Parsing fort.14 (ADCIRC Mesh)...")
    with open(fort14_path, 'r') as f:
        f.readline() # Header
        ne, nn = map(int, f.readline().split())
        
        nodes = np.zeros((nn, 3)) # lon, lat, depth
        elements = np.zeros((ne, 3), dtype=int)
        
        for i in range(nn):
            parts = f.readline().split()
            nodes[i, 0] = float(parts[1]) # lon
            nodes[i, 1] = float(parts[2]) # lat
            nodes[i, 2] = float(parts[3]) # depth
            
        for i in range(ne):
            parts = f.readline().split()
            elements[i, 0] = int(parts[2]) - 1
            elements[i, 1] = int(parts[3]) - 1
            elements[i, 2] = int(parts[4]) - 1
            
        open_boundary_nodes = []
        try:
            nope = int(f.readline().split()[0])
            neta = int(f.readline().split()[0])
            for _ in range(nope):
                num_nodes_in_seg = int(f.readline().split()[0])
                for _ in range(num_nodes_in_seg):
                    open_boundary_nodes.append(int(f.readline().strip()) - 1)
        except Exception as e:
            print(f"Warning: Could not parse open boundaries: {e}")
            
    return nodes, elements, np.array(open_boundary_nodes)

def create_graph_edges(elements):
    edges = []
    for el in elements:
        edges.extend([
            [el[0], el[1]], [el[1], el[0]],
            [el[1], el[2]], [el[2], el[1]],
            [el[2], el[0]], [el[0], el[2]]
        ])
    edges = np.unique(edges, axis=0)
    return torch.tensor(edges.T, dtype=torch.long)

def parse_fort22(fort22_path):
    """
    Parses ADCIRC fort.22 (ATCF format storm track).
    Returns lists of: times, lats, lons, vmax (knots), pc (mb).
    """
    print("Parsing fort.22 (ADCIRC Track Input)...")
    track_data = []
    with open(fort22_path, 'r') as f:
        for line in f:
            parts = line.split(',')
            if len(parts) < 10: continue
            
            # Format: YYYYMMDDHH, lat (e.g. 104N = 10.4), lon (e.g. 870E = 87.0)
            date_str = parts[2].strip()
            lat_str = parts[6].strip()
            lon_str = parts[7].strip()
            vmax = float(parts[8].strip())
            pc = float(parts[9].strip())
            
            lat = float(lat_str[:-1]) / 10.0 if lat_str[-1] == 'N' else -float(lat_str[:-1]) / 10.0
            lon = float(lon_str[:-1]) / 10.0 if lon_str[-1] == 'E' else -float(lon_str[:-1]) / 10.0
            
            time_obj = datetime.datetime.strptime(date_str, "%Y%m%d%H")
            track_data.append({'time': time_obj, 'lat': lat, 'lon': lon, 'vmax': vmax, 'pc': pc})
            
    return track_data

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def interpolate_track(track_data, target_time_steps):
    """
    Interpolates 3-hourly track data into exact simulation time steps.
    For simplicity in this skeleton, we assume linear interpolation.
    """
    # Assuming target_time_steps is just an index for now, mapping to the track.
    # In a full model, we'd map the netCDF time variable to the track datetime.
    interpolated_track = []
    for i in range(target_time_steps):
        # Placeholder: map simulation step to nearest track data point
        idx = min(i // (target_time_steps // len(track_data) + 1), len(track_data)-1)
        interpolated_track.append(track_data[idx])
    return interpolated_track

def create_sequence_dataset(f14, f22, f63, window_size=6, horizon=1):
    """
    Creates dataset using purely fort.14 (Mesh) and fort.22 (Track) as inputs.
    fort.63.nc is ONLY used for target (labels).
    """
    nodes, elements, open_boundary_nodes = load_adcirc_mesh(f14)
    edge_index = create_graph_edges(elements)
    
    # Target Data (Used ONLY for loss calculation)
    print("Loading fort.63.nc for Ground Truth Labels ONLY...")
    ds63 = nc.Dataset(f63)
    zeta = ds63.variables['zeta'][:]
    ds63.close()
    zeta = np.ma.filled(zeta, 0.0)
    time_steps, num_nodes = zeta.shape
    
    # Input Data: Track Parsing
    track_data = parse_fort22(f22)
    sim_track = interpolate_track(track_data, time_steps)
    
    dataset = []
    print(f"Building Autoregressive Input Features for {time_steps} timesteps...")
    
    depth = torch.tensor(nodes[:, 2], dtype=torch.float32).unsqueeze(1)
    lons = nodes[:, 0]
    lats = nodes[:, 1]
    
    for t in range(time_steps - window_size - horizon + 1):
        node_features = []
        for step in range(t, t + window_size):
            current_storm = sim_track[step]
            
            # Calculate dynamic parametric features for EVERY node
            dist_to_eye = haversine_distance(lons, lats, current_storm['lon'], current_storm['lat'])
            
            # Feature Tensors
            f_depth = depth.squeeze()
            f_dist = torch.tensor(dist_to_eye, dtype=torch.float32)
            f_vmax = torch.full((num_nodes,), current_storm['vmax'], dtype=torch.float32)
            f_pc = torch.full((num_nodes,), current_storm['pc'], dtype=torch.float32)
            
            # Note: No 'zeta', 'wind', or 'pressure' grids are fed in here!
            feat_t = torch.stack([f_depth, f_dist, f_vmax, f_pc], dim=1)
            node_features.append(feat_t)
            
        x = torch.stack(node_features, dim=1)
        y = torch.tensor(zeta[t + window_size + horizon - 1], dtype=torch.float32).unsqueeze(1)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
        
    print(f"Total pure-physics sequences created: {len(dataset)}")
    return dataset, open_boundary_nodes
