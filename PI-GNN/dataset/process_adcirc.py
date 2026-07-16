import os
import numpy as np
import torch
import netCDF4 as nc
import datetime
from torch_geometric.data import Data

def load_adcirc_mesh(fort14_path):
    print("Parsing fort.14 (ADCIRC Mesh)...")
    with open(fort14_path, 'r') as f:
        f.readline()
        ne, nn = map(int, f.readline().split())
        
        nodes = np.zeros((nn, 3))
        elements = np.zeros((ne, 3), dtype=int)
        
        for i in range(nn):
            parts = f.readline().split()
            nodes[i, 0], nodes[i, 1], nodes[i, 2] = float(parts[1]), float(parts[2]), float(parts[3])
            
        for i in range(ne):
            parts = f.readline().split()
            elements[i, 0], elements[i, 1], elements[i, 2] = int(parts[2])-1, int(parts[3])-1, int(parts[4])-1
            
        open_boundary_nodes = []
        try:
            nope = int(f.readline().split()[0])
            neta = int(f.readline().split()[0])
            for _ in range(nope):
                num_nodes_in_seg = int(f.readline().split()[0])
                for _ in range(num_nodes_in_seg):
                    open_boundary_nodes.append(int(f.readline().strip()) - 1)
        except:
            pass
            
    return nodes, elements, np.array(open_boundary_nodes)

def create_graph_edges(elements):
    edges = []
    for el in elements:
        edges.extend([[el[0], el[1]], [el[1], el[0]], [el[1], el[2]], [el[2], el[1]], [el[2], el[0]], [el[0], el[2]]])
    edges = np.unique(edges, axis=0)
    return torch.tensor(edges.T, dtype=torch.long)

def parse_fort22(fort22_path):
    track_data = []
    with open(fort22_path, 'r') as f:
        for line in f:
            parts = line.split(',')
            if len(parts) < 10: continue
            lat_str, lon_str = parts[6].strip(), parts[7].strip()
            lat = float(lat_str[:-1]) / 10.0 if lat_str[-1] == 'N' else -float(lat_str[:-1]) / 10.0
            lon = float(lon_str[:-1]) / 10.0 if lon_str[-1] == 'E' else -float(lon_str[:-1]) / 10.0
            track_data.append({'lat': lat, 'lon': lon, 'vmax': float(parts[8].strip()), 'pc': float(parts[9].strip())})
    return track_data

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def holland_wind_model(lons, lats, storm_lon, storm_lat, vmax_knots, pc_mb, pn_mb=1010.0):
    rho_air, e, omega = 1.15, np.exp(1), 7.2921e-5
    r_km = np.maximum(haversine_distance(lons, lats, storm_lon, storm_lat), 0.1)
    r_meters = r_km * 1000.0
    theta = np.arctan2(np.radians(lats - storm_lat), np.radians(lons - storm_lon))
    f = 2 * omega * np.sin(np.radians(lats))
    
    vmax_ms = vmax_knots * 0.514444
    rmax_meters = np.maximum(47.0 - 0.41 * (pn_mb - pc_mb), 15.0) * 1000.0
    
    delta_p_pa = (pn_mb - pc_mb) * 100.0
    if delta_p_pa <= 0: return np.full_like(lons, pn_mb), np.zeros_like(lons), np.zeros_like(lons)
        
    B = np.clip((vmax_ms**2 * rho_air * e) / delta_p_pa, 1.0, 2.5)
    
    pressure_field = pc_mb + (pn_mb - pc_mb) * np.exp(-1.0 * (rmax_meters / r_meters)**B)
    
    term1 = (B / rho_air) * delta_p_pa * (rmax_meters / r_meters)**B * np.exp(-1.0 * (rmax_meters / r_meters)**B)
    term2 = (r_meters * f / 2.0)**2
    v_gradient = np.sqrt(term1 + term2) - (r_meters * np.abs(f) / 2.0)
    
    inflow_angle = np.radians(15.0)
    wind_u = -v_gradient * np.sin(theta + inflow_angle)
    wind_v = v_gradient * np.cos(theta + inflow_angle)
    
    return pressure_field, wind_u, wind_v

def create_full_simulation_dataset(f14, f22, f63):
    """Returns a single massive forcing tensor covering all timesteps."""
    nodes, elements, open_boundary_nodes = load_adcirc_mesh(f14)
    edge_index = create_graph_edges(elements)
    
    # Target Data (ONLY for loss)
    print("Loading fort.63.nc for Ground Truth Labels ONLY...")
    ds63 = nc.Dataset(f63)
    zeta = ds63.variables['zeta'][:]
    ds63.close()
    zeta = np.ma.filled(zeta, 0.0)
    time_steps, num_nodes = zeta.shape
    
    track_data = parse_fort22(f22)
    
    print(f"Building Forcing Tensors for {time_steps} timesteps...")
    depth = torch.tensor(nodes[:, 2], dtype=torch.float32).unsqueeze(1)
    lons, lats = nodes[:, 0], nodes[:, 1]
    
    forcing_sequence = []
    
    for t in range(time_steps):
        # Linearly map timestep to storm track (simplified mapping)
        idx = min(t // (time_steps // len(track_data) + 1), len(track_data)-1)
        current_storm = track_data[idx]
        
        p_field, u_field, v_field = holland_wind_model(lons, lats, current_storm['lon'], current_storm['lat'], current_storm['vmax'], current_storm['pc'])
        
        f_depth = depth.squeeze()
        f_press = torch.tensor(p_field, dtype=torch.float32)
        f_windu = torch.tensor(u_field, dtype=torch.float32)
        f_windv = torch.tensor(v_field, dtype=torch.float32)
        
        feat_t = torch.stack([f_depth, f_press, f_windu, f_windv], dim=1)
        forcing_sequence.append(feat_t)
        
    forcing_sequence = torch.stack(forcing_sequence, dim=0) # [time_steps, num_nodes, 4]
    true_zetas = torch.tensor(zeta, dtype=torch.float32).unsqueeze(2) # [time_steps, num_nodes, 1]
    
    return forcing_sequence, edge_index, true_zetas, open_boundary_nodes
