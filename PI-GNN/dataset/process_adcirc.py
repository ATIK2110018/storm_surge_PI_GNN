import os
import numpy as np
import torch
import netCDF4 as nc
import datetime
import math
from torch_geometric.data import Data

def generate_boundary_tides(f15, f63, open_boundary_nodes):
    """
    Synthesizes exact Astronomical Tides from the fort.15 input file parameters.
    No data leakage from fort.63 water levels is used!
    """
    print("Synthesizing Astronomical Tides from fort.15 Inputs...")
    ds63 = nc.Dataset(f63)
    t_seconds = ds63.variables['time'][:] # Get the exact simulation time in seconds
    ds63.close()
    
    time_steps = len(t_seconds)
    num_bnodes = len(open_boundary_nodes)
    boundary_tides = np.zeros((time_steps, num_bnodes))
    
    with open(f15, 'r') as f:
        lines = f.readlines()
        
    nbfr = 0
    start_idx = 0
    for i, line in enumerate(lines):
        if 'NBFR' in line:
            nbfr = int(line.split()[0])
            start_idx = i + 1
            break
            
    freqs = []
    idx = start_idx
    for k in range(nbfr):
        name = lines[idx].strip()
        idx += 1
        parts = lines[idx].split()
        freqs.append({
            'name': name,
            'amigt': float(parts[0]),
            'fft': float(parts[1]),
            'facet': float(parts[2])
        })
        idx += 1
        
    for k in range(nbfr):
        idx += 1 # Skip Name
        emo = np.zeros(num_bnodes)
        efa = np.zeros(num_bnodes)
        for j in range(num_bnodes):
            parts = lines[idx].split()
            emo[j] = float(parts[0])
            efa[j] = float(parts[1])
            idx += 1
        freqs[k]['emo'] = emo
        freqs[k]['efa'] = efa
        
    for t_idx in range(time_steps):
        t = t_seconds[t_idx]
        zeta = np.zeros(num_bnodes)
        for k in range(nbfr):
            amigt = freqs[k]['amigt']
            fft = freqs[k]['fft']
            facet = freqs[k]['facet']
            emo = freqs[k]['emo']
            efa = freqs[k]['efa']
            
            phase = (math.pi / 180.0) * (facet - efa)
            zeta += fft * emo * np.cos(amigt * t + phase)
            
        boundary_tides[t_idx, :] = zeta
        
    return torch.tensor(boundary_tides, dtype=torch.float32)

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
    
    # Convert scalar wind to U, V components (inflowing cyclonic swirl)
    inflow_angle = np.radians(15.0)
    wind_u = -v_gradient * np.sin(theta + inflow_angle)
    wind_v = v_gradient * np.cos(theta + inflow_angle)
    
    # === EXPLICIT WIND STRESS CONVERSION ===
    # Garratt's Drag Coefficient Formula
    wind_mag = np.sqrt(wind_u**2 + wind_v**2)
    Cd = (0.75 + 0.067 * wind_mag) * 1e-3
    Cd = np.clip(Cd, 0.0, 0.0035) # Cap drag at extreme hurricane speeds
    
    # Wind Stress (tau)
    tau_x = Cd * rho_air * wind_u * wind_mag
    tau_y = Cd * rho_air * wind_v * wind_mag
    
    return pressure_field, tau_x, tau_y

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
    
    # Interpolate the track data to perfectly match the fort.63 output timesteps!
    # This prevents the cyclone from moving too fast or too slow relative to the target data
    orig_indices = np.linspace(0, 1, len(track_data))
    target_indices = np.linspace(0, 1, time_steps)
    
    orig_lons = np.array([pt['lon'] for pt in track_data])
    orig_lats = np.array([pt['lat'] for pt in track_data])
    orig_vmax = np.array([pt['vmax'] for pt in track_data])
    orig_pc = np.array([pt['pc'] for pt in track_data])
    
    interp_lons = np.interp(target_indices, orig_indices, orig_lons)
    interp_lats = np.interp(target_indices, orig_indices, orig_lats)
    interp_vmax = np.interp(target_indices, orig_indices, orig_vmax)
    interp_pc = np.interp(target_indices, orig_indices, orig_pc)
    
    print(f"Building Forcing Tensors for {time_steps} interpolated timesteps...")
    depth = torch.tensor(nodes[:, 2], dtype=torch.float32).unsqueeze(1)
    
    # Spatially differing Manning's n from Depth (ADCIRC logic)
    # Deep water = 0.02, Coastal = 0.035, Land = 0.10
    mannings_n = torch.where(depth > 20.0, torch.tensor(0.02),
                 torch.where(depth > 2.0, torch.tensor(0.035),
                 torch.where(depth > 0.0, torch.tensor(0.05), torch.tensor(0.10))))
    
    lons, lats = nodes[:, 0], nodes[:, 1]
    
    forcing_sequence = []
    
    for t in range(time_steps):
        lon = interp_lons[t]
        lat = interp_lats[t]
        vmax = interp_vmax[t]
        pc = interp_pc[t]
        
        p_field, u_field, v_field = holland_wind_model(lons, lats, lon, lat, vmax, pc)
        
        f_depth = depth.squeeze()
        f_press = torch.tensor(p_field, dtype=torch.float32)
        # Note: windu and windv are now technically tau_x and tau_y (Wind Stress)
        f_windu = torch.tensor(u_field, dtype=torch.float32)
        f_windv = torch.tensor(v_field, dtype=torch.float32)
        f_n = mannings_n.squeeze()
        
        # Now 5 Features: Depth, Pressure, Tau_X, Tau_Y, Manning's N
        feat_t = torch.stack([f_depth, f_press, f_windu, f_windv, f_n], dim=1)
        forcing_sequence.append(feat_t)
        
    forcing_sequence = torch.stack(forcing_sequence, dim=0) # [time_steps, num_nodes, 4]
    true_zetas = torch.tensor(zeta, dtype=torch.float32).unsqueeze(2) # [time_steps, num_nodes, 1]
    
    # Generate Legal Boundary Forcing from fort.15
    boundary_tides = generate_boundary_tides(f14.replace('fort.14', 'fort.15'), f63, open_boundary_nodes)
    
    return forcing_sequence, edge_index, true_zetas, open_boundary_nodes, boundary_tides
