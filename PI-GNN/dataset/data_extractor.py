import os
import numpy as np
import torch
import netCDF4 as nc
import datetime
import math
from torch_geometric.data import Data

def haversine(lon1, lat1, lon2, lat2):
    """Calculates the exact great-circle distance between two points in meters."""
    R = 6371000 # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    a = np.clip(a, 0.0, 1.0) # CRITICAL: Prevent float precision errors from causing negative sqrt (NaN)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def generate_boundary_tides(f15, t_seconds_5min, open_boundary_nodes):
    """
    Synthesizes exact Astronomical Tides from the fort.15 input file parameters.
    No data leakage from fort.63 water levels is used!
    """
    print("Synthesizing Astronomical Tides from fort.15 Inputs...")
    
    time_steps = len(t_seconds_5min)
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
        t = t_seconds_5min[t_idx]
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

    # FIX: Use metric Cartesian coordinates for bearing, not raw degree differences.
    # Raw degrees are not isometric: 1 deg lon != 1 deg lat in metres at lat != 0.
    cos_lat_s = np.cos(np.radians(storm_lat))
    dx_m = (lons - storm_lon) * 111320.0 * cos_lat_s
    dy_m = (lats - storm_lat) * 110540.0
    theta = np.arctan2(dy_m, dx_m)

    f = 2 * omega * np.sin(np.radians(lats))

    vmax_ms = vmax_knots * 0.514444
    rmax_meters = np.maximum(47.0 - 0.41 * (pn_mb - pc_mb), 15.0) * 1000.0

    delta_p_pa = (pn_mb - pc_mb) * 100.0
    if delta_p_pa <= 0:
        return np.full_like(lons, pn_mb), np.zeros_like(lons), np.zeros_like(lons)

    B = np.clip((vmax_ms**2 * rho_air * e) / delta_p_pa, 1.0, 2.5)

    pressure_field = pc_mb + (pn_mb - pc_mb) * np.exp(-1.0 * (rmax_meters / r_meters)**B)

    term1 = (B / rho_air) * delta_p_pa * (rmax_meters / r_meters)**B * np.exp(
        -1.0 * (rmax_meters / r_meters)**B)
    term2 = (r_meters * f / 2.0)**2
    v_gradient = np.sqrt(np.maximum(term1 + term2, 0.0)) - (r_meters * np.abs(f) / 2.0)
    # FIX: v_gradient must be >= 0. Far from the eye, the Coriolis subtraction
    # can make it negative, reversing wind direction entirely.
    v_gradient = np.maximum(v_gradient, 0.0)

    inflow_angle = np.radians(15.0)
    wind_u = -v_gradient * np.sin(theta + inflow_angle)
    wind_v =  v_gradient * np.cos(theta + inflow_angle)

    # Garratt (1977) drag coefficient → wind stress [Pa]
    wind_mag = np.sqrt(wind_u**2 + wind_v**2)
    Cd = np.clip((0.75 + 0.067 * wind_mag) * 1e-3, 0.0, 0.0035)
    tau_x = Cd * rho_air * wind_u * wind_mag
    tau_y = Cd * rho_air * wind_v * wind_mag

    return pressure_field, tau_x, tau_y

def create_full_simulation_dataset(f14, f22, f63):
    """Returns a single massive forcing tensor covering all timesteps."""
    nodes, elements, open_boundary_nodes = load_adcirc_mesh(f14)
    edge_index = create_graph_edges(elements)
    
    # Precompute edge geometry ONCE for pressure gradient scatter
    # (same edges used later for edge_weight; done here in numpy for speed)
    src_np = edge_index[0].numpy()
    dst_np = edge_index[1].numpy()
    cos_lat_val = np.cos(np.radians(np.mean(nodes[:, 1])))
    _lon_m = 111320.0 * cos_lat_val   # m per degree longitude
    _lat_m = 110540.0                  # m per degree latitude
    _dx = (nodes[dst_np, 0] - nodes[src_np, 0]) * _lon_m
    _dy = (nodes[dst_np, 1] - nodes[src_np, 1]) * _lat_m
    _dist = np.sqrt(_dx**2 + _dy**2 + 1e-6)
    _cos_e = _dx / _dist
    _sin_e = _dy / _dist
    _inv_e = 1.0 / _dist
    _N = len(nodes)
    _degree = np.zeros(_N)
    np.add.at(_degree, src_np, 1)
    np.add.at(_degree, dst_np, 1)
    _degree = np.maximum(_degree, 1.0)
    
    # Load Target Data (ONLY for loss)
    print("Loading fort.63.nc to generate target loss arrays...")
    ds63 = nc.Dataset(f63)
    orig_t_seconds = ds63.variables['time'][:]
    orig_zeta = ds63.variables['zeta'][:]   # masked array
    ds63.close()
    # FIX: Fill dry/land nodes with -9999.0 sentinel (NOT 0.0).
    # Filling with 0.0 incorrectly trains the model to predict sea-level
    # at land nodes, polluting the loss with ~50% meaningless targets.
    orig_zeta = np.ma.filled(orig_zeta, -9999.0)

    dt_seconds = 900.0
    start_time = orig_t_seconds[0]
    end_time   = orig_t_seconds[-1]
    t_seconds_5min = np.arange(start_time, end_time, dt_seconds)
    time_steps = len(t_seconds_5min)

    print(f"Generating 15-min interpolated timeline ({time_steps} steps)...")

    from scipy.interpolate import interp1d
    interp_func_zeta = interp1d(orig_t_seconds, orig_zeta, axis=0, fill_value="extrapolate")
    zeta_5min = interp_func_zeta(t_seconds_5min)

    # Build wet mask BEFORE zeroing dry nodes.
    # A node is considered wet if it has a valid water level (> -9000).
    wet_mask_np = zeta_5min > -9000.0          # [T, N]  True = wet
    zeta_5min[~wet_mask_np] = 0.0              # zero out sentinels for numerics
    true_zetas = torch.tensor(zeta_5min, dtype=torch.float32).unsqueeze(2)  # [T, N, 1]
    wet_mask   = torch.tensor(wet_mask_np, dtype=torch.bool)                 # [T, N]
    
    track_data = parse_fort22(f22)
    
    # Interpolate the track data to perfectly match the dense 5-minute output timesteps!
    orig_indices = np.linspace(start_time, end_time, len(track_data))
    
    orig_lons = np.array([pt['lon'] for pt in track_data])
    orig_lats = np.array([pt['lat'] for pt in track_data])
    orig_vmax = np.array([pt['vmax'] for pt in track_data])
    orig_pc = np.array([pt['pc'] for pt in track_data])
    
    interp_func_lons = interp1d(orig_indices, orig_lons, fill_value="extrapolate")
    interp_func_lats = interp1d(orig_indices, orig_lats, fill_value="extrapolate")
    interp_func_vmax = interp1d(orig_indices, orig_vmax, fill_value="extrapolate")
    interp_func_pc = interp1d(orig_indices, orig_pc, fill_value="extrapolate")
    
    interp_lons = interp_func_lons(t_seconds_5min)
    interp_lats = interp_func_lats(t_seconds_5min)
    interp_vmax = interp_func_vmax(t_seconds_5min)
    interp_pc = interp_func_pc(t_seconds_5min)
    
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
        
        # ============================================================
        # Pressure GRADIENT (Pa/m) — the actual hydrodynamic forcing.
        # Raw pressure at a node is meaningless; only ∂P/∂x drives water.
        # Computed via FD scatter on the mesh graph (same operator as
        # the physics loss): ∂P/∂x_i = Σ_j (P_j-P_i)·cos_θ·inv_dist / deg_i
        # ============================================================
        p_pa = p_field * 100.0           # mb → Pa
        dp_edge = p_pa[dst_np] - p_pa[src_np]   # [E]
        grad_px = np.zeros(_N)
        grad_py = np.zeros(_N)
        np.add.at(grad_px, dst_np, -dp_edge * _cos_e * _inv_e)
        np.add.at(grad_px, src_np,  dp_edge * _cos_e * _inv_e)
        np.add.at(grad_py, dst_np, -dp_edge * _sin_e * _inv_e)
        np.add.at(grad_py, src_np,  dp_edge * _sin_e * _inv_e)
        grad_px /= _degree    # Pa/m  x-direction
        grad_py /= _degree    # Pa/m  y-direction
        
        f_depth   = depth.squeeze()
        f_grad_px = torch.tensor(grad_px, dtype=torch.float32)   # ∂P/∂x  [Pa/m]
        f_grad_py = torch.tensor(grad_py, dtype=torch.float32)   # ∂P/∂y  [Pa/m]
        # u_field / v_field from holland_wind_model are already τ_x / τ_y [Pa]
        f_tau_x   = torch.tensor(u_field, dtype=torch.float32)
        f_tau_y   = torch.tensor(v_field, dtype=torch.float32)
        f_n       = mannings_n.squeeze()
        
        # 6 Features: Depth, dP/dx, dP/dy, τ_x, τ_y, Manning's N
        feat_t = torch.stack([f_depth, f_grad_px, f_grad_py, f_tau_x, f_tau_y, f_n], dim=1)
        forcing_sequence.append(feat_t)
        
    forcing_sequence = torch.stack(forcing_sequence, dim=0) # [time_steps, num_nodes, 4]
    
    # Generate Legal Boundary Forcing from fort.15 explicitly on the 15-minute timeline
    boundary_tides = generate_boundary_tides(f14.replace('fort.14', 'fort.15'), t_seconds_5min, open_boundary_nodes)
    
    # === SPATIAL PHYSICS (EDGE WEIGHTS) ===
    # Calculate exact physical distance between connected nodes to allow GNN to compute spatial gradients (wave slopes)
    print("Computing Haversine Spatial Gradients for Edges...")
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    dists = haversine(nodes[src, 0], nodes[src, 1], nodes[dst, 0], nodes[dst, 1])
    dists = np.clip(dists, 1.0, None) # Prevent divide by zero
    edge_weight = torch.tensor(1.0 / dists, dtype=torch.float32)

    return (forcing_sequence, edge_index, edge_weight, true_zetas,
            open_boundary_nodes, boundary_tides, nodes, wet_mask)
