"""
create_inference_dataset: builds the forcing tensor from storm track + mesh
WITHOUT requiring fort.63.nc (no truth labels needed).
This enables parametric inference for any storm in the same domain.
"""
import numpy as np
import torch
from scipy.interpolate import interp1d
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_extractor import (
    parse_fort14, create_graph_edges, holland_wind_model,
    generate_boundary_tides, haversine
)


def build_forcing_from_track(nodes, edge_index, track_df,
                              fort15_path, open_boundary_nodes,
                              duration_hours=None, dt_minutes=15):
    """
    Build the forcing_sequence, boundary_tides, nodes_xy, edge_weight
    from a storm track DataFrame — NO fort.63.nc required.

    Parameters
    ----------
    nodes : ndarray [N, 3]  — lon, lat, depth (from parse_fort14)
    edge_index : LongTensor [2, E]
    track_df : pandas.DataFrame with columns:
                  time_s   : seconds since epoch (float, monotone increasing)
                  lat      : storm centre latitude (deg N)
                  lon      : storm centre longitude (deg E)
                  vmax_kt  : max sustained wind speed (knots)
                  pc_mb    : minimum central pressure (mb)
    fort15_path : str  — path to fort.15 (for tidal BCs)
    open_boundary_nodes : ndarray of int
    duration_hours : float or None.  If None, inferred from track_df.
    dt_minutes : int  — output timestep (default 15 min)

    Returns
    -------
    forcing_sequence : FloatTensor [T, N, 7]
    boundary_tides   : FloatTensor [T, n_obn]
    nodes_xy         : ndarray [N, 2]  (lon, lat)
    edge_weight      : FloatTensor [E]
    t_seconds        : ndarray [T]  — absolute seconds for each timestep
    """
    import pandas as pd

    # ---- Time grid ----
    t0 = float(track_df['time_s'].iloc[0])
    if duration_hours is None:
        t_end = float(track_df['time_s'].iloc[-1])
    else:
        t_end = t0 + duration_hours * 3600.0
    dt_s = dt_minutes * 60.0
    t_seconds = np.arange(t0, t_end + dt_s, dt_s)
    t_seconds = t_seconds[t_seconds <= t_end]
    time_steps = len(t_seconds)
    print(f"   Time grid: {time_steps} steps × {dt_minutes} min = {time_steps * dt_minutes / 60:.1f} h")

    # ---- Interpolate track to model timestep ----
    interp_lat  = interp1d(track_df['time_s'], track_df['lat'],  fill_value='extrapolate')(t_seconds)
    interp_lon  = interp1d(track_df['time_s'], track_df['lon'],  fill_value='extrapolate')(t_seconds)
    interp_vmax = interp1d(track_df['time_s'], track_df['vmax_kt'], fill_value='extrapolate')(t_seconds)
    interp_pc   = interp1d(track_df['time_s'], track_df['pc_mb'],   fill_value='extrapolate')(t_seconds)

    # ---- Holland wind + pressure at every node & timestep ----
    lons = nodes[:, 0]
    lats = nodes[:, 1]
    depth_np = nodes[:, 2].astype(np.float32)

    print(f"   Running Holland wind model ({time_steps} steps, vectorized)...")
    p_pa_list, tau_x_list, tau_y_list = [], [], []
    for t in range(time_steps):
        p_field, tx, ty = holland_wind_model(
            lons, lats, interp_lon[t], interp_lat[t],
            interp_vmax[t], interp_pc[t])
        p_pa_list.append(p_field * 100.0)
        tau_x_list.append(tx)
        tau_y_list.append(ty)

    p_pa_all  = np.stack(p_pa_list,  axis=0).astype(np.float32)
    tau_x_all = np.stack(tau_x_list, axis=0).astype(np.float32)
    tau_y_all = np.stack(tau_y_list, axis=0).astype(np.float32)

    # ---- Pressure gradient via graph edge differences ----
    src_np = edge_index[0].numpy()
    dst_np = edge_index[1].numpy()
    N = nodes.shape[0]

    from scipy.sparse import csr_matrix
    cos_lat = np.cos(np.radians(lats))
    dx_m = (lons[dst_np] - lons[src_np]) * 111320.0 * cos_lat[src_np]
    dy_m = (lats[dst_np] - lats[src_np]) * 110540.0
    dist_m = np.maximum(np.sqrt(dx_m**2 + dy_m**2), 1.0)

    rows_x = np.concatenate([src_np, dst_np])
    cols_x = np.concatenate([np.arange(len(src_np)), np.arange(len(src_np))])
    vals_x = np.concatenate([dx_m / dist_m**2, -dx_m / dist_m**2])
    _A_x = csr_matrix((vals_x, (rows_x, cols_x)), shape=(N, len(src_np)))

    rows_y = np.concatenate([src_np, dst_np])
    cols_y = np.concatenate([np.arange(len(src_np)), np.arange(len(src_np))])
    vals_y = np.concatenate([dy_m / dist_m**2, -dy_m / dist_m**2])
    _A_y = csr_matrix((vals_y, (rows_y, cols_y)), shape=(N, len(src_np)))

    degree = np.bincount(src_np, minlength=N).astype(np.float32)
    degree = np.maximum(degree, 1.0)

    dp_edge_all = p_pa_all[:, dst_np] - p_pa_all[:, src_np]
    grad_px_all = (_A_x @ dp_edge_all.T).T / degree
    grad_py_all = (_A_y @ dp_edge_all.T).T / degree

    # ---- Manning's n from depth ----
    depth_t = torch.tensor(depth_np).unsqueeze(1)
    mannings_n = torch.where(depth_t > 20.0, torch.tensor(0.02),
                 torch.where(depth_t > 2.0,  torch.tensor(0.035),
                 torch.where(depth_t > 0.0,  torch.tensor(0.05),
                             torch.tensor(0.10))))
    manning_np = mannings_n.squeeze().numpy()

    # ---- Boundary tides ----
    print("   Synthesizing tidal boundary conditions...")
    bt_np = generate_boundary_tides(fort15_path, t_seconds, open_boundary_nodes)
    mean_bt = bt_np.mean(axis=1, keepdims=True)
    mean_bt_col = np.tile(mean_bt, (1, N))

    # ---- Assemble [T, N, 7] forcing tensor ----
    f_all = np.stack([
        np.tile(depth_np,   (time_steps, 1)),   # col 0: depth
        grad_px_all,                             # col 1: dP/dx
        grad_py_all,                             # col 2: dP/dy
        tau_x_all,                               # col 3: tau_x
        tau_y_all,                               # col 4: tau_y
        np.tile(manning_np, (time_steps, 1)),    # col 5: Manning n
        mean_bt_col,                             # col 6: mean boundary tide
    ], axis=2).astype(np.float32)               # [T, N, 7]

    forcing_sequence = torch.tensor(f_all, dtype=torch.float32)
    boundary_tides_t = torch.tensor(bt_np,  dtype=torch.float32)

    # ---- Edge weights (inverse haversine distance) ----
    dists = haversine(nodes[src_np, 0], nodes[src_np, 1],
                      nodes[dst_np, 0], nodes[dst_np, 1])
    dists = np.clip(dists, 1.0, None)
    edge_weight = torch.tensor(1.0 / dists, dtype=torch.float32)

    nodes_xy = nodes[:, :2]  # lon, lat
    return forcing_sequence, boundary_tides_t, nodes_xy, edge_weight, t_seconds
