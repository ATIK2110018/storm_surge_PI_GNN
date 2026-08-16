"""
PI-GNN Parametric Storm Surge Inference
========================================
Predict surge for ANY cyclone in the Bay of Bengal domain using only:
  - fort.14   : ADCIRC mesh (domain geometry, fixed)
  - fort.15   : tidal constituents (fixed for domain)
  - storm track : CSV or dict with (time_s, lat, lon, vmax_kt, pc_mb)
  - trained model weights : pi_gnn_model.pth

No ADCIRC run required.

Example usage:
    from PI-GNN.inference import PIGNNSurgeModel
    import pandas as pd

    model = PIGNNSurgeModel(
        model_path  = 'PI-GNN/training/pi_gnn_model.pth',
        fort14_path = 'model_io/fort.14',
        fort15_path = 'model_io/fort.15',
    )

    # Predict for a new storm (e.g. Yaas 2021)
    track = pd.DataFrame({
        'time_s'  : [0, 21600, 43200, 64800, 86400],   # 6-hourly
        'lat'     : [12.5, 14.0, 16.2, 18.5, 20.8],
        'lon'     : [87.5, 87.2, 87.0, 86.5, 85.9],
        'vmax_kt' : [35, 55, 75, 100, 120],
        'pc_mb'   : [1000, 990, 975, 955, 932],
    })
    zeta, t_sec, nodes_xy = model.predict(track, duration_hours=36)

    # zeta shape: [T, N]  — water level (m) at each timestep and mesh node
"""
import os, sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class PIGNNSurgeModel:
    """
    Parametric PI-GNN surrogate for storm surge prediction.

    The model predicts water levels η(t, x) across the ADCIRC mesh
    for any tropical cyclone described by its best-track parameters.
    """

    def __init__(self, model_path, fort14_path, fort15_path, device=None):
        """
        Parameters
        ----------
        model_path  : str  path to pi_gnn_model.pth
        fort14_path : str  path to fort.14 (ADCIRC mesh — fixed for domain)
        fort15_path : str  path to fort.15 (tidal constituents — fixed for domain)
        device      : str or None  ('cuda', 'cpu', or None for auto-detect)
        """
        from model.pignn_model import ParametricPIGNN
        from dataset.data_extractor import parse_fort14, create_graph_edges

        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"[PI-GNN] Initializing on {self.device}")

        # Load mesh (fixed for the domain)
        print(f"[PI-GNN] Parsing mesh: {fort14_path}")
        self.nodes, elements, self.open_boundary_nodes = parse_fort14(fort14_path)
        self.edge_index = create_graph_edges(elements).to(self.device)
        self.fort15_path = fort15_path
        self.num_nodes   = self.nodes.shape[0]
        self.nodes_xy_t  = torch.tensor(
            self.nodes[:, :2], dtype=torch.float32, device=self.device)

        # Load trained model
        print(f"[PI-GNN] Loading weights: {model_path}")
        # num_forcing_features=7 (depth + dPx + dPy + tau_x + tau_y + manning + mean_bt)
        self.model = ParametricPIGNN(
            num_nodes=self.num_nodes, num_forcing_features=7).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[PI-GNN] Ready — {self.num_nodes:,} nodes, "
              f"{self.edge_index.size(1):,} edges")

    # ------------------------------------------------------------------
    def predict(self, track_df, duration_hours=None, dt_minutes=15,
                return_velocities=False):
        """
        Predict storm surge water levels for the given cyclone track.

        Parameters
        ----------
        track_df : pandas.DataFrame with columns:
                      time_s   — seconds since any reference epoch (monotone ↑)
                      lat      — storm centre latitude (degrees N)
                      lon      — storm centre longitude (degrees E)
                      vmax_kt  — maximum sustained wind speed (knots)
                      pc_mb    — minimum central pressure (mb / hPa)
        duration_hours : float or None.
                      Simulation duration. If None, uses track end time.
        dt_minutes : int  — output timestep resolution (default 15 min)
        return_velocities : bool  — if True also return (u, v) depth-avg velocities

        Returns
        -------
        zeta     : ndarray [T, N]  — water surface elevation (m, MSL)
        t_sec    : ndarray [T]     — simulation time (seconds, same reference as track)
        nodes_xy : ndarray [N, 2]  — (lon, lat) of each mesh node
        (u, v)   : optional ndarrays [T, N] if return_velocities=True
        """
        import pandas as pd
        from dataset.inference_dataset import build_forcing_from_track

        required_cols = {'time_s', 'lat', 'lon', 'vmax_kt', 'pc_mb'}
        missing = required_cols - set(track_df.columns)
        if missing:
            raise ValueError(f"track_df is missing columns: {missing}")

        print(f"[PI-GNN] Building forcing for storm track "
              f"({len(track_df)} track points)...")
        forcing_seq, boundary_tides, nodes_xy, edge_weight, t_sec = \
            build_forcing_from_track(
                self.nodes, self.edge_index.cpu(),
                track_df, self.fort15_path,
                self.open_boundary_nodes,
                duration_hours=duration_hours,
                dt_minutes=dt_minutes)

        T = forcing_seq.size(0)
        forcing_seq    = forcing_seq.to(self.device)
        boundary_tides = boundary_tides.to(self.device)
        edge_weight    = edge_weight.to(self.device)

        print(f"[PI-GNN] Running inference ({T} timesteps = "
              f"{T * dt_minutes / 60:.1f} h)...")
        all_zeta, all_u, all_v = [], [], []
        with torch.no_grad():
            for t in range(T):
                zeta_t, u_t, v_t, _ = self.model(
                    forcing_seq,
                    self.edge_index,
                    edge_weight,
                    self.nodes_xy_t,
                    open_boundary_nodes=self.open_boundary_nodes,
                    boundary_tides=boundary_tides[t : t + 1],
                    t_start=t,
                )
                all_zeta.append(zeta_t.cpu())
                if return_velocities:
                    all_u.append(u_t.cpu())
                    all_v.append(v_t.cpu())

        zeta = torch.cat(all_zeta, dim=0).squeeze(-1).numpy()   # [T, N]
        print(f"[PI-GNN] Done! Peak surge: {zeta.max():.3f} m "
              f"at node {zeta.max(axis=0).argmax()}")

        if return_velocities:
            u = torch.cat(all_u, dim=0).squeeze(-1).numpy()
            v = torch.cat(all_v, dim=0).squeeze(-1).numpy()
            return zeta, t_sec, nodes_xy, u, v
        return zeta, t_sec, nodes_xy

    # ------------------------------------------------------------------
    @staticmethod
    def track_from_csv(csv_path):
        """
        Load a storm track from a CSV file.

        Expected columns (case-insensitive):
            time_s, lat, lon, vmax_kt, pc_mb

        Also accepts IBTrACS-style columns:
            ISO_TIME → converted to seconds since first entry
            LAT, LON, USA_WIND (knots), USA_PRES (mb)
        """
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]

        # IBTrACS compatibility
        if 'iso_time' in df.columns:
            df['time_s'] = (pd.to_datetime(df['iso_time']) -
                            pd.to_datetime(df['iso_time'].iloc[0])
                            ).dt.total_seconds()
            df = df.rename(columns={
                'lat': 'lat', 'lon': 'lon',
                'usa_wind': 'vmax_kt', 'usa_pres': 'pc_mb'})
        df = df.dropna(subset=['time_s', 'lat', 'lon', 'vmax_kt', 'pc_mb'])
        df = df.sort_values('time_s').reset_index(drop=True)
        return df[['time_s', 'lat', 'lon', 'vmax_kt', 'pc_mb']]

    # ------------------------------------------------------------------
    @staticmethod
    def track_from_dict(time_s, lat, lon, vmax_kt, pc_mb):
        """
        Build a track DataFrame directly from arrays/lists.

        Parameters
        ----------
        time_s  : seconds since reference (list or array, monotone ↑)
        lat     : storm latitude (deg N)
        lon     : storm longitude (deg E)
        vmax_kt : max wind speed (knots)
        pc_mb   : central pressure (mb)
        """
        import pandas as pd
        return pd.DataFrame({
            'time_s'  : time_s,
            'lat'     : lat,
            'lon'     : lon,
            'vmax_kt' : vmax_kt,
            'pc_mb'   : pc_mb,
        })

    # ------------------------------------------------------------------
    def plot_hydrograph(self, zeta, t_sec, node_ids,
                        truth=None, title='PI-GNN Surge Hydrograph',
                        dt_minutes=15):
        """Quick hydrograph plot for a list of node IDs."""
        import matplotlib.pyplot as plt
        t_hr = t_sec / 3600.0
        fig, axes = plt.subplots(len(node_ids), 1,
                                 figsize=(12, 4 * len(node_ids)), sharex=True)
        if len(node_ids) == 1:
            axes = [axes]
        for ax, nid in zip(axes, node_ids):
            ax.plot(t_hr, zeta[:, nid], 'r--', lw=2, label='PI-GNN')
            if truth is not None:
                ax.plot(t_hr, truth[:, nid], 'k-', lw=2, label='ADCIRC')
            ax.set_ylabel('η (m)')
            ax.set_title(f'Node {nid}')
            ax.legend(); ax.grid(True)
        axes[-1].set_xlabel('Time (hours)')
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def plot_spatial_peak(self, zeta, title='Peak Surge (m)', fort14_path=None):
        """Spatial map of peak water level across the whole domain."""
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        peak = zeta.max(axis=0)
        x, y = self.nodes[:, 0], self.nodes[:, 1]

        # Read elements for triangulation
        elements = []
        if fort14_path:
            with open(fort14_path) as f:
                f.readline()
                ne, nn = map(int, f.readline().split())
                for _ in range(nn): f.readline()
                for _ in range(ne):
                    p = f.readline().split()
                    elements.append([int(p[2])-1, int(p[3])-1, int(p[4])-1])
            triang = mtri.Triangulation(x, y, elements)
        else:
            triang = mtri.Triangulation(x, y)

        fig, ax = plt.subplots(figsize=(12, 10))
        cf = ax.tricontourf(triang, peak, levels=30, cmap='Reds')
        plt.colorbar(cf, ax=ax, label='Peak η (m)')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        return fig
