"""
Example: Parametric PI-GNN Storm Surge Prediction
===================================================
Predict surge for Cyclone Amphan 2020 using ONLY the storm track + mesh.
No ADCIRC run needed after the model is trained.

For a NEW storm, just replace amphan_track.csv with your own track data.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'PI-GNN'))

from PI_GNN.inference import PIGNNSurgeModel

# ── 1. Load the trained surrogate ────────────────────────────────────────────
model = PIGNNSurgeModel(
    model_path  = 'PI-GNN/training/pi_gnn_model.pth',
    fort14_path = 'model_io/fort.14',
    fort15_path = 'model_io/fort.15',
)

# ── 2a. Predict from CSV track file ─────────────────────────────────────────
track = PIGNNSurgeModel.track_from_csv('model_io/amphan_track.csv')
zeta, t_sec, nodes_xy = model.predict(track, duration_hours=78, dt_minutes=15)

print(f"Output shape: {zeta.shape}")   # [T, N]
print(f"Peak surge: {zeta.max():.3f} m at node {zeta.max(axis=0).argmax()}")

# ── 2b. OR predict from inline arrays (e.g. for Cyclone Yaas 2021) ──────────
# track_yaas = PIGNNSurgeModel.track_from_dict(
#     time_s  = [0, 21600, 43200, 64800, 86400],
#     lat     = [14.0, 16.5, 18.2, 19.8, 21.2],
#     lon     = [88.5, 88.2, 88.0, 87.5, 87.0],
#     vmax_kt = [35, 60, 90, 110, 85],
#     pc_mb   = [1002, 988, 966, 944, 968],
# )
# zeta_yaas, t_sec_yaas, _ = model.predict(track_yaas, duration_hours=48)

# ── 3. Visualise ─────────────────────────────────────────────────────────────
fig_hydro = model.plot_hydrograph(
    zeta, t_sec,
    node_ids=[1000, 5000, 15000],
    title='Cyclone Amphan 2020 — PI-GNN Surge Prediction',
    dt_minutes=15)
fig_hydro.savefig('amphan_hydrographs.png', dpi=150, bbox_inches='tight')

fig_peak = model.plot_spatial_peak(
    zeta, title='Cyclone Amphan — Peak Surge (m)',
    fort14_path='model_io/fort.14')
fig_peak.savefig('amphan_peak_surge.png', dpi=150, bbox_inches='tight')

print("Saved: amphan_hydrographs.png, amphan_peak_surge.png")
