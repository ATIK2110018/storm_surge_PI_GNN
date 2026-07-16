import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from datetime import datetime, timedelta

# Default style (clean white background)
plt.style.use('default')

print("Loading data...")
ds_63 = nc.Dataset('fort.63.nc')
ds_74 = nc.Dataset('fort.74.nc')

x = ds_63.variables['x'][:]
y = ds_63.variables['y'][:]
elements = ds_63.variables['element'][:, :] - 1  # 0-based

time_var = ds_63.variables['time'][:]
zeta = ds_63.variables['zeta']
windx = ds_74.variables['windx']
windy = ds_74.variables['windy']

base_time = datetime(2020, 5, 18, 0, 0, 0)

print("Setting up triangulation...")
triang = mtri.Triangulation(x, y, elements)

# Adjusted frame to be vertically tighter to match the mesh aspect ratio
fig, ax = plt.subplots(figsize=(10, 8.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.subplots_adjust(bottom=0.15, top=0.90, left=0.05, right=0.95)

# Equal aspect ratio for accurate geometry, but completely turn off axes/map lines
ax.set_aspect('equal')
ax.axis('off')  # Removes all spines, ticks, grids, and axes
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())

print("Selecting quiver points...")
try:
    from scipy.spatial import cKDTree
    xi = np.linspace(x.min(), x.max(), 55)
    yi = np.linspace(y.min(), y.max(), 55)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_pts = np.vstack([Xi.ravel(), Yi.ravel()]).T
    tree = cKDTree(np.column_stack((x, y)))
    _, idx_quiver = tree.query(grid_pts)
    idx_quiver = np.unique(idx_quiver)
except ImportError:
    np.random.seed(0)
    idx_quiver = np.random.choice(len(x), size=3000, replace=False)

def get_face_values(z_nodes):
    if np.ma.is_masked(z_nodes):
        z_nodes = z_nodes.filled(np.nan)
    z_nodes = np.where((z_nodes < -100) | (z_nodes > 4.0), np.nan, z_nodes)
    z_faces = np.mean(z_nodes[elements], axis=1)
    return z_faces

print("Plotting initial frame...")
z0_nodes = zeta[0, :]
z0_faces = get_face_values(z0_nodes)
    
# Tighter vmin/vmax to make water level changes highly visible
tc = ax.tripcolor(triang, z0_faces, shading='flat', cmap='jet', vmin=-1.0, vmax=2.5, edgecolors='none')

fig.suptitle("ADCIRC+SWAN: Cyclone Amphan Storm Surge", fontsize=18, fontweight='bold', y=0.96)
title = ax.set_title("Time: ", fontsize=14, pad=5)

# Water Level Colorbar (Vertical, Right Side)
cbar_ax = fig.add_axes([0.90, 0.3, 0.02, 0.4])
cbar = plt.colorbar(tc, cax=cbar_ax)
cbar.set_label('Storm Surge Level (m)', fontsize=12)

u0 = windx[0, idx_quiver]
v0 = windy[0, idx_quiver]
z0_q = z0_nodes[idx_quiver] if not np.ma.is_masked(z0_nodes) else z0_nodes.filled(np.nan)[idx_quiver]

mask0 = (u0 < -1000) | (v0 < -1000) | (z0_q > 4.0) | (z0_q < -100)
u0 = np.where(mask0, np.nan, u0)
v0 = np.where(mask0, np.nan, v0)
mag0 = np.hypot(u0, v0)

# Wind vectors using 'magma'. Adjusted width for thicker arrows.
q = ax.quiver(x[idx_quiver], y[idx_quiver], u0, v0, mag0, cmap='magma', 
              scale=3500, alpha=0.9, width=0.002, zorder=3)

# Wind Colorbar (Horizontal, Bottom Center)
cbar_wind_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])
cbar_wind = plt.colorbar(q, cax=cbar_wind_ax, orientation='horizontal')
cbar_wind.set_label('Wind Velocity (m/s)', fontsize=12)

# Quiver Key - explicitly pass arrow length and adjust position
qk = ax.quiverkey(q, 0.85, 0.1, 20, '20 m/s', labelpos='E', coordinates='figure', fontproperties={'size': 12})

# Copyright Text
fig.text(0.5, 0.02, "© Atikur Rahman, Department of Water Resources Engineering, Chittagong University of Engineering and Technology", 
         ha='center', va='bottom', fontsize=10, color='gray', style='italic')

def update(frame):
    if frame % 10 == 0:
        print(f"Processing frame {frame}/{len(time_var)}")
        
    z_nodes = zeta[frame, :]
    if np.ma.is_masked(z_nodes):
        z_nodes = z_nodes.filled(np.nan)
        
    z_faces = get_face_values(z_nodes)
    tc.set_array(z_faces)
    
    u = windx[frame, idx_quiver]
    v = windy[frame, idx_quiver]
    z_q = z_nodes[idx_quiver]
    
    mask = (u < -1000) | (v < -1000) | (z_q > 4.0) | (z_q < -100)
    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)
    mag = np.hypot(u, v)
    
    q.set_UVC(u, v, mag)
    
    current_time = base_time + timedelta(seconds=float(time_var[frame]))
    title.set_text(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    return tc, q, title

print("Animating...")
frames_to_plot = range(0, len(time_var), 2)
anim = FuncAnimation(fig, update, frames=frames_to_plot, blit=False)

output_file = 'water_level_wind_anim.gif'
print(f"Saving to {output_file} at optimized resolution...")
anim.save(output_file, writer=PillowWriter(fps=8), dpi=150)
print("Done! Saved as", output_file)
