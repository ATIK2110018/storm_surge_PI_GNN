import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from datetime import datetime, timedelta

# Use a sleek dark background for premium LinkedIn look
plt.style.use('dark_background')

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

fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0d1117')
fig.subplots_adjust(bottom=0.2, top=0.88, left=0.05, right=0.95)
ax.set_facecolor('#0d1117')

# Subtle grid and axes
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.3, color='white')
for spine in ax.spines.values():
    spine.set_color('#30363d')
ax.tick_params(colors='#8b949e', labelsize=10)
ax.set_xlabel('Longitude', color='#8b949e', fontsize=12)
ax.set_ylabel('Latitude', color='#8b949e', fontsize=12)

print("Selecting quiver points...")
try:
    from scipy.spatial import cKDTree
    xi = np.linspace(x.min(), x.max(), 40)
    yi = np.linspace(y.min(), y.max(), 40)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_pts = np.vstack([Xi.ravel(), Yi.ravel()]).T
    tree = cKDTree(np.column_stack((x, y)))
    _, idx_quiver = tree.query(grid_pts)
    idx_quiver = np.unique(idx_quiver)
except ImportError:
    np.random.seed(0)
    idx_quiver = np.random.choice(len(x), size=1200, replace=False)

def get_face_values(z_nodes):
    if np.ma.is_masked(z_nodes):
        z_nodes = z_nodes.filled(np.nan)
    z_nodes = np.where((z_nodes < -100) | (z_nodes > 4.0), np.nan, z_nodes)
    z_faces = np.mean(z_nodes[elements], axis=1)
    return z_faces

print("Plotting initial frame...")
z0_nodes = zeta[0, :]
z0_faces = get_face_values(z0_nodes)
    
# Use 'turbo' colormap: a modern, highly visible, and beautiful alternative to jet
tc = ax.tripcolor(triang, z0_faces, shading='flat', cmap='turbo', vmin=-0.5, vmax=3.5, edgecolors='none')

# Add subtle coastlines/boundaries if we can, but mesh edges are too dense.
# Just rely on the tripcolor to define the shape.

# Main Title (Professional and clean)
fig.suptitle("Cyclone Amphan: Storm Surge & Wind Dynamics", 
             fontsize=20, color='white', fontweight='bold', y=0.96)
subtitle = ax.set_title("ADCIRC + SWAN Coupled Model | Time: ", 
                        fontsize=14, color='#c9d1d9', pad=15)

# Water Level Colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
cbar = plt.colorbar(tc, cax=cbar_ax)
cbar.set_label('Storm Surge Level (m)', color='white', fontsize=12)
cbar.ax.yaxis.set_tick_params(color='white')
cbar.outline.set_edgecolor('#30363d')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

u0 = windx[0, idx_quiver]
v0 = windy[0, idx_quiver]
z0_q = z0_nodes[idx_quiver] if not np.ma.is_masked(z0_nodes) else z0_nodes.filled(np.nan)[idx_quiver]

mask0 = (u0 < -1000) | (v0 < -1000) | (z0_q > 4.0) | (z0_q < -100)
u0 = np.where(mask0, np.nan, u0)
v0 = np.where(mask0, np.nan, v0)
mag0 = np.hypot(u0, v0)

# Vectors using a bright contrasting map for dark themes
q = ax.quiver(x[idx_quiver], y[idx_quiver], u0, v0, mag0, cmap='spring', 
              scale=1800, alpha=0.9, width=0.0018, zorder=3)

# Wind Colorbar horizontally at the bottom
cbar_wind_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
cbar_wind = plt.colorbar(q, cax=cbar_wind_ax, orientation='horizontal')
cbar_wind.set_label('Wind Velocity (m/s)', color='white', fontsize=12)
cbar_wind.ax.xaxis.set_tick_params(color='white')
cbar_wind.outline.set_edgecolor('#30363d')
plt.setp(plt.getp(cbar_wind.ax.axes, 'xticklabels'), color='white')

# Quiver Key
qk = ax.quiverkey(q, 0.85, 0.08, 20, '20 m/s Wind', labelpos='E', 
                  coordinates='figure', color='white', labelcolor='white', fontproperties={'size': 12})

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
    subtitle.set_text(f"ADCIRC + SWAN Coupled Model | Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    return tc, q, subtitle

print("Animating...")
# Use every frame for a smoother LinkedIn video feel, but we'll still save as GIF. 
# We'll use 15 fps and every 2nd frame for balance of smoothness and file size.
frames_to_plot = range(0, len(time_var), 2)
anim = FuncAnimation(fig, update, frames=frames_to_plot, blit=False)

output_file = 'linkedin_ready_anim.gif'
print(f"Saving to {output_file} at high resolution...")
# 150 DPI is a good sweet spot for social media GIFs (size vs clarity)
anim.save(output_file, writer=PillowWriter(fps=15), dpi=150)
print("Done! Saved as", output_file)
