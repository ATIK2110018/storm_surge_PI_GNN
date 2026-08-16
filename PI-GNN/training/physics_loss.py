import torch


def compute_swe_physics_loss(zeta_chunk, u_chunk, v_chunk, forcing_chunk,
                              edge_index, nodes_xy, dt=900.0):
    """
    Real linearized Shallow Water Equation physics residuals on the ADCIRC mesh.

    Continuity:  dζ/dt + ∇·(H·U) = 0
    X-Momentum:  dU/dt + g·∂ζ/∂x = τ_x / (ρ_w·H)  -  Cf·U·|U| / H
    Y-Momentum:  dV/dt + g·∂ζ/∂y = τ_y / (ρ_w·H)  -  Cf·V·|U| / H

    Spatial gradients are computed via a finite-difference scatter over the
    ADCIRC mesh edges (unstructured FD approximation), using the real
    Haversine-derived edge direction cosines.

    Args:
        zeta_chunk:    [T, N, 1]  predicted water-surface elevations
        u_chunk:       [T, N, 1]  predicted depth-averaged x-velocity
        v_chunk:       [T, N, 1]  predicted depth-averaged y-velocity
        forcing_chunk: [T, N, 5]  columns = [depth, pressure, tau_x, tau_y, mannings_n]
        edge_index:    [2, E]     mesh edge connectivity (src, dst)
        nodes_xy:      [N, 3]     node table (lon, lat, depth) from fort.14
        dt:            float      integration timestep in seconds (default 900 s = 15 min)
    Returns:
        Scalar physics loss (sum of squared residuals).
    """
    T = zeta_chunk.size(0)
    if T < 2:
        return torch.tensor(0.0, device=zeta_chunk.device)

    device = zeta_chunk.device
    rho_water = 1025.0  # kg/m³
    g = 9.81            # m/s²

    src, dst = edge_index[0], edge_index[1]
    N = zeta_chunk.size(1)
    Tm1 = T - 1

    # ------------------------------------------------------------------
    # Edge geometry: direction cosines from node coordinates
    # Convert geographic degrees → approximate metres for gradient scaling
    # Bay of Bengal domain centred around ~21 °N
    # ------------------------------------------------------------------
    # BUG 2 FIX: Compute cos_lat from actual mesh nodes, not hardcoded 21°N.
    mean_lat_rad = (nodes_xy[:, 1].mean() * 3.14159265 / 180.0)
    cos_lat = torch.cos(mean_lat_rad)
    lon_m = 111320.0 * cos_lat   # m per degree longitude
    lat_m = torch.tensor(110540.0, device=device)

    x_m = nodes_xy[:, 0].to(device) * lon_m   # [N]
    y_m = nodes_xy[:, 1].to(device) * lat_m   # [N]

    dx = x_m[dst] - x_m[src]   # [E]
    dy = y_m[dst] - y_m[src]
    dist = torch.sqrt(dx**2 + dy**2 + 1e-6)
    cos_e = (dx / dist).detach()    # [E]
    sin_e = (dy / dist).detach()
    inv_e = (1.0 / dist).detach()

    # Expand to [Tm1, E, 1] for batch scatter
    cos_e = cos_e.unsqueeze(0).unsqueeze(-1).expand(Tm1, -1, 1)
    sin_e = sin_e.unsqueeze(0).unsqueeze(-1).expand(Tm1, -1, 1)
    inv_e = inv_e.unsqueeze(0).unsqueeze(-1).expand(Tm1, -1, 1)
    
    # BUG 9 FIX: scatter_add_ requires contiguous index tensors on GPU
    dst_b = dst.unsqueeze(0).unsqueeze(-1).expand(Tm1, -1, 1).contiguous()
    src_b = src.unsqueeze(0).unsqueeze(-1).expand(Tm1, -1, 1).contiguous()

    # Node degree for gradient normalisation
    degree = torch.zeros(N, device=device)
    degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    degree = degree.clamp(min=1.0).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

    # ------------------------------------------------------------------
    # Time derivatives (central finite difference over chunk)
    # ------------------------------------------------------------------
    dzeta_dt = (zeta_chunk[1:] - zeta_chunk[:-1]) / dt   # [Tm1, N, 1]
    du_dt    = (u_chunk[1:]    - u_chunk[:-1])    / dt
    dv_dt    = (v_chunk[1:]    - v_chunk[:-1])    / dt

    # Midpoint states for spatial terms (numerically stable)
    zeta = 0.5 * (zeta_chunk[1:] + zeta_chunk[:-1])   # [Tm1, N, 1]
    u    = 0.5 * (u_chunk[1:]    + u_chunk[:-1])
    v    = 0.5 * (v_chunk[1:]    + v_chunk[:-1])

    # Forcing at the later timestep
    depth      = forcing_chunk[1:, :, 0:1]   # [Tm1, N, 1]
    dPdx_atm   = forcing_chunk[1:, :, 1:2]   # ∂P/∂x  [Pa/m]  ← pre-computed gradient
    dPdy_atm   = forcing_chunk[1:, :, 2:3]   # ∂P/∂y  [Pa/m]
    tau_x_wind = forcing_chunk[1:, :, 3:4]   # wind stress x  [Pa]
    tau_y_wind = forcing_chunk[1:, :, 4:5]   # wind stress y  [Pa]
    mannings_n = forcing_chunk[1:, :, 5:6]

    H       = torch.clamp(depth + zeta, min=0.05)          # total water depth [m]
    Cf      = (g * mannings_n**2) / (H**(1.0/3.0) + 1e-8) # Manning friction coefficient
    vel_mag = torch.sqrt(u**2 + v**2 + 1e-8)

    # ------------------------------------------------------------------
    # Spatial gradient operator: ∂φ/∂x, ∂φ/∂y at each node
    # Unstructured FD approximation:
    #   ∂φ/∂x_i ≈ (1/deg_i) Σ_{j∈N(i)} (φ_j - φ_i) · cos_θ_ij / |r_ij|
    # ------------------------------------------------------------------
    def grad_xy(phi):
        """phi: [Tm1, N, 1] → grad_x, grad_y each [Tm1, N, 1]"""
        dphi = phi[:, dst, :] - phi[:, src, :]   # [Tm1, E, 1]
        gx = torch.zeros(Tm1, N, 1, device=device)
        gy = torch.zeros(Tm1, N, 1, device=device)
        # dst node gathers (φ_src - φ_dst) contribution
        gx.scatter_add_(1, dst_b, -dphi * cos_e * inv_e)
        gy.scatter_add_(1, dst_b, -dphi * sin_e * inv_e)
        # src node gathers (φ_dst - φ_src) contribution
        gx.scatter_add_(1, src_b,  dphi * cos_e * inv_e)
        gy.scatter_add_(1, src_b,  dphi * sin_e * inv_e)
        return gx / degree, gy / degree

    grad_zeta_x, grad_zeta_y = grad_xy(zeta)

    # ------------------------------------------------------------------
    # Continuity residual: dζ/dt + ∇·(H·U) = 0
    # ∇·(HU) computed as divergence of flux vector via scatter
    # ------------------------------------------------------------------
    Hu = H * u   # [Tm1, N, 1]
    Hv = H * v
    dHu = Hu[:, dst, :] - Hu[:, src, :]
    dHv = Hv[:, dst, :] - Hv[:, src, :]

    divHU = torch.zeros(Tm1, N, 1, device=device)
    divHU.scatter_add_(1, dst_b, -dHu * cos_e * inv_e - dHv * sin_e * inv_e)
    divHU.scatter_add_(1, src_b,  dHu * cos_e * inv_e + dHv * sin_e * inv_e)
    divHU = divHU / degree

    R_cont = dzeta_dt + divHU

    # ------------------------------------------------------------------
    # Momentum residuals (with wind stress source term)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # X-Momentum:  dU/dt + g·∂ζ/∂x + (1/ρ_w)·∂P/∂x = τ_x/(ρ_w·H) - Cf·U·|U|/H
    # The (1/ρ_w)·∂P/∂x term is the inverse barometer: pressure at storm eye
    # drops ~50 mb → forces ~0.5 m surge even before wind arrives.
    # ------------------------------------------------------------------
    wind_x = tau_x_wind / (rho_water * H + 1e-8)          # wind forcing [m/s²]
    wind_y = tau_y_wind / (rho_water * H + 1e-8)
    baro_x = dPdx_atm   / (rho_water + 1e-8)              # barometric forcing [m/s²]
    baro_y = dPdy_atm   / (rho_water + 1e-8)
    fric_x = Cf * u * vel_mag / (H + 1e-8)                # bottom friction [m/s²]
    fric_y = Cf * v * vel_mag / (H + 1e-8)

    R_mom_x = du_dt + g * grad_zeta_x + baro_x - wind_x + fric_x
    R_mom_y = dv_dt + g * grad_zeta_y + baro_y - wind_y + fric_y

    # Weight: continuity is the primary constraint; momentum scaled down
    return (torch.mean(R_cont**2) +
            0.1 * torch.mean(R_mom_x**2) +
            0.1 * torch.mean(R_mom_y**2))


# ---------------------------------------------------------------------------
# Keep the old stubs for backward-compatibility (they are no longer called)
# ---------------------------------------------------------------------------
def compute_data_loss(predictions, targets, mask=None):
    loss = torch.nn.functional.mse_loss(predictions, targets, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum()
    return loss.mean()


def compute_boundary_loss(predictions, boundary_targets, boundary_node_indices):
    if len(boundary_node_indices) == 0:
        return torch.tensor(0.0, device=predictions.device)
    pred_b   = predictions[:, boundary_node_indices, :]
    target_b = boundary_targets[:, boundary_node_indices, :]
    return torch.nn.functional.mse_loss(pred_b, target_b)
