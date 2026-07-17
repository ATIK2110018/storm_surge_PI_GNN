import torch
from torch_geometric.nn import GCNConv

class AutoregressiveSurrogate(torch.nn.Module):
    def __init__(self, num_nodes, num_forcing_features=5):
        super(AutoregressiveSurrogate, self).__init__()
        # State variables: [Zeta_t, U_t, V_t] (3)
        # Forcing: [Depth, Pressure, Tau_x, Tau_y, Cf] (5)
        # Total input features = 8
        self.gcn1 = GCNConv(3 + num_forcing_features, 32)
        self.gcn2 = GCNConv(32, 16)
        
        # Output is the RATE OF CHANGE of Water Level, U_velocity, and V_velocity
        self.out = torch.nn.Linear(16, 3)

    def forward(self, forcing_sequence, edge_index, open_boundary_nodes=None, boundary_tides=None, initial_states=None):
        time_steps = forcing_sequence.size(0)
        num_nodes = forcing_sequence.size(1)
        
        device = forcing_sequence.device
        if initial_states is not None:
            zeta_t, u_t, v_t = initial_states
        else:
            zeta_t = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            u_t = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            v_t = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            
        simulated_zetas = []
        
        for t in range(time_steps):
            forcing_t = forcing_sequence[t] # [num_nodes, 5]
            
            # === ADCIRC EXACT PHYSICS ===
            depth = forcing_t[:, 0:1]
            mannings_n = forcing_t[:, 4:5]
            
            # Total Water Depth
            H = torch.clamp(depth + zeta_t, min=0.1) 
            
            # Exact ADCIRC Bottom Friction Coefficient (Cf)
            Cf = (9.81 * mannings_n**2) / (H**(1/3))
            
            # Replace raw Manning's n with the mathematically exact Cf
            physical_forcing = torch.cat([forcing_t[:, 0:4], Cf], dim=1)
            
            # Combine the model's STATE with physical forcing
            x_t = torch.cat([zeta_t, u_t, v_t, physical_forcing], dim=1)
            
            import torch.nn.functional as F
            
            # Spatial propagation across the mesh using SiLU (Smooth Non-Linearity)
            h = F.silu(self.gcn1(x_t, edge_index))
            h = F.silu(self.gcn2(h, edge_index))
            
            # Predict the RATE OF CHANGE (dzeta/dt, du/dt, dv/dt)
            rates = self.out(h) 
            dzeta = rates[:, 0:1]
            du = rates[:, 1:2]
            dv = rates[:, 2:3]
            
            # Explicit Euler Integration Step
            zeta_t = zeta_t + dzeta
            u_t = u_t + du
            v_t = v_t + dv
            
            # === DYNAMIC WETTING & DRYING (WD) ALGORITHM ===
            # If total water depth is very shallow (< 0.05m), kill momentum to prevent dry land flooding
            H_check = depth + zeta_t
            wd_mask = (H_check > 0.05).float()
            u_t = u_t * wd_mask
            v_t = v_t * wd_mask
            
            # === EXPLICIT TIDAL BOUNDARY FORCING (Dirichlet BC) ===
            # Force the open ocean nodes to exactly match astronomical tides (from fort.15 inputs)
            if open_boundary_nodes is not None and boundary_tides is not None:
                zeta_t[open_boundary_nodes, 0] = boundary_tides[t]
                
            simulated_zetas.append(zeta_t)
            
        return torch.stack(simulated_zetas, dim=0), zeta_t, u_t, v_t
