import torch
import torch.nn as nn

def _mlp(in_dim, hidden_dim, out_dim, n_hidden=2):
    """Build a small MLP with SiLU activations."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

class GNNLayer(nn.Module):
    """
    Single message-passing layer using scatter_add_.
    Messages flow bidirectionally: c_L -> c_R and c_R -> c_L.
    """
    def __init__(self, hidden_dim, edge_feat_dim):
        super().__init__()
        self.message_net = _mlp(2 * hidden_dim + edge_feat_dim, hidden_dim, hidden_dim)
        self.update_net = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, src, dst, edge_feat):
        N = h.size(0)
        
        # Forward direction: src -> dst
        msg_fwd = self.message_net(torch.cat([h[src], h[dst], edge_feat], dim=-1))
        # Reverse direction: dst -> src
        msg_rev = self.message_net(torch.cat([h[dst], h[src], edge_feat], dim=-1))
        
        agg = torch.zeros(N, msg_fwd.size(-1), device=h.device, dtype=msg_fwd.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_fwd), msg_fwd)
        agg.scatter_add_(0, src.unsqueeze(1).expand_as(msg_rev), msg_rev)
        
        h_new = self.update_net(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)

class ParametricPIGNN(torch.nn.Module):
    def __init__(self, num_nodes, num_forcing_features=5, hidden_dim=128, n_layers=4):
        super(ParametricPIGNN, self).__init__()
        
        # Spatial inputs (X, Y) = 2
        # Physical forcing features (Depth, Pressure, TauX, TauY, Cf) = 5
        node_in_channels = 2 + num_forcing_features
        
        self.node_encoder = _mlp(node_in_channels, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(1, 16, 16) # Encode scalar edge_weight (inverse dist)
        
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, 16) for _ in range(n_layers)
        ])
        
        # Output is the ABSOLUTE state: Water Level, U_velocity, and V_velocity
        self.decoder = _mlp(hidden_dim, hidden_dim // 2, 3)

    def forward(self, forcing_sequence, edge_index, edge_weight, nodes_xy, open_boundary_nodes=None, boundary_tides=None, initial_states=None):
        time_steps = forcing_sequence.size(0)
        num_nodes = forcing_sequence.size(1)
        device = forcing_sequence.device
        
        src, dst = edge_index[0], edge_index[1]
        
        # Edge features: expand edge_weight to [N_edges, 1] and encode
        e_feat_raw = edge_weight.unsqueeze(1)
        e = self.edge_encoder(e_feat_raw)
        
        # Normalize XY coordinates
        x_norm = (nodes_xy[:, 0:1] - nodes_xy[:, 0:1].mean()) / (nodes_xy[:, 0:1].std() + 1e-6)
        y_norm = (nodes_xy[:, 1:2] - nodes_xy[:, 1:2].mean()) / (nodes_xy[:, 1:2].std() + 1e-6)
        xy_features = torch.cat([x_norm, y_norm], dim=1).to(device)
        
        simulated_zetas = []
        zeta_t, u_t, v_t = None, None, None
        
        for t in range(time_steps):
            forcing_t = forcing_sequence[t] # [num_nodes, 5]
            
            # === ADCIRC EXACT PHYSICS ===
            depth = forcing_t[:, 0:1]
            mannings_n = forcing_t[:, 4:5]
            
            # Nominal Depth for Friction (no previous zeta available)
            H_approx = torch.clamp(depth, min=0.1) 
            
            # Exact ADCIRC Bottom Friction Coefficient (Cf)
            Cf = (9.81 * mannings_n**2) / (H_approx**(1/3))
            
            # Replace raw Manning's n with the mathematically exact Cf
            physical_forcing = torch.cat([forcing_t[:, 0:4], Cf], dim=1)
            
            # Combine SPATIAL parameters (X, Y) with physical forcing (Depth, Pressure, Wind, Cf)
            node_feat = torch.cat([xy_features, physical_forcing], dim=1)
            
            # === DEEP MESSAGE PASSING ===
            h = self.node_encoder(node_feat)
            for layer in self.gnn_layers:
                h = layer(h, src, dst, e)
                
            # Predict the ABSOLUTE STATE directly (Parametric PINN)
            preds = self.decoder(h) 
            zeta_t = preds[:, 0:1]
            u_t = preds[:, 1:2]
            v_t = preds[:, 2:3]
            
            # === DYNAMIC WETTING & DRYING (WD) ALGORITHM ===
            H_check = depth + zeta_t
            wd_mask = (H_check > 0.05).float()
            u_t = u_t * wd_mask
            v_t = v_t * wd_mask
            
            # === EXPLICIT TIDAL BOUNDARY FORCING (Dirichlet BC) ===
            if open_boundary_nodes is not None and boundary_tides is not None:
                zeta_t[open_boundary_nodes, 0] = boundary_tides[t]
                
            simulated_zetas.append(zeta_t)
            
        return torch.stack(simulated_zetas, dim=0), zeta_t, u_t, v_t
