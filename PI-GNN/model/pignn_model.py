import torch
import torch.nn as nn

def _mlp(in_dim, hidden_dim, out_dim, n_hidden=1):
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
    def __init__(self, num_nodes, num_forcing_features=6, hidden_dim=32, n_layers=3):
        super(ParametricPIGNN, self).__init__()

        # node_in = XY(2) + physics(num_forcing_features) + prev_state(3)
        node_in_channels = 2 + num_forcing_features + 3

        # FIXED: Removed LayerNorm on input features.
        # LayerNorm normalizes across feature dimensions per-node. At inference time
        # with a single continuous sequence, the running stats are completely different
        # from the random 4-step training windows, causing catastrophic feature shift.
        # Instead we use a learnable per-feature scale+bias (equivalent to an affine
        # layer with no normalization dependency on batch statistics).
        self.feature_scale = nn.Parameter(torch.ones(node_in_channels))
        self.feature_bias  = nn.Parameter(torch.zeros(node_in_channels))
        self.node_encoder = _mlp(node_in_channels, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(1, 16, 16)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, 16) for _ in range(n_layers)
        ])

        # Output: ABSOLUTE state (Zeta, U, V) directly.
        # Delta-prediction was biased toward 0 and caused drift at inference time.
        # Absolute prediction forces the model to output physically plausible values.
        self.decoder = _mlp(hidden_dim, hidden_dim, 3)

    def forward(self, forcing_sequence, edge_index, edge_weight, nodes_xy, open_boundary_nodes=None, boundary_tides=None, initial_states=None):
        time_steps = forcing_sequence.size(0)
        num_nodes = forcing_sequence.size(1)
        device = forcing_sequence.device
        
        src, dst = edge_index[0], edge_index[1]

        # Encode edge weights (inverse distances) with the updated 16-dim encoder
        e_feat_raw = edge_weight.unsqueeze(1)
        e = self.edge_encoder(e_feat_raw)
        
        # Normalize XY coordinates
        x_norm = (nodes_xy[:, 0:1] - nodes_xy[:, 0:1].mean()) / (nodes_xy[:, 0:1].std() + 1e-6)
        y_norm = (nodes_xy[:, 1:2] - nodes_xy[:, 1:2].mean()) / (nodes_xy[:, 1:2].std() + 1e-6)
        xy_features = torch.cat([x_norm, y_norm], dim=1).to(device)
        
        if initial_states is None:
            prev_state = torch.zeros((num_nodes, 3), dtype=torch.float32, device=device)
        else:
            prev_state = initial_states

        # BUG 5 FIX: open_boundary_nodes is a numpy array.
        # GPU tensor indexing requires a LongTensor, not a numpy array.
        # Convert once here before the time loop.
        if open_boundary_nodes is not None and len(open_boundary_nodes) > 0:
            obn = torch.tensor(open_boundary_nodes, dtype=torch.long, device=device)
        else:
            obn = None
            
        simulated_zetas = []
        simulated_u = []
        simulated_v = []
        zeta_t, u_t, v_t = None, None, None
        
        for t in range(time_steps):
            forcing_t = forcing_sequence[t] # [num_nodes, 5]
            
            # === ADCIRC EXACT PHYSICS ===
            depth      = forcing_t[:, 0:1]
            mannings_n = forcing_t[:, 5:6]

            # FIX: Use DYNAMIC total water depth H = depth + zeta_prev.
            # Static depth ignores surge-induced depth change, freezing Cf at
            # a wrong pre-storm value and breaking the depth-friction feedback.
            zeta_prev  = prev_state[:, 0:1]
            H_total    = torch.clamp(depth + zeta_prev, min=0.1)
            Cf = (9.81 * mannings_n**2) / (H_total**(1.0/3.0))

            # physical_forcing: [Depth, dP/dx, dP/dy, tau_x, tau_y, Cf] = 6 cols
            physical_forcing = torch.cat([forcing_t[:, 0:5], Cf], dim=1)
            
            # Combine SPATIAL parameters (X, Y) with physical forcing (Depth, Pressure, Wind, Cf) and PREVIOUS STATE
            node_feat = torch.cat([xy_features, physical_forcing, prev_state], dim=1)
            
            # Apply learnable per-feature affine scaling (no batch-stat dependency)
            node_feat = node_feat * self.feature_scale + self.feature_bias
            
            # === DEEP MESSAGE PASSING ===
            h = self.node_encoder(node_feat)
            for layer in self.gnn_layers:
                h = layer(h, src, dst, e)
                
            # Predict ABSOLUTE state directly (not delta).
            # Delta prediction biases the model toward zero output and causes
            # monotonic drift at inference time on long autoregressive sequences.
            preds = self.decoder(h)
            zeta_t = preds[:, 0:1]   # absolute zeta
            u_t    = preds[:, 1:2]   # absolute u
            v_t    = preds[:, 2:3]   # absolute v

            # Clamp water level to physically plausible range
            zeta_t = torch.clamp(zeta_t, min=-10.0, max=15.0)

            # === DYNAMIC WETTING & DRYING (WD) ALGORITHM ===
            H_check = depth + zeta_t
            wd_mask = (H_check > 0.05).float()
            u_t = u_t * wd_mask
            v_t = v_t * wd_mask

            # === EXPLICIT TIDAL BOUNDARY FORCING (Dirichlet BC) ===
            if obn is not None and boundary_tides is not None:
                zeta_t = zeta_t.clone()
                zeta_t[obn, 0] = boundary_tides[t]
                
            prev_state = torch.cat([zeta_t, u_t, v_t], dim=1)
            simulated_zetas.append(zeta_t)
            simulated_u.append(u_t)
            simulated_v.append(v_t)
            
        return (torch.stack(simulated_zetas, dim=0), 
                torch.stack(simulated_u, dim=0), 
                torch.stack(simulated_v, dim=0), 
                prev_state)
