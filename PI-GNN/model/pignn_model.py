import torch
import torch.nn as nn

# Lag Configuration
N_FORCING_LAGS = 8   # number of lagged forcing snapshots (including current)


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
        msg_fwd = self.message_net(torch.cat([h[src], h[dst], edge_feat], dim=-1))
        msg_rev = self.message_net(torch.cat([h[dst], h[src], edge_feat], dim=-1))
        agg = torch.zeros(N, msg_fwd.size(-1), device=h.device, dtype=msg_fwd.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_fwd), msg_fwd)
        agg.scatter_add_(0, src.unsqueeze(1).expand_as(msg_rev), msg_rev)
        h_new = self.update_net(torch.cat([h.to(msg_fwd.dtype), agg], dim=-1))
        return self.norm(h.to(msg_fwd.dtype) + h_new)


class ParametricPIGNN(torch.nn.Module):
    """
    Non-autoregressive PI-GNN for storm surge prediction.
    """
    def __init__(self, num_nodes, num_forcing_features=6, hidden_dim=32, n_layers=3,
                 n_lags=N_FORCING_LAGS):
        super(ParametricPIGNN, self).__init__()
        self.n_lags = n_lags

        # Node input: XY(2) + depth(1) + lagged_forcing(n_lags * 5) + mannings(1)
        n_lagged_feat = n_lags * 5
        node_in_channels = 2 + 1 + n_lagged_feat + 1

        self.node_encoder = _mlp(node_in_channels, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(1, 16, 16)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, 16) for _ in range(n_layers)
        ])

        # Output: (zeta, u, v)
        self.decoder = _mlp(hidden_dim, hidden_dim, 3)

    def forward(self, forcing_sequence, edge_index, edge_weight, nodes_xy,
                open_boundary_nodes=None, boundary_tides=None, initial_states=None,
                t_start=0):
        """
        Args:
            forcing_sequence: [T, N, 6] — slice of the full forcing tensor
            edge_index, edge_weight, nodes_xy: graph topology
            open_boundary_nodes: Dirichlet BC node list
            boundary_tides:   [T, n_obn] water level at open boundary nodes
            initial_states:   ignored
            t_start:          global timestep index of forcing_sequence[0]
        """
        T = forcing_sequence.size(0)
        N = forcing_sequence.size(1)
        device = forcing_sequence.device

        src, dst = edge_index[0], edge_index[1]

        # Encode edge weights once (same for all timesteps)
        e = self.edge_encoder(edge_weight.unsqueeze(1))

        # Normalize XY coordinates once
        x_norm = (nodes_xy[:, 0:1] - nodes_xy[:, 0:1].mean()) / (nodes_xy[:, 0:1].std() + 1e-6)
        y_norm = (nodes_xy[:, 1:2] - nodes_xy[:, 1:2].mean()) / (nodes_xy[:, 1:2].std() + 1e-6)
        xy_feat = torch.cat([x_norm, y_norm], dim=1)  # [N, 2]

        # Static features: depth [N,1] and mannings [N,1] (constant over time)
        depth_feat    = forcing_sequence[0, :, 0:1]   # [N, 1]
        mannings_feat = forcing_sequence[0, :, 5:6]   # [N, 1]

        # Convert open boundary nodes to LongTensor once
        if open_boundary_nodes is not None and len(open_boundary_nodes) > 0:
            obn = torch.tensor(open_boundary_nodes, dtype=torch.long, device=device)
        else:
            obn = None

        # T_out = number of timesteps to OUTPUT (from boundary_tides or 1).
        # The full forcing_sequence is used ONLY for lag lookups via global_t.
        T_out = boundary_tides.size(0) if boundary_tides is not None else 1

        simulated_zetas, simulated_u, simulated_v = [], [], []

        for t in range(T_out):
            # Global timestep index into the FULL forcing_sequence
            global_t = t_start + t

            # Build lagged forcing features
            lag_feats = []
            for k in range(self.n_lags):
                lag_t = max(0, global_t - k)
                cols = [1, 2, 3, 4, 6]
                raw_lag = forcing_sequence[lag_t, :, cols].clone()
                
                # Normalize features
                raw_lag[:, 0] *= 100.0   # dP/dx
                raw_lag[:, 1] *= 100.0   # dP/dy
                raw_lag[:, 2] *= 0.2     # tau_x
                raw_lag[:, 3] *= 0.2     # tau_y
                
                lag_feats.append(raw_lag)  # [N, 5]
            lagged = torch.cat(lag_feats, dim=1)  # [N, n_lags*5]

            # Depth and mannings from current global timestep
            raw_depth_t = forcing_sequence[global_t, :, 0:1]
            depth_t     = raw_depth_t / 1000.0  # Scale depth (0-3000m) to roughly 0-3
            
            mannings_t  = forcing_sequence[global_t, :, 5:6] * 10.0 # Scale mannings (0.02 - 0.1) to roughly 0.2 - 1.0

            # Full node feature vector
            node_feat = torch.cat([xy_feat, depth_t, lagged, mannings_t], dim=1)

            # Message passing
            h = self.node_encoder(node_feat)
            for layer in self.gnn_layers:
                h = layer(h, src, dst, e)

            # Predict state
            preds  = self.decoder(h)
            zeta_t = torch.clamp(preds[:, 0:1], min=-10.0, max=15.0)
            u_t    = preds[:, 1:2]
            v_t    = preds[:, 2:3]

            # Wetting & Drying
            H_check = raw_depth_t + zeta_t
            wd_mask = (H_check > 0.05).float()
            u_t = u_t * wd_mask
            v_t = v_t * wd_mask

            # Soft BC (Network learns the boundary, no hard overwrite)
            # if obn is not None and boundary_tides is not None:
            #     zeta_t = zeta_t.clone()
            #     zeta_t[obn, 0] = boundary_tides[t]

            simulated_zetas.append(zeta_t)
            simulated_u.append(u_t)
            simulated_v.append(v_t)

        dummy_state = torch.zeros((N, 3), device=device)
        return (torch.stack(simulated_zetas, dim=0),
                torch.stack(simulated_u, dim=0),
                torch.stack(simulated_v, dim=0),
                dummy_state)
