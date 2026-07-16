import torch
import torch.nn as torch_nn
from torch_geometric.nn import GCNConv

class SpatioTemporalGNN(torch.nn.Module):
    def __init__(self, num_nodes, num_features, window_size, hidden_dim=64):
        """
        A Recurrent Graph Neural Network for Storm Surge Prediction.
        Uses GCN for spatial relationships and GRU for temporal dynamics.
        """
        super(SpatioTemporalGNN, self).__init__()
        
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        # Spatial Convolutions (processes each time step)
        self.gcn = GCNConv(num_features, hidden_dim)
        
        # Temporal Recurrence (processes the sequence of spatial embeddings)
        self.gru = torch_nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Final prediction layer
        self.fc1 = torch_nn.Linear(hidden_dim, 32)
        self.fc2 = torch_nn.Linear(32, 1)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, window_size, num_features]
        num_nodes = x.size(0)
        
        # 1. Spatial Processing
        # We apply GCN to each time step independently
        spatial_out = []
        for t in range(self.window_size):
            x_t = x[:, t, :] # [num_nodes, num_features]
            h_t = torch.relu(self.gcn(x_t, edge_index))
            spatial_out.append(h_t)
            
        # Stack back: [num_nodes, window_size, hidden_dim]
        spatial_seq = torch.stack(spatial_out, dim=1)
        
        # 2. Temporal Processing
        # Pass the sequence for each node through the GRU
        # GRU expects input of shape (batch, seq, feature), so num_nodes acts as batch
        gru_out, _ = self.gru(spatial_seq) 
        
        # Take the output of the last time step
        last_hidden_state = gru_out[:, -1, :] # [num_nodes, hidden_dim]
        
        # 3. Prediction
        out = torch.relu(self.fc1(last_hidden_state))
        out = self.fc2(out) # [num_nodes, 1]
        
        return out
