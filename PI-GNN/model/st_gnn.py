import torch
from torch_geometric.nn import GCNConv

class AutoregressiveSurrogate(torch.nn.Module):
    def __init__(self, num_nodes, num_forcing_features=5):
        super(AutoregressiveSurrogate, self).__init__()
        # Input features: [Zeta_previous, Depth, Pressure, Wind_U, Wind_V, Manning_n] = 6 features
        self.gcn1 = GCNConv(1 + num_forcing_features, 32)
        self.gcn2 = GCNConv(32, 16)
        
        # Output is the RATE OF CHANGE of Water Level (dZeta/dt)
        self.out = torch.nn.Linear(16, 1)

    def forward(self, forcing_sequence, edge_index):
        """
        forcing_sequence shape: [time_steps, num_nodes, num_forcing_features]
        This loop is a true numerical solver emulator.
        It starts with Water Level = 0, and simulates the entire storm sequentially.
        """
        time_steps = forcing_sequence.size(0)
        num_nodes = forcing_sequence.size(1)
        
        # Initial condition: Water level is 0 everywhere at t=0
        zeta_t = torch.zeros((num_nodes, 1), device=forcing_sequence.device)
        
        simulated_zetas = []
        
        for t in range(time_steps):
            forcing_t = forcing_sequence[t] # [num_nodes, 4]
            
            # Combine the model's OWN prediction from the last step with the physical forcing
            x_t = torch.cat([zeta_t, forcing_t], dim=1)
            
            # Spatial propagation across the mesh
            h = torch.relu(self.gcn1(x_t, edge_index))
            h = torch.relu(self.gcn2(h, edge_index))
            
            # Predict the RATE OF CHANGE (dz/dt)
            dzeta_dt = self.out(h) 
            
            # Explicit Euler Integration Step (Exactly like ADCIRC!)
            zeta_t = zeta_t + dzeta_dt
            
            simulated_zetas.append(zeta_t)
            
        # Return the fully simulated storm hydrograph
        return torch.stack(simulated_zetas, dim=0) # [time_steps, num_nodes, 1]
