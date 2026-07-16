import sys
import os
import torch
from torch.optim import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.st_gnn import AutoregressiveSurrogate
from dataset.process_adcirc import create_full_simulation_dataset

def train_model():
    print("=== PI-GNN Autoregressive Simulator Training ===")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_io'))
    f14 = os.path.join(base_dir, 'fort.14')
    f22 = os.path.join(base_dir, 'fort.22')
    f63 = os.path.join(base_dir, 'fort.63.nc')
    
    for f in [f14, f22, f63]:
        if not os.path.exists(f):
            print(f"CRITICAL ERROR: {f} not found!")
            return

    epochs = 50
    learning_rate = 0.005
    
    print("1. Compiling Full Storm Dataset (Track + Mesh + Boundaries)...")
    forcing_sequence, edge_index, true_zetas, open_boundary_nodes = create_full_simulation_dataset(f14, f22, f63)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Initializing Autoregressive Simulator on {device}...")
    
    num_nodes = forcing_sequence.size(1)
    num_features = forcing_sequence.size(2) # 4 forcing features
    
    model = AutoregressiveSurrogate(num_nodes=num_nodes, num_forcing_features=num_features).to(device)
    
    # Move huge tensors to device
    forcing_sequence = forcing_sequence.to(device)
    edge_index = edge_index.to(device)
    true_zetas = true_zetas.to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    print("3. Starting True Simulation Loop...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. The model starts at t=0 and simulates ALL time steps blindly!
        simulated_zetas = model(forcing_sequence, edge_index)
        
        # 2. Compare the fully simulated 3-day hydrograph against ADCIRC fort.63
        loss = criterion(simulated_zetas, true_zetas)
        
        # 3. Penalize the model and update weights
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Global Simulation Data Loss: {loss.item():.6f}")
        
    print("Training Complete. Saving simulator...")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'pi_gnn_model.pth'))
    print("Simulator saved to PI-GNN/training/pi_gnn_model.pth")

if __name__ == "__main__":
    train_model()
