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

    epochs = 2500
    learning_rate = 0.0005 # Dropped 10x to ensure smooth, linear convergence
    
    print("1. Compiling Full Storm Dataset (Track + Mesh + Boundaries)...")
    forcing_sequence, edge_index, edge_weight, true_zetas, open_boundary_nodes, boundary_tides = create_full_simulation_dataset(f14, f22, f63)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Initializing Autoregressive Simulator on {device}...")
    
    num_nodes = forcing_sequence.size(1)
    time_steps = forcing_sequence.size(0)
    num_features = forcing_sequence.size(2) # 4 forcing features
    
    model = AutoregressiveSurrogate(num_nodes=num_nodes, num_forcing_features=num_features).to(device)
    
    # Move huge tensors to device
    forcing_sequence = forcing_sequence.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    true_zetas = true_zetas.to(device)
    boundary_tides = boundary_tides.to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = torch.nn.MSELoss()
    
    print("3. Starting True Simulation Loop...")
    for epoch in range(epochs):
        
        zeta_t = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
        u_t = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
        v_t = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
        
        total_loss = 0
        # Dropped chunk_size back to 24 because the new Deep 128-dimensional GNN 
        # is massive and requires more VRAM per step than the old 16-dim network.
        chunk_size = 24
        num_chunks = 0
        
        for start_t in range(0, time_steps, chunk_size):
            optimizer.zero_grad()
            end_t = min(start_t + chunk_size, time_steps)
            
            # Standard float32 precision for Explicit Euler numerical stability!
            sim_chunk, zeta_t, u_t, v_t = model(
                forcing_sequence[start_t:end_t], 
                edge_index, 
                edge_weight,
                open_boundary_nodes, 
                boundary_tides[start_t:end_t] if boundary_tides is not None else None,
                initial_states=(zeta_t, u_t, v_t)
            )
            
            loss = criterion(sim_chunk, true_zetas[start_t:end_t])
            loss.backward()
            
            # Gradient Clipping: Prevents abrupt spikes in the loss curve!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # True TBPTT detachment
            zeta_t = zeta_t.detach()
            u_t = u_t.detach()
            v_t = v_t.detach()
            
            total_loss += loss.item()
            num_chunks += 1
            
        avg_loss = total_loss / num_chunks
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Avg TBPTT Loss (MSE): {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6e}")
        
    print("Training Complete. Saving simulator...")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'pi_gnn_model.pth'))
    print("Simulator saved to PI-GNN/training/pi_gnn_model.pth")

if __name__ == "__main__":
    train_model()
