import sys
import os
import torch
from torch.optim import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.pignn_model import ParametricPIGNN
from dataset.data_extractor import create_full_simulation_dataset
from training.physics_loss import compute_physics_loss

def train_model():
    print("=== Parametric PI-GNN Surrogate Training ===")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_io'))
    f14 = os.path.join(base_dir, 'fort.14')
    f22 = os.path.join(base_dir, 'fort.22')
    f63 = os.path.join(base_dir, 'fort.63.nc')
    
    for f in [f14, f22, f63]:
        if not os.path.exists(f):
            print(f"CRITICAL ERROR: {f} not found!")
            return

    epochs = 30
    learning_rate = 0.0005 # Dropped 10x to ensure smooth, linear convergence
    
    print("1. Compiling Full Storm Dataset (Track + Mesh + Boundaries)...")
    forcing_sequence, edge_index, edge_weight, true_zetas, open_boundary_nodes, boundary_tides, nodes_xy = create_full_simulation_dataset(f14, f22, f63)
    
    num_nodes = forcing_sequence.size(1)
    time_steps = forcing_sequence.size(0)
    
    split_idx = int(time_steps * 0.8)
    print(f"Train/Test Split: {split_idx} train steps ({split_idx * 15 / 60:.1f} hours) | {time_steps - split_idx} test steps ({(time_steps - split_idx) * 15 / 60:.1f} hours)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Initializing Parametric PI-GNN on {device}...")
    
    num_features = forcing_sequence.size(2) # 4 forcing features
    
    model = ParametricPIGNN(num_nodes=num_nodes, num_forcing_features=num_features).to(device)
    
    # Move huge tensors to device
    forcing_sequence = forcing_sequence.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    true_zetas = true_zetas.to(device)
    boundary_tides = boundary_tides.to(device)
    nodes_xy = torch.tensor(nodes_xy, dtype=torch.float32).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = torch.nn.MSELoss()
    
    print("3. Starting True Simulation Loop...")
    for epoch in range(epochs):
        
        # === TRAINING PHASE ===
        model.train()
        total_train_loss = 0
        # The deep 128-dimensional Parametric GNN has ~5x the parameters and vastly more 
        # intermediate edge activations than the old simple GCN. 
        # We must drop the mini-batch chunk_size significantly to prevent CUDA OOM.
        chunk_size = 2
        num_train_chunks = 0
        
        for start_t in range(0, split_idx, chunk_size):
            optimizer.zero_grad()
            end_t = min(start_t + chunk_size, split_idx)
            
            sim_chunk, _, _, _ = model(
                forcing_sequence[start_t:end_t], 
                edge_index, 
                edge_weight,
                nodes_xy,
                open_boundary_nodes, 
                boundary_tides[start_t:end_t] if boundary_tides is not None else None
            )
            
            data_loss = criterion(sim_chunk, true_zetas[start_t:end_t])
            
            # === FLOWFM MULTI-STAGE PHYSICS SCHEDULE ===
            # Epochs 0-10: Data-only (burn-in period for stability)
            # Epochs 10-20: Ramp up physics weight from 0 to 4.0
            # Epochs 20+: Full physics optimization (weight 4.0)
            if epoch < 10:
                physics_weight = 0.0
            elif epoch < 20:
                physics_weight = 4.0 * ((epoch - 10) / 10.0)
            else:
                physics_weight = 4.0
                
            # Physics Loss: Enforces SWE continuity (mass conservation) and spatial smoothness
            if physics_weight > 0:
                phys_loss = compute_physics_loss(sim_chunk, None, None, None, edge_index)
                loss = data_loss + physics_weight * phys_loss
            else:
                phys_loss = torch.tensor(0.0)
                loss = data_loss
                
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_chunks += 1
            
        avg_train_loss = total_train_loss / num_train_chunks
        scheduler.step()
        
        # === VALIDATION PHASE (UNSEEN DATA) ===
        model.eval()
        total_val_loss = 0
        num_val_chunks = 0
        
        with torch.no_grad():
            for start_t in range(split_idx, time_steps, chunk_size):
                end_t = min(start_t + chunk_size, time_steps)
                
                sim_chunk, _, _, _ = model(
                    forcing_sequence[start_t:end_t], 
                    edge_index, 
                    edge_weight,
                    nodes_xy,
                    open_boundary_nodes, 
                    boundary_tides[start_t:end_t] if boundary_tides is not None else None
                )
                
                val_loss = criterion(sim_chunk, true_zetas[start_t:end_t])
                total_val_loss += val_loss.item()
                num_val_chunks += 1
                
        avg_val_loss = total_val_loss / (num_val_chunks + 1e-8)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6e}")
        
    print("Training Complete. Saving simulator...")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'pi_gnn_model.pth'))
    print("Simulator saved to PI-GNN/training/pi_gnn_model.pth")

if __name__ == "__main__":
    train_model()
