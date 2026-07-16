import sys
import os
import torch
from torch.optim import Adam
from torch.utils.data.dataset import random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.st_gnn import SpatioTemporalGNN
from dataset.process_adcirc import create_sequence_dataset

def train_model():
    print("=== PI-GNN Actual Model Training ===")
    
    # Paths to the actual output data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_io'))
    f14 = os.path.join(base_dir, 'fort.14')
    f63 = os.path.join(base_dir, 'fort.63.nc')
    f73 = os.path.join(base_dir, 'fort.73.nc')
    f74 = os.path.join(base_dir, 'fort.74.nc') # Wind velocity

    # Check if files exist
    for f in [f14, f63]:
        if not os.path.exists(f):
            print(f"CRITICAL ERROR: {f} not found!")
            return

    # Hyperparameters
    epochs = 50
    learning_rate = 0.005
    window_size = 6  # Look back 6 time steps
    horizon = 1      # Predict 1 time step ahead
    
    print("1. Compiling Dataset from netCDF files (Full Storm Dataset)...")
    # This processes the actual massive datasets
    dataset = create_sequence_dataset(f14, f63, f73, f74, window_size=window_size, horizon=horizon)
    
    # Split into Train (80%) and Val (20%) sequences
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train sequences: {train_size}, Validation sequences: {val_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Initializing Model on {device}...")
    
    # Features = 5 (Depth, Zeta, Pressure, WindX, WindY)
    num_nodes = dataset[0].x.size(0)
    model = SpatioTemporalGNN(num_nodes=num_nodes, num_features=5, window_size=window_size).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    print("3. Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Iterate over sequences (each graph is one snapshot in time)
        # Note: Depending on RAM, you might need graph mini-batching (ClusterData). 
        # For standard memory, processing one full mesh sequence at a time is feasible.
        for batch_idx, data in enumerate(train_dataset):
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            
            # Compute Loss only on wet nodes (where target != 0 if dry, depending on formulation)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"   Seq {batch_idx}/{len(train_dataset)} - Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_dataset:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                loss = criterion(out, data.y)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_dataset):.4f} | Val Loss: {val_loss/len(val_dataset):.4f}")
        
    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'pi_gnn_model.pth'))
    print("Model saved to PI-GNN/training/pi_gnn_model.pth")

if __name__ == "__main__":
    train_model()
