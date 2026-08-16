import sys
import os
import torch
import numpy as np
from torch.optim import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.pignn_model import ParametricPIGNN
from dataset.data_extractor import create_full_simulation_dataset
from training.physics_loss import compute_swe_physics_loss


def train_model():
    print("=== Parametric PI-GNN Surrogate Training (Hydrodynamically Honest) ===")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_io'))
    f14 = os.path.join(base_dir, 'fort.14')
    f22 = os.path.join(base_dir, 'fort.22')
    f63 = os.path.join(base_dir, 'fort.63.nc')

    for f in [f14, f22, f63]:
        if not os.path.exists(f):
            print(f"CRITICAL ERROR: {f} not found!")
            return

    # -----------------------------------------------------------------------
    # Multi-stage curriculum (mirrors the FlowFM project that solved this):
    #   Epochs 1-7  : pure data-driven pre-training (physics_weight = 0)
    #   Epochs 8-17 : gradual ramp  0.5 → 4.0
    #   Epochs 18+  : full physics constraints (physics_weight = 4.0)
    # -----------------------------------------------------------------------
    epochs        = 30
    learning_rate = 5e-4

    print("1. Compiling Full Storm Dataset (Track + Mesh + Boundaries)...")
    (forcing_sequence, edge_index, edge_weight, true_zetas,
     open_boundary_nodes, boundary_tides, nodes_xy) = create_full_simulation_dataset(
        f14, f22, f63)

    num_nodes  = forcing_sequence.size(1)
    time_steps = forcing_sequence.size(0)
    num_features = forcing_sequence.size(2)

    # 80 / 20 train-test split on time axis
    split_idx  = int(time_steps * 0.8)
    print(f"   Train: {split_idx} steps ({split_idx * 15 / 60:.1f} h) | "
          f"Test: {time_steps - split_idx} steps ({(time_steps - split_idx) * 15 / 60:.1f} h)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Initializing PI-GNN on {device}...")

    model = ParametricPIGNN(
        num_nodes=num_nodes, num_forcing_features=num_features).to(device)

    forcing_sequence = forcing_sequence.to(device)
    edge_index       = edge_index.to(device)
    edge_weight      = edge_weight.to(device)
    true_zetas       = true_zetas.to(device)
    boundary_tides   = boundary_tides.to(device)
    nodes_xy_t       = torch.tensor(nodes_xy, dtype=torch.float32, device=device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    criterion = torch.nn.MSELoss()

    # Training indices: skip t=0 so teacher-forcing always has a valid t-1
    # Randomize each epoch — prevents the model from learning to copy prev_state
    # instead of learning the wind→surge relationship.
    train_indices = np.arange(1, split_idx)

    # Window expansion: start with first 2 000 steps, grow to full dataset by epoch 17
    total_t = split_idx

    best_loss = float('inf')
    print("3. Starting Curriculum Training...")

    for epoch in range(1, epochs + 1):

        # ---- Physics weight schedule ----
        if epoch <= 7:
            phys_weight = 0.0
            stage = "Data-Only Pre-training"
        elif epoch <= 17:
            phys_weight = 0.5 + 3.5 * (epoch - 8) / (17 - 8)
            stage = f"Physics Ramp-Up (w={phys_weight:.2f})"
        else:
            phys_weight = 4.0
            stage = "Full Physics"

        # Window expansion (same schedule as FlowFM reference)
        window = int(min(2000 + (epoch - 1) * (total_t / 17.0), total_t))
        valid_train = train_indices[train_indices < window]

        # Sample 1 000 random time steps per epoch
        steps_per_epoch = min(1000, len(valid_train))
        t_sample = np.random.choice(valid_train, size=steps_per_epoch, replace=False)

        print(f"\n--- Epoch {epoch}/{epochs} | {stage} | window={window} steps ---")

        # ===== TRAINING PHASE =====
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0

        for step, t_idx in enumerate(t_sample):
            optimizer.zero_grad()

            # -----------------------------------------------------------------
            # Teacher forcing: provide the TRUE previous state (zeta_t-1, 0, 0)
            # as initial_states. We don't have true U/V from fort.63, so we
            # initialise velocity to zero — physically correct (start fresh each
            # random window, no sequential-copy shortcut).
            # -----------------------------------------------------------------
            true_prev_zeta = true_zetas[t_idx - 1]       # [N, 1]
            prev_state = torch.cat([
                true_prev_zeta,
                torch.zeros_like(true_prev_zeta),          # U₀ = 0
                torch.zeros_like(true_prev_zeta),          # V₀ = 0
            ], dim=1)                                      # [N, 3]

            # Run 1 time step (chunk_size = 1) with teacher-forced initial state
            sim_chunk, u_pred, v_pred, _, _ = model(
                forcing_sequence[t_idx : t_idx + 1],
                edge_index,
                edge_weight,
                nodes_xy_t,
                open_boundary_nodes,
                boundary_tides[t_idx : t_idx + 1] if boundary_tides is not None else None,
                prev_state,
            )

            data_loss = criterion(sim_chunk, true_zetas[t_idx : t_idx + 1])

            # Physics loss uses a 2-step window [t-1, t] so we need chunk_size >= 2
            # Build a 2-step chunk: [prev_true_zeta, sim_zeta_t]
            if phys_weight > 0.0:
                zeta_2step = torch.cat([
                    true_prev_zeta.unsqueeze(0),  # t-1: ground truth
                    sim_chunk,                     # t  : model prediction
                ], dim=0)                          # [2, N, 1]
                u_2step = torch.cat([
                    torch.zeros_like(true_prev_zeta).unsqueeze(0),
                    u_pred.unsqueeze(0),
                ], dim=0)
                v_2step = torch.cat([
                    torch.zeros_like(true_prev_zeta).unsqueeze(0),
                    v_pred.unsqueeze(0),
                ], dim=0)
                forcing_2step = forcing_sequence[t_idx - 1 : t_idx + 1]

                phys_loss = compute_swe_physics_loss(
                    zeta_2step, u_2step, v_2step,
                    forcing_2step, edge_index, nodes_xy_t,
                    dt=900.0,
                )
            else:
                phys_loss = torch.tensor(0.0, device=device)

            loss = data_loss + phys_weight * phys_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_data_loss += data_loss.item()
            epoch_phys_loss += phys_loss.item() if phys_weight > 0 else 0.0

            if (step + 1) % 200 == 0 or step == steps_per_epoch - 1:
                avg_d = epoch_data_loss / (step + 1)
                avg_p = epoch_phys_loss / (step + 1)
                print(f"  Step {step + 1}/{steps_per_epoch} | "
                      f"Data: {avg_d:.5f} | Phys: {avg_p:.5f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        avg_epoch_data = epoch_data_loss / steps_per_epoch

        # ===== VALIDATION PHASE (sequential, no teacher forcing) =====
        model.eval()
        val_loss_total = 0.0
        val_chunks     = 0
        current_state  = None

        with torch.no_grad():
            for start_t in range(split_idx, time_steps, 4):
                end_t = min(start_t + 4, time_steps)
                sim_val, _, _, _, current_state = model(
                    forcing_sequence[start_t:end_t],
                    edge_index,
                    edge_weight,
                    nodes_xy_t,
                    open_boundary_nodes,
                    boundary_tides[start_t:end_t] if boundary_tides is not None else None,
                    current_state,
                )
                val_loss_total += criterion(sim_val, true_zetas[start_t:end_t]).item()
                val_chunks     += 1

        avg_val = val_loss_total / max(val_chunks, 1)
        print(f"  >>> Epoch {epoch} Summary | Avg Data: {avg_epoch_data:.5f} | "
              f"Val Loss: {avg_val:.5f}")

        # Save best checkpoint only from epoch 20+ (physics-constrained phase)
        if epoch == 20:
            best_loss = float('inf')
        if epoch >= 20 and avg_epoch_data < best_loss:
            best_loss = avg_epoch_data
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), 'pi_gnn_model_best.pth'))
            print(f"  *** New best model saved (epoch {epoch}) ***")

    # Always save the final model for inference
    print("\nTraining Complete. Saving final simulator...")
    torch.save(model.state_dict(),
               os.path.join(os.path.dirname(__file__), 'pi_gnn_model.pth'))
    print("Saved to PI-GNN/training/pi_gnn_model.pth")


if __name__ == "__main__":
    train_model()
