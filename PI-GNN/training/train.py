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
     open_boundary_nodes, boundary_tides, nodes_xy, wet_mask) = create_full_simulation_dataset(
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
    wet_mask         = wet_mask.to(device)          # [T, N] bool
    nodes_xy_t       = torch.tensor(nodes_xy, dtype=torch.float32, device=device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    # criterion = torch.nn.MSELoss() (removed: using manual masked loss)

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
        # Start physics much earlier to prevent the model from collapsing into the trivial "predict zero" state
        if epoch <= 3:
            phys_weight = 0.1
            stage = "Burn-in (w=0.1)"
        else:
            phys_weight = 0.1 + 3.9 * (epoch - 3) / (epochs - 3)
            stage = f"Full Physics Ramp (w={phys_weight:.2f})"

        # Window expansion (same schedule as FlowFM reference)
        window = int(min(2000 + (epoch - 1) * (total_t / 17.0), total_t))
        # chunk_size=1: 30k-node graph × 64 hidden × 4 layers already uses ~14GB.
        # Multi-step unrolling would OOM. Drift is prevented by absolute-zeta
        # prediction and noise injection instead.
        chunk_size = 1
        valid_train = train_indices[(train_indices >= 1) & (train_indices < window)]

        steps_per_epoch = min(500, len(valid_train))
        t_sample = np.random.choice(valid_train, size=steps_per_epoch, replace=False)

        print(f"\n--- Epoch {epoch}/{epochs} | {stage} | window={window} steps ---")

        # ===== TRAINING PHASE =====
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0

        for step, t_idx in enumerate(t_sample):
            optimizer.zero_grad()

            # ----------------------------------------------------------------
            # 2-STEP SCHEDULED SAMPLING (Pushforward Trick)
            # ----------------------------------------------------------------
            # STEP 1: Start from noisy true state → predict zeta at t_idx
            # STEP 2: Feed model's OWN step-1 output (detached) → predict t+1
            # Loss at BOTH steps. This forces the model to experience its own
            # prediction errors during training, closing the train/inference gap
            # that caused monotonic drift. Memory cost = 2 × 1-step (no OOM).
            # ----------------------------------------------------------------
            t2 = t_idx + 1
            if t2 >= split_idx:
                t2 = t_idx  # Stay in bounds; only use step 1 at boundary

            # --- Step 1 ---
            true_prev_zeta = true_zetas[t_idx - 1]           # [N, 1]
            noise = torch.randn_like(true_prev_zeta) * 0.05
            prev_state_1 = torch.cat([
                true_prev_zeta + noise,
                torch.zeros_like(true_prev_zeta),
                torch.zeros_like(true_prev_zeta),
            ], dim=1)                                          # [N, 3]

            sim_1, u_1, v_1, _ = model(
                forcing_sequence[t_idx : t_idx + 1],
                edge_index, edge_weight, nodes_xy_t,
                open_boundary_nodes,
                boundary_tides[t_idx : t_idx + 1] if boundary_tides is not None else None,
                prev_state_1,
            )

            mask_1 = wet_mask[t_idx].unsqueeze(0).unsqueeze(-1).float()
            n_wet_1 = mask_1.sum().clamp(min=1.0)
            loss_1 = ((sim_1 - true_zetas[t_idx : t_idx + 1])**2 * mask_1).sum() / n_wet_1

            # --- Step 2: feed model's OWN output (detached — no extra backprop graph) ---
            prev_state_2 = torch.cat([
                sim_1.detach(),                               # Model's prediction, not ground truth
                u_1.detach(),
                v_1.detach(),
            ], dim=1)

            sim_2, u_2, v_2, _ = model(
                forcing_sequence[t2 : t2 + 1],
                edge_index, edge_weight, nodes_xy_t,
                open_boundary_nodes,
                boundary_tides[t2 : t2 + 1] if boundary_tides is not None else None,
                prev_state_2,
            )

            mask_2 = wet_mask[t2].unsqueeze(0).unsqueeze(-1).float()
            n_wet_2 = mask_2.sum().clamp(min=1.0)
            loss_2 = ((sim_2 - true_zetas[t2 : t2 + 1])**2 * mask_2).sum() / n_wet_2

            data_loss = (loss_1 + loss_2) * 0.5

            # Physics loss on step 1 window [t-1, t]
            if phys_weight > 0.0:
                zeta_phys = torch.cat([(true_prev_zeta + noise).unsqueeze(0), sim_1], dim=0)
                u_phys    = torch.cat([torch.zeros_like(true_prev_zeta).unsqueeze(0), u_1], dim=0)
                v_phys    = torch.cat([torch.zeros_like(true_prev_zeta).unsqueeze(0), v_1], dim=0)
                phys_loss = compute_swe_physics_loss(
                    zeta_phys, u_phys, v_phys,
                    forcing_sequence[t_idx - 1 : t_idx + 1], edge_index, nodes_xy_t, dt=900.0,
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
        
        # FIX: Initialize validation with the TRUE state right before the split!
        # This prevents the model from dropping into the peak of the storm with a flat zero-state ocean.
        true_val_start = true_zetas[split_idx - 1]
        current_state = torch.cat([
            true_val_start,
            torch.zeros_like(true_val_start), # U=0
            torch.zeros_like(true_val_start)  # V=0
        ], dim=1)

        with torch.no_grad():
            for start_t in range(split_idx, time_steps, 4):
                end_t = min(start_t + 4, time_steps)
                sim_val, _, _, current_state = model(
                    forcing_sequence[start_t:end_t],
                    edge_index,
                    edge_weight,
                    nodes_xy_t,
                    open_boundary_nodes,
                    boundary_tides[start_t:end_t] if boundary_tides is not None else None,
                    current_state,
                )
                # Apply same wet_mask as training so val loss is directly comparable
                vmask_t = wet_mask[start_t:end_t].unsqueeze(-1).float()  # [chunk, N, 1]
                n_wet_v  = vmask_t.sum().clamp(min=1.0)
                v_loss = ((sim_val - true_zetas[start_t:end_t])**2 * vmask_t).sum() / n_wet_v
                val_loss_total += v_loss.item()
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
