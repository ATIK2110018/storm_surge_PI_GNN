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
    print("=== Parametric PI-GNN Surrogate Training (Non-Autoregressive) ===")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_io'))
    f14 = os.path.join(base_dir, 'fort.14')
    f22 = os.path.join(base_dir, 'fort.22')
    f63 = os.path.join(base_dir, 'fort.63.nc')

    for f in [f14, f22, f63]:
        if not os.path.exists(f):
            print(f"CRITICAL ERROR: {f} not found!")
            return

    # -----------------------------------------------------------------------
    # Non-autoregressive curriculum (mirrors FlowFM project that solved drift)
    # Each time step is predicted INDEPENDENTLY from lagged forcing history.
    # No hidden state is passed forward → zero autoregressive drift possible.
    # -----------------------------------------------------------------------
    epochs        = 30
    learning_rate = 5e-4

    print("1. Compiling Full Storm Dataset (Track + Mesh + Boundaries)...")
    (forcing_sequence, edge_index, edge_weight, true_zetas,
     open_boundary_nodes, boundary_tides, nodes_xy, wet_mask,
     true_uvels, true_vvels) = create_full_simulation_dataset(
        f14, f22, f63)

    num_nodes    = forcing_sequence.size(1)
    time_steps   = forcing_sequence.size(0)
    num_features = forcing_sequence.size(2)

    # 80 / 20 train-test split on time axis
    split_idx = int(time_steps * 0.8)
    print(f"   Train: {split_idx} steps ({split_idx * 15 / 60:.1f} h) | "
          f"Test: {time_steps - split_idx} steps ({(time_steps - split_idx) * 15 / 60:.1f} h)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Initializing Non-Autoregressive PI-GNN on {device}...")

    model = ParametricPIGNN(
        num_nodes=num_nodes, num_forcing_features=num_features).to(device)

    forcing_sequence = forcing_sequence.to(device)
    edge_index       = edge_index.to(device)
    edge_weight      = edge_weight.to(device)
    true_zetas       = true_zetas.to(device)
    boundary_tides   = boundary_tides.to(device)
    wet_mask         = wet_mask.to(device)
    nodes_xy_t       = torch.tensor(nodes_xy, dtype=torch.float32, device=device)
    
    if true_uvels is not None:
        true_uvels = true_uvels.to(device)
        true_vvels = true_vvels.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # Training indices: model needs N_FORCING_LAGS past steps as context
    from model.pignn_model import N_FORCING_LAGS
    train_indices = np.arange(N_FORCING_LAGS, split_idx)

    total_t = split_idx
    best_loss = float('inf')
    print("3. Starting Curriculum Training (NO teacher forcing, NO hidden state)...")

    for epoch in range(1, epochs + 1):

        # Physics weight schedule
        if epoch <= 7:
            phys_weight = 0.0
            stage = "Data-Only Pre-training"
        elif epoch <= 17:
            phys_weight = 0.5 + 3.5 * (epoch - 8) / (17 - 8)
            stage = f"Physics Ramp-Up (w={phys_weight:.2f})"
        else:
            phys_weight = 4.0
            stage = "Full Physics"

        # Window expansion: grow from 2000 steps to full dataset by epoch 17
        window      = int(min(2000 + (epoch - 1) * (total_t / 17.0), total_t))
        valid_train = train_indices[train_indices < window]

        steps_per_epoch = min(500, len(valid_train))
        t_sample = np.random.choice(valid_train, size=steps_per_epoch, replace=False)

        print(f"\n--- Epoch {epoch}/{epochs} | {stage} | window={window} steps ---")

        # ===== TRAINING PHASE =====
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0

        for step, t_idx in enumerate(t_sample):
            optimizer.zero_grad()

            # NON-AUTOREGRESSIVE FORWARD PASS
            # The model uses the FULL forcing_sequence internally to build
            # lagged features for timestep t_idx. No prev_state needed.
            sim_t, u_t, v_t, _ = model(
                forcing_sequence,          # full sequence for lag lookup
                edge_index, edge_weight, nodes_xy_t,
                open_boundary_nodes,
                boundary_tides[t_idx : t_idx + 1] if boundary_tides is not None else None,
                t_start=int(t_idx),
            )

            # Masked MSE loss
            mask_t = wet_mask[t_idx].unsqueeze(0).unsqueeze(-1).float()
            n_wet  = mask_t.sum().clamp(min=1.0)
            data_loss_zeta = ((sim_t - true_zetas[t_idx : t_idx + 1])**2 * mask_t).sum() / n_wet
            
            data_loss_uv = 0.0
            if true_uvels is not None:
                data_loss_u = ((u_t - true_uvels[t_idx : t_idx + 1])**2 * mask_t).sum() / n_wet
                data_loss_v = ((v_t - true_vvels[t_idx : t_idx + 1])**2 * mask_t).sum() / n_wet
                data_loss_uv = data_loss_u + data_loss_v
            
            # Velocity values are typically smaller than surge elevations; weight them equally for now
            data_loss = data_loss_zeta + 1.0 * data_loss_uv

            # Physics loss: use t-1 prediction (also non-autoregressive)
            if phys_weight > 0.0:
                with torch.no_grad():
                    sim_tm1, u_tm1, v_tm1, _ = model(
                        forcing_sequence,
                        edge_index, edge_weight, nodes_xy_t,
                        open_boundary_nodes,
                        boundary_tides[t_idx - 1 : t_idx] if boundary_tides is not None else None,
                        t_start=int(t_idx - 1),
                    )
                zeta_phys = torch.cat([sim_tm1.detach(), sim_t], dim=0)  # [2, N, 1]
                u_phys    = torch.cat([u_tm1.detach(),   u_t],  dim=0)
                v_phys    = torch.cat([v_tm1.detach(),   v_t],  dim=0)
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

        # ===== VALIDATION PHASE =====
        # With non-autoregressive model, validation is simply predicting each
        # test timestep independently — no state initialization needed.
        model.eval()
        val_loss_total = 0.0
        val_chunks = 0

        with torch.no_grad():
            for val_t in range(split_idx, time_steps):
                sim_val, _, _, _ = model(
                    forcing_sequence,
                    edge_index, edge_weight, nodes_xy_t,
                    open_boundary_nodes,
                    boundary_tides[val_t : val_t + 1] if boundary_tides is not None else None,
                    t_start=int(val_t),
                )
                vmask_t = wet_mask[val_t].unsqueeze(0).unsqueeze(-1).float()
                n_wet_v = vmask_t.sum().clamp(min=1.0)
                v_loss  = ((sim_val - true_zetas[val_t : val_t + 1])**2 * vmask_t).sum() / n_wet_v
                val_loss_total += v_loss.item()
                val_chunks     += 1

        avg_val = val_loss_total / max(val_chunks, 1)
        print(f"  >>> Epoch {epoch} Summary | Avg Data: {avg_epoch_data:.5f} | "
              f"Val Loss: {avg_val:.5f}")

        # Save best checkpoint from epoch 20+
        if epoch == 20:
            best_loss = float('inf')
        if epoch >= 20 and avg_epoch_data < best_loss:
            best_loss = avg_epoch_data
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), 'pi_gnn_model_best.pth'))
            print(f"  *** New best model saved (epoch {epoch}) ***")

    print("\nTraining Complete. Saving final simulator...")
    torch.save(model.state_dict(),
               os.path.join(os.path.dirname(__file__), 'pi_gnn_model.pth'))
    print("Saved to PI-GNN/training/pi_gnn_model.pth")


if __name__ == "__main__":
    train_model()
