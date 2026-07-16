import torch
import numpy as np

def compute_data_loss(predictions, targets, mask=None):
    """
    Standard Data Loss: L_data = MSE(Zeta_pred, Zeta_true)
    """
    loss = torch.nn.functional.mse_loss(predictions, targets, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum()
    return loss.mean()

def compute_boundary_loss(predictions, boundary_targets, boundary_node_indices):
    """
    Boundary Condition Loss: L_BC = MSE(Zeta_pred_boundary, Tide_actual)
    Penalizes the model if it doesn't match the forced tidal elevation at the open ocean.
    """
    if len(boundary_node_indices) == 0:
        return torch.tensor(0.0, device=predictions.device)
        
    pred_boundary = predictions[:, boundary_node_indices, :]
    target_boundary = boundary_targets[:, boundary_node_indices, :]
    
    return torch.nn.functional.mse_loss(pred_boundary, target_boundary)

def compute_physics_loss(zeta, depth, wind_x, wind_y, edge_index, dt=600):
    """
    Physics Loss: L_physics = Mean( | d(zeta)/dt + div(H * V) |^2 )
    Since we don't have predicted velocities (U,V) directly from a simple model, 
    we approximate the shallow water momentum driven by wind stress.
    
    For a fully rigorous GWCE, the model MUST predict U and V as well. 
    Here, we penalize unnatural spikes in the water surface gradient to enforce continuity.
    """
    # 1. Temporal Derivative: d(zeta)/dt
    # zeta shape: [batch/time, nodes, 1]
    if zeta.size(0) < 2:
        return torch.tensor(0.0, device=zeta.device)
        
    dzeta_dt = (zeta[1:] - zeta[:-1]) / dt
    
    # 2. Spatial Smoothness & Continuity Penalty (Simplified Physics)
    # Water cannot stack up infinitely; gradients should be bounded by wind stress and depth.
    # In a full PINN, you compute the spatial derivative on the graph.
    
    src, dst = edge_index
    # Difference in water level between neighboring nodes
    zeta_diff = zeta[:, src, :] - zeta[:, dst, :]
    
    # Penalize extreme, non-physical gradients
    spatial_penalty = torch.mean(zeta_diff**2)
    
    # Combine (placeholder for full SWE PDEs)
    physics_loss = torch.mean(dzeta_dt**2) + 0.1 * spatial_penalty
    return physics_loss
