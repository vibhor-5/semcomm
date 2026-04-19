"""
losses/flow_loss.py — Standalone flow-matching loss for use outside the model.

In practice the loss is computed inside FlowDecoder.compute_loss().
This module provides the building blocks so they can be imported separately
and used in training scripts that want explicit control over the loss terms.
"""

import torch
import torch.nn.functional as F


def flow_matching_loss(pred_v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
    """Simple MSE between predicted velocity and target velocity.

    Args:
        pred_v:   [B, C, H, W] predicted vector field
        target_v: [B, C, H, W] target velocity (x0 - x1 for linear path)

    Returns:
        Scalar loss.
    """
    return F.mse_loss(pred_v, target_v)


def linear_interpolation(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Interpolate between x1 (noise) and x0 (data) at time t.

    Args:
        x0: [B, C, H, W] clean data
        x1: [B, C, H, W] noise
        t:  [B] time values in [0, 1]

    Returns:
        x_t: [B, C, H, W] interpolated sample
    """
    t_b = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - t_b) * x1 + t_b * x0


def ot_matched_noise(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Re-order x1 (noise) to minimise transport cost to x0 using linear EMD.

    Falls back to identity assignment if POT is not installed.

    Args:
        x0: [B, D] or [B, C, H, W] data samples
        x1: [B, D] or [B, C, H, W] noise samples (will be re-ordered)

    Returns:
        x1_matched: same shape as x1 but rows re-ordered by OT plan
    """
    try:
        import ot
        import numpy as np

        B = x0.shape[0]
        x0_np = x0.view(B, -1).detach().cpu().numpy()
        x1_np = x1.view(B, -1).detach().cpu().numpy()
        M = ot.dist(x0_np, x1_np)
        a, b = np.ones(B) / B, np.ones(B) / B
        P = ot.emd(a, b, M)
        assignment = np.argmax(P, axis=1)
        return x1[assignment]
    except ImportError:
        return x1  # identity fallback
