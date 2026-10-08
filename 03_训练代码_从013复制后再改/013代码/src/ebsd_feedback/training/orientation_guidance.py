from __future__ import annotations

import torch


def js_divergence(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence (natural logs; range 0..ln(2))."""
    p, q = prediction.float().clamp_min(1e-8), target.float().clamp_min(1e-8)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    m = (p + q) * 0.5
    return 0.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1))


def orientation_consistency(latent, target, timesteps, drop_mask, head,
                            total_timesteps: int, low_noise_fraction: float):
    active = (timesteps < int(total_timesteps * low_noise_fraction)) & ~drop_mask
    if not active.any():
        return latent.sum() * 0.0, active
    # Freeze head weights, but keep the gradient path to the generated latent.
    prediction = head(latent[active])
    return js_divergence(prediction, target[active]).mean(), active
