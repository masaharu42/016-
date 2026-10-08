from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..constants import DESCRIPTOR_COLUMNS
from ..models.descriptor_head import DescriptorTargetTransform, LatentDescriptorHead


def fit_fold_transform(records: Any, train_ids: list[str]) -> DescriptorTargetTransform:
    """Fit descriptor scaling from training rows only; holdout rows are never read."""
    train_rows = records[records["alloy_id"].isin(train_ids)]
    columns = [f"desc_true_{name}" for name in DESCRIPTOR_COLUMNS]
    if len(train_rows) != len(train_ids):
        raise ValueError("描述符头拟合时训练合金清单不完整")
    return DescriptorTargetTransform.fit(train_rows[columns].to_numpy(dtype="float32"))


def descriptor_consistency(
    clean_latent: torch.Tensor,
    target_physical: torch.Tensor,
    timesteps: torch.Tensor,
    condition_drop_mask: torch.Tensor,
    head: LatentDescriptorHead,
    transform: DescriptorTargetTransform,
    total_timesteps: int,
    low_noise_fraction: float,
    huber_beta: float,
    decoded_image: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a low-noise, condition-preserving descriptor consistency loss."""
    fraction = min(max(float(low_noise_fraction), 0.0), 1.0)
    threshold = int(total_timesteps * fraction)
    active = (timesteps < threshold) & ~condition_drop_mask if threshold > 0 else torch.zeros_like(
        timesteps, dtype=torch.bool
    )
    predictions = clean_latent.new_zeros((clean_latent.shape[0], len(DESCRIPTOR_COLUMNS)))
    if not bool(active.any()):
        zero = clean_latent.sum() * 0.0
        return zero, {"predictions": predictions.detach(), "active": active}
    active_predictions = head(decoded_image[active, :3])
    predictions = predictions.to(active_predictions.dtype).index_copy(0, active.nonzero().flatten(), active_predictions)
    target = transform.transform(target_physical)
    loss = F.smooth_l1_loss(
        predictions[active], target[active], beta=float(huber_beta), reduction="mean"
    )
    return loss, {"predictions": predictions, "active": active}


def descriptor_error_metrics(
    predictions: torch.Tensor,
    target_physical: torch.Tensor,
    active: torch.Tensor,
    transform: DescriptorTargetTransform,
) -> dict[str, float]:
    """Compute physical-unit MAE for logging without affecting gradients."""
    if not bool(active.any()):
        return {f"descriptor_mae_{name}": 0.0 for name in DESCRIPTOR_COLUMNS}
    with torch.no_grad():
        predicted_physical = transform.inverse(predictions.detach())
        errors = (predicted_physical[active] - target_physical.detach()[active]).abs().mean(0)
    return {
        f"descriptor_mae_{name}": float(errors[index].cpu())
        for index, name in enumerate(DESCRIPTOR_COLUMNS)
    }
