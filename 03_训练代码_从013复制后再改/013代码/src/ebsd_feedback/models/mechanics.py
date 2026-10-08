from __future__ import annotations

import torch
from torch import nn
from ..paths import IMAGE_CHANNELS


CURVE_PARAMETER_EPS = 1e-4


def encode_curve_targets(
    targets: torch.Tensor,
    endpoint_min_strain_pct: float = 10.0,
) -> torch.Tensor:
    """Encode physical curve targets as positive increments and bounded fractions."""
    if targets.shape[-1] != 9:
        raise ValueError(f"力学目标必须为9维，实际为: {targets.shape}")
    eps = CURVE_PARAMETER_EPS
    fixed_stresses = torch.cummax(targets[:, :5].clamp_min(eps), dim=1).values
    fixed_increments = torch.cat(
        [fixed_stresses[:, :1], fixed_stresses[:, 1:] - fixed_stresses[:, :-1]], dim=1
    ).clamp_min(eps)
    peak_stress = torch.maximum(targets[:, 5], fixed_stresses[:, -1] + eps)
    peak_increment = (peak_stress - fixed_stresses[:, -1]).clamp_min(eps)
    endpoint_strain = targets[:, 7].clamp_min(endpoint_min_strain_pct + eps)
    peak_fraction = (
        (targets[:, 6] - 0.5) / (endpoint_strain - 0.5).clamp_min(eps)
    ).clamp(eps, 1.0 - eps)
    endpoint_fraction = (targets[:, 8] / peak_stress.clamp_min(eps)).clamp(
        eps, 1.0 - eps
    )
    return torch.stack(
        [
            *torch.log(fixed_increments).unbind(dim=1),
            torch.log(peak_increment),
            torch.logit(peak_fraction),
            torch.log((endpoint_strain - endpoint_min_strain_pct).clamp_min(eps)),
            torch.logit(endpoint_fraction),
        ],
        dim=1,
    )


def decode_curve_parameters(
    parameters: torch.Tensor,
    endpoint_min_strain_pct: float = 10.0,
) -> torch.Tensor:
    """Decode parameters to a smooth, physically valid nine-target curve."""
    if parameters.shape[-1] != 9:
        raise ValueError(f"力学参数必须为9维，实际为: {parameters.shape}")
    fixed_increments = torch.exp(parameters[:, :5])
    fixed_stresses = torch.cumsum(fixed_increments, dim=1)
    peak_stress = fixed_stresses[:, -1] + torch.exp(parameters[:, 5])
    endpoint_strain = endpoint_min_strain_pct + torch.exp(parameters[:, 7])
    peak_strain = 0.5 + torch.sigmoid(parameters[:, 6]) * (endpoint_strain - 0.5)
    endpoint_stress = peak_stress * torch.sigmoid(parameters[:, 8])
    return torch.stack(
        [*fixed_stresses.unbind(dim=1), peak_stress, peak_strain, endpoint_strain, endpoint_stress],
        dim=1,
    )


def constrain_curve_targets(
    candidates: torch.Tensor,
    endpoint_min_strain_pct: float = 10.0,
) -> torch.Tensor:
    """Backward-compatible name for decoding unconstrained physical parameters."""
    return decode_curve_parameters(candidates, endpoint_min_strain_pct)


def differentiable_curve_from_targets(
    targets: torch.Tensor,
    strain_grid: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear dense curve used for differentiable full-curve supervision."""
    if strain_grid.ndim == 1:
        strain_grid = strain_grid.unsqueeze(0).expand(targets.shape[0], -1)
    fixed_strain = targets.new_tensor([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    fixed_strain = fixed_strain.unsqueeze(0).expand(targets.shape[0], -1)
    anchor_strain = torch.cat(
        [fixed_strain, targets[:, 6:7], targets[:, 7:8]], dim=1
    )
    zero = torch.zeros_like(targets[:, :1])
    anchor_stress = torch.cat(
        [zero, targets[:, :5], targets[:, 5:6], targets[:, 8:9]], dim=1
    )
    anchor_strain, order = torch.sort(anchor_strain, dim=1, stable=True)
    anchor_stress = torch.gather(anchor_stress, 1, order)
    query = strain_grid.contiguous()
    upper = torch.searchsorted(anchor_strain.contiguous(), query, right=True)
    upper = upper.clamp(1, anchor_strain.shape[1] - 1)
    lower = upper - 1
    x0 = torch.gather(anchor_strain, 1, lower)
    x1 = torch.gather(anchor_strain, 1, upper)
    y0 = torch.gather(anchor_stress, 1, lower)
    y1 = torch.gather(anchor_stress, 1, upper)
    fraction = (query - x0) / (x1 - x0).clamp_min(1e-4)
    interpolated = y0 + fraction.clamp(0.0, 1.0) * (y1 - y0)
    last_stress = anchor_stress[:, -1:].expand_as(interpolated)
    return torch.where(query >= anchor_strain[:, -1:], last_stress, interpolated)


class ConvEncoder(nn.Module):
    """Compact EBSD encoder that does not require downloading natural-image weights."""

    def __init__(self, base_channels: int = 64, feature_dim: int = 384) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, feature_dim]
        layers: list[nn.Module] = []
        current = IMAGE_CHANNELS
        for output in channels:
            layers.extend(
                [
                    nn.Conv2d(current, output, 4, stride=2, padding=1),
                    nn.GroupNorm(min(32, output), output),
                    nn.GELU(),
                    nn.Conv2d(output, output, 3, padding=1),
                    nn.GroupNorm(min(32, output), output),
                    nn.GELU(),
                ]
            )
            current = output
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = feature_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(image)).flatten(1)


class EbsdMechanicsSurrogate(nn.Module):
    """Predicts corrections, then maps them to physically valid curve targets."""

    def __init__(
        self,
        composition_dim: int = 9,
        descriptor_dim: int = 6,
        target_dim: int = 9,
        image_feature_dim: int = 128,
        image_base_channels: int = 32,
        dropout: float = 0.35,
        endpoint_min_strain_pct: float = 10.0,
    ) -> None:
        super().__init__()
        self.endpoint_min_strain_pct = float(endpoint_min_strain_pct)
        self.image_encoder = ConvEncoder(
            base_channels=image_base_channels, feature_dim=image_feature_dim
        )
        self.composition_encoder = nn.Sequential(
            nn.Linear(composition_dim + descriptor_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.image_only_head = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Dropout(dropout),
            nn.Linear(image_feature_dim, 96),
            nn.GELU(),
            nn.Linear(96, target_dim),
        )
        self.fused_head = nn.Sequential(
            nn.LayerNorm(image_feature_dim + 64),
            nn.Dropout(dropout),
            nn.Linear(image_feature_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, target_dim),
        )
        self.register_buffer("composition_mean", torch.zeros(composition_dim))
        self.register_buffer("composition_std", torch.ones(composition_dim))
        self.register_buffer("descriptor_mean", torch.zeros(descriptor_dim))
        self.register_buffer("descriptor_std", torch.ones(descriptor_dim))
        self.register_buffer("residual_mean", torch.zeros(target_dim))
        self.register_buffer("residual_std", torch.ones(target_dim))
        self.register_buffer("curve_scale", torch.ones(target_dim))

    def set_statistics(
        self,
        composition_mean: torch.Tensor,
        composition_std: torch.Tensor,
        descriptor_mean: torch.Tensor,
        descriptor_std: torch.Tensor,
        residual_mean: torch.Tensor,
        residual_std: torch.Tensor,
        curve_scale: torch.Tensor,
    ) -> None:
        values = {
            "composition_mean": composition_mean,
            "composition_std": composition_std,
            "descriptor_mean": descriptor_mean,
            "descriptor_std": descriptor_std,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "curve_scale": curve_scale,
        }
        for name, value in values.items():
            target = getattr(self, name)
            if target.shape != value.shape:
                raise ValueError(f"{name}统计量维度错误: {value.shape} != {target.shape}")
            target.copy_(value)

    def forward(
        self,
        image: torch.Tensor,
        composition: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_feature = self.image_encoder(image)
        composition = (composition - self.composition_mean) / self.composition_std.clamp_min(1e-6)
        descriptor = (descriptor - self.descriptor_mean) / self.descriptor_std.clamp_min(1e-6)
        tabular = self.composition_encoder(torch.cat([composition, descriptor], dim=1))
        fused_normalized = self.fused_head(torch.cat([image_feature, tabular], dim=1))
        image_normalized = self.image_only_head(image_feature)
        return {
            "residual_normalized": fused_normalized,
            "image_residual_normalized": image_normalized,
            "residual": fused_normalized * self.residual_std + self.residual_mean,
            "image_residual": image_normalized * self.residual_std + self.residual_mean,
            "image_feature": image_feature,
        }

    def predict_curve(
        self,
        image: torch.Tensor,
        composition: torch.Tensor,
        descriptor: torch.Tensor,
        baseline_curve: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self(image, composition, descriptor)
        baseline_parameters = encode_curve_targets(
            baseline_curve, self.endpoint_min_strain_pct
        )
        output["baseline_parameters"] = baseline_parameters
        output["curve_parameters"] = baseline_parameters + output["residual"]
        output["image_curve_parameters"] = baseline_parameters + output["image_residual"]
        output["curve"] = decode_curve_parameters(
            output["curve_parameters"], self.endpoint_min_strain_pct
        )
        output["image_curve"] = decode_curve_parameters(
            output["image_curve_parameters"], self.endpoint_min_strain_pct
        )
        return output
