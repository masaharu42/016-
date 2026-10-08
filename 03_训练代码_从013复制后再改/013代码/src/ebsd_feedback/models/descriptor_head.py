from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..constants import DESCRIPTOR_COLUMNS


@dataclass
class DescriptorTargetTransform:
    """Fold-local physical-to-standardized descriptor transform.

    The transform is fitted only on the training alloys of one strict fold. Positive
    descriptors use logarithms, while fraction descriptors use a logit;
    this keeps the head's unconstrained output physically meaningful after inversion.
    """

    columns: tuple[str, ...]
    modes: tuple[str, ...]
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor | Any) -> "DescriptorTargetTransform":
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] != len(DESCRIPTOR_COLUMNS):
            raise ValueError(
                f"描述符目标必须是[N,{len(DESCRIPTOR_COLUMNS)}]，实际为{tuple(tensor.shape)}"
            )
        transformed = cls._forward_physical(tensor)
        mean = transformed.mean(dim=0)
        scale = transformed.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(tuple(DESCRIPTOR_COLUMNS), cls._modes(), mean, scale)

    @staticmethod
    def _modes() -> tuple[str, ...]:
        return tuple("logit" if "fraction" in name else "log_shift1" if name == "area_weighted_aspect_ratio" else "log" for name in DESCRIPTOR_COLUMNS)

    @classmethod
    def _forward_physical(cls, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != len(DESCRIPTOR_COLUMNS):
            raise ValueError("描述符列数不正确")
        eps = torch.finfo(values.dtype).eps
        result = values.clone()
        for index, mode in enumerate(cls._modes()):
            if mode == "logit":
                result[..., index] = torch.logit(values[..., index].clamp(eps, 1.0 - eps))
            elif mode == "log_shift1":
                result[..., index] = torch.log((values[..., index] - 1).clamp_min(eps))
            else:
                result[..., index] = torch.log(values[..., index].clamp_min(eps))
        return result

    @staticmethod
    def _inverse_transformed(values: torch.Tensor) -> torch.Tensor:
        result = values.clone()
        for index, mode in enumerate(DescriptorTargetTransform._modes()):
            result[..., index] = (
                torch.sigmoid(values[..., index])
                if mode == "logit"
                else 1 + torch.exp(values[..., index]) if mode == "log_shift1" else torch.exp(values[..., index])
            )
        return result

    def transform(self, physical: torch.Tensor) -> torch.Tensor:
        transformed = self._forward_physical(physical.float())
        return (transformed - self.mean.to(transformed)) / self.scale.to(transformed).clamp_min(
            1e-6
        )

    def inverse(self, standardized: torch.Tensor) -> torch.Tensor:
        standardized = standardized.float()
        transformed = standardized * self.scale.to(standardized) + self.mean.to(standardized)
        return self._inverse_transformed(transformed)

    def state_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "modes": list(self.modes),
            "mean": self.mean.detach().cpu(),
            "scale": self.scale.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DescriptorTargetTransform":
        columns = tuple(state.get("columns", DESCRIPTOR_COLUMNS))
        if columns != tuple(DESCRIPTOR_COLUMNS):
            raise ValueError("描述符变换的列顺序与项目定义不一致")
        modes = tuple(state.get("modes", cls._modes()))
        if modes != cls._modes():
            raise ValueError("描述符变换不是013格式，不能混用旧权重")
        return cls(
            columns,
            modes,
            torch.as_tensor(state["mean"], dtype=torch.float32),
            torch.as_tensor(state["scale"], dtype=torch.float32).clamp_min(1e-6),
        )


class LatentDescriptorHead(nn.Module):
    """Predict descriptors from decoded IPF RGB, retaining spatial grain cues."""

    def __init__(
        self,
        latent_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = len(DESCRIPTOR_COLUMNS),
    ) -> None:
        super().__init__()
        if output_dim != len(DESCRIPTOR_COLUMNS):
            raise ValueError("描述符头输出维度必须与当前图像分支的描述符定义一致")
        hidden_dim = max(int(hidden_dim), 16)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = hidden_dim
        self.output_dim = int(output_dim)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.latent_channels, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(next(g for g in (32, 16, 8, 4, 2, 1) if hidden_dim % g == 0), hidden_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, clean_latent: torch.Tensor) -> torch.Tensor:
        if clean_latent.ndim != 4 or clean_latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"IPF图像必须是[N,{self.latent_channels},H,W]，实际为{tuple(clean_latent.shape)}"
            )
        pooled = self.pool(self.feature_extractor(clean_latent)).flatten(1)
        return self.network(pooled)

    def config(self) -> dict[str, int]:
        return {
            "latent_channels": self.latent_channels,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }
