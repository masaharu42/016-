from __future__ import annotations

import torch
from torch import nn

from ..orientation import ORIENTATION_COLUMNS


class LatentOrientationHead(nn.Module):
    """Small spatial feature head, returning normalized IPF bin fractions."""

    def __init__(self, latent_channels: int, hidden_dim: int = 32,
                 output_dim: int = len(ORIENTATION_COLUMNS)) -> None:
        super().__init__()
        if output_dim != len(ORIENTATION_COLUMNS):
            raise ValueError("Orientation schema/output dimension mismatch")
        self.latent_channels, self.hidden_dim, self.output_dim = latent_channels, hidden_dim, output_dim
        self.features = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.output = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.features(latent.float()).flatten(2)
        pooled = torch.cat([features.mean(2), features.std(2, unbiased=False)], 1)
        return self.output(pooled).float().softmax(-1)

    def config(self) -> dict[str, int]:
        return {"latent_channels": self.latent_channels, "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim}
