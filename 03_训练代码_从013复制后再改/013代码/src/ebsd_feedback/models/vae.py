from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .blocks import ResBlock, group_count
from ..paths import IMAGE_CHANNELS


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor
    latent: torch.Tensor


class BoundaryAwareVAE(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        channels = [base_channels * value for value in channel_multipliers]
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** (len(channels) - 1)
        self.encoder_in = nn.Conv2d(IMAGE_CHANNELS, channels[0], 3, padding=1)
        encoder_layers: list[nn.Module] = []
        current = channels[0]
        for index, output in enumerate(channels):
            encoder_layers.extend([ResBlock(current, output), ResBlock(output, output)])
            current = output
            if index < len(channels) - 1:
                encoder_layers.append(nn.Conv2d(current, current, 4, stride=2, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_mid = nn.Sequential(ResBlock(current, current), ResBlock(current, current))
        self.to_statistics = nn.Sequential(
            nn.GroupNorm(group_count(current), current),
            nn.SiLU(),
            nn.Conv2d(current, latent_channels * 2, 3, padding=1),
        )

        self.decoder_in = nn.Conv2d(latent_channels, channels[-1], 3, padding=1)
        self.decoder_mid = nn.Sequential(
            ResBlock(channels[-1], channels[-1]), ResBlock(channels[-1], channels[-1])
        )
        decoder_layers: list[nn.Module] = []
        current = channels[-1]
        for index in reversed(range(len(channels))):
            output = channels[index]
            decoder_layers.extend([ResBlock(current, output), ResBlock(output, output)])
            current = output
            if index > 0:
                decoder_layers.extend(
                    [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(current, current, 3, padding=1)]
                )
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_out = nn.Sequential(
            nn.GroupNorm(group_count(current), current),
            nn.SiLU(),
            nn.Conv2d(current, IMAGE_CHANNELS, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, image: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder_mid(self.encoder(self.encoder_in(image)))
        mean, log_variance = self.to_statistics(hidden).chunk(2, dim=1)
        log_variance = log_variance.clamp(-20.0, 8.0)
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        return latent, mean, log_variance

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder_out(self.decoder(self.decoder_mid(self.decoder_in(latent))))

    def forward(self, image: torch.Tensor, sample: bool = True) -> VAEOutput:
        latent, mean, log_variance = self.encode(image, sample=sample)
        return VAEOutput(self.decode(latent), mean, log_variance, latent)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64, layers: int = 4) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = IMAGE_CHANNELS
        for index in range(layers):
            output = min(base_channels * (2**index), 512)
            blocks.append(spectral_norm(nn.Conv2d(current, output, 4, stride=2, padding=1)))
            if index:
                blocks.append(nn.GroupNorm(group_count(output), output))
            blocks.append(nn.LeakyReLU(0.2, inplace=True))
            current = output
        blocks.append(spectral_norm(nn.Conv2d(current, 1, 3, padding=1)))
        self.network = nn.Sequential(*blocks)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
