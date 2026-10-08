from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import BottleneckAttention, FiLMResBlock, group_count, sinusoidal_embedding


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5).square()
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999).float()


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = 1000) -> None:
        super().__init__()
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = F.pad(cumulative[:-1], (1, 0), value=1.0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumulative", cumulative)
        self.register_buffer("alpha_previous", previous)
        self.register_buffer("sqrt_alpha_cumulative", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_cumulative", (1.0 - cumulative).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        result = values.gather(0, timesteps)
        return result.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        clean_scale = self._extract(self.sqrt_alpha_cumulative, timesteps, clean.shape)
        noise_scale = self._extract(self.sqrt_one_minus_alpha_cumulative, timesteps, clean.shape)
        return clean_scale * clean + noise_scale * noise

    def predict_clean(
        self, noisy: torch.Tensor, predicted_noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        clean_scale = self._extract(self.sqrt_alpha_cumulative, timesteps, noisy.shape)
        noise_scale = self._extract(self.sqrt_one_minus_alpha_cumulative, timesteps, noisy.shape)
        return (noisy - noise_scale * predicted_noise) / clean_scale.clamp_min(1e-6)

    def min_snr_weight(self, timesteps: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        cumulative = self.alpha_cumulative.gather(0, timesteps)
        snr = cumulative / (1 - cumulative).clamp_min(1e-8)
        return snr.clamp(max=gamma) / snr.clamp_min(1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: "ConditionalLatentUNet",
        shape: tuple[int, ...],
        condition: torch.Tensor,
        steps: int,
        guidance_scale: float,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        latent = torch.randn(shape, device=device, generator=generator)
        sequence = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for position, timestep in enumerate(sequence):
            batch_t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            conditional_noise = model(latent, batch_t, condition)
            if guidance_scale != 1.0:
                unconditional_noise = model(
                    latent,
                    batch_t,
                    condition,
                    condition_drop_mask=torch.ones(shape[0], device=device, dtype=torch.bool),
                )
                predicted_noise = unconditional_noise + guidance_scale * (
                    conditional_noise - unconditional_noise
                )
            else:
                predicted_noise = conditional_noise
            clean = self.predict_clean(latent, predicted_noise, batch_t).clamp(-4.0, 4.0)
            if position == len(sequence) - 1:
                latent = clean
                continue
            previous_timestep = sequence[position + 1]
            alpha_previous = self.alpha_cumulative[previous_timestep]
            latent = alpha_previous.sqrt() * clean + (1 - alpha_previous).sqrt() * predicted_noise
        return latent

    def differentiable_ddim_sample(
        self,
        model: "ConditionalLatentUNet",
        shape: tuple[int, ...],
        condition: torch.Tensor,
        steps: int,
        device: torch.device,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """Short unrolled DDIM path used only for periodic mechanics feedback."""
        from torch.utils.checkpoint import checkpoint

        latent = torch.randn(shape, device=device)
        sequence = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for position, timestep in enumerate(sequence):
            batch_t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            if use_checkpoint:
                predicted_noise = checkpoint(
                    model,
                    latent,
                    batch_t,
                    condition,
                    use_reentrant=False,
                )
            else:
                predicted_noise = model(latent, batch_t, condition)
            clean = self.predict_clean(latent, predicted_noise, batch_t).clamp(-5.0, 5.0)
            if position == len(sequence) - 1:
                return clean
            previous_timestep = sequence[position + 1]
            alpha_previous = self.alpha_cumulative[previous_timestep]
            latent = alpha_previous.sqrt() * clean + (1 - alpha_previous).sqrt() * predicted_noise
        return latent


class ConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(embedding_dim))

    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.input_mean.shape or std.shape != self.input_std.shape:
            raise ValueError("条件统计维度与条件编码器不一致")
        self.input_mean.copy_(mean)
        self.input_std.copy_(std.clamp_min(1e-6))

    def forward(
        self, condition: torch.Tensor, condition_drop_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = (condition - self.input_mean) / self.input_std
        embedding = self.network(normalized)
        if condition_drop_mask is not None:
            embedding = torch.where(
                condition_drop_mask[:, None], self.null_embedding[None], embedding
            )
        return embedding


class ConditionalLatentUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        condition_dim: int = 30,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 2, 3, 4),
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        channels = [base_channels * value for value in channel_multipliers]
        embedding_dim = base_channels * 4
        self.time_dimension = base_channels
        self.time_network = nn.Sequential(
            nn.Linear(base_channels, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.condition_encoder = ConditionEncoder(condition_dim, embedding_dim)
        self.input_conv = nn.Conv2d(latent_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for index, output in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        FiLMResBlock(current, output, embedding_dim),
                        FiLMResBlock(output, output, embedding_dim),
                    ]
                )
            )
            current = output
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(current, current, 4, stride=2, padding=1))
        self.middle1 = FiLMResBlock(current, current, embedding_dim)
        self.middle_attention = BottleneckAttention(current, attention_heads)
        self.middle2 = FiLMResBlock(current, current, embedding_dim)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in reversed(range(len(channels))):
            output = channels[index]
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        FiLMResBlock(current + output, output, embedding_dim),
                        FiLMResBlock(output, output, embedding_dim),
                    ]
                )
            )
            current = output
            if index > 0:
                self.upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(current, current, 3, padding=1),
                    )
                )
        self.output = nn.Sequential(
            nn.GroupNorm(group_count(current), current),
            nn.SiLU(),
            nn.Conv2d(current, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        condition_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.time_network(sinusoidal_embedding(timesteps, self.time_dimension))
        embedding = embedding + self.condition_encoder(condition, condition_drop_mask)
        x = self.input_conv(noisy_latent)
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            x = blocks[0](x, embedding)
            x = blocks[1](x, embedding)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        x = self.middle2(self.middle_attention(self.middle1(x, embedding)), embedding)
        upsample_index = 0
        for reverse_index, blocks in enumerate(self.up_blocks):
            skip = skips[-1 - reverse_index]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = blocks[0](torch.cat([x, skip], dim=1), embedding)
            x = blocks[1](x, embedding)
            if reverse_index < len(self.upsamples):
                x = self.upsamples[upsample_index](x)
                upsample_index += 1
        return self.output(x)
