from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .models.vae import VAEOutput


def to_unit_range(image: torch.Tensor) -> torch.Tensor:
    return image.add(1.0).mul(0.5).clamp(0.0, 1.0)


def gradient_components(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = image.shape[1]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    return (
        F.conv2d(image, kernel_x, padding=1, groups=channels),
        F.conv2d(image, kernel_y, padding=1, groups=channels),
    )


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    gradient_x, gradient_y = gradient_components(image)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-6)


def soft_boundary_map(image: torch.Tensor) -> torch.Tensor:
    unit = to_unit_range(image)
    if image.shape[1] == 4:
        return 1.0 - unit[:, 3:4]
    if image.shape[1] == 1:
        return 1.0 - unit
    gray = unit.mean(dim=1, keepdim=True)
    darkness = torch.sigmoid((0.18 - gray) * 24.0)
    edges = torch.tanh(gradient_magnitude(gray) * 2.5)
    return torch.maximum(darkness, edges)


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    prediction = to_unit_range(prediction)
    target = to_unit_range(target)
    padding = window // 2
    mean_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mean_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    variance_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mean_x.square()
    variance_y = F.avg_pool2d(target.square(), window, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    return 1.0 - (numerator / denominator.clamp_min(1e-8)).mean()


def haar_high_frequency(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_left = image[:, :, 0::2, 0::2]
    top_right = image[:, :, 0::2, 1::2]
    bottom_left = image[:, :, 1::2, 0::2]
    bottom_right = image[:, :, 1::2, 1::2]
    horizontal = (top_left - top_right + bottom_left - bottom_right) * 0.5
    vertical = (top_left + top_right - bottom_left - bottom_right) * 0.5
    diagonal = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return horizontal, vertical, diagonal


def haar_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_bands = haar_high_frequency(prediction)
    target_bands = haar_high_frequency(target)
    return sum(F.l1_loss(left, right) for left, right in zip(prediction_bands, target_bands)) / 3.0


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(gradient_magnitude(prediction), gradient_magnitude(target))


def boundary_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # BCE on probabilities is intentionally excluded from autocast: PyTorch rejects
    # BCELoss under CUDA autocast because its backward pass can overflow in fp16/bf16.
    with torch.amp.autocast(prediction.device.type, enabled=False):
        predicted_boundary = soft_boundary_map(prediction.float()).clamp(1e-5, 1 - 1e-5)
        target_boundary = soft_boundary_map(target.float()).detach().clamp(0.0, 1.0)
        positive = target_boundary.mean(dim=(1, 2, 3), keepdim=True)
        pos_weight = ((1.0 - positive) / positive.clamp_min(1e-4)).clamp(4.0, 20.0)
        weights = 1.0 + (pos_weight - 1.0) * target_boundary
        weighted_bce = (F.binary_cross_entropy(predicted_boundary, target_boundary, reduction="none") * weights).mean()
        intersection = (predicted_boundary * target_boundary).sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-5) / (
            predicted_boundary.sum(dim=(1, 2, 3)) + target_boundary.sum(dim=(1, 2, 3)) + 1e-5
        )
        return weighted_bce + dice.mean()


def intragranular_flatness_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    non_boundary = (1.0 - soft_boundary_map(target).detach()).clamp(0.0, 1.0)
    # Preserve true intragranular color variation rather than forcing flat grains.
    gradient = (gradient_magnitude(prediction[:, :3]) - gradient_magnitude(target[:, :3])).abs().mean(dim=1, keepdim=True)
    return (gradient * non_boundary).sum() / non_boundary.sum().clamp_min(1.0)


@dataclass
class VaeLossWeights:
    rgb: float = 1.0
    ssim: float = 0.25
    edge: float = 0.5
    boundary: float = 0.25
    haar: float = 0.25
    flatness: float = 0.05
    kl: float = 1e-6


def boundary_aware_vae_loss(
    output: VAEOutput,
    target: torch.Tensor,
    weights: VaeLossWeights,
    kl_multiplier: float = 1.0,
) -> dict[str, torch.Tensor]:
    # IPF RGB and the explicit fourth GB channel have different meanings.
    # Keep the RGB reconstruction dominant while giving the black boundary mask
    # its own pixel term; this avoids treating a boundary as an IPF color error.
    ipf_l1 = F.l1_loss(output.reconstruction[:, :3], target[:, :3])
    gb_l1 = F.l1_loss(output.reconstruction[:, 3:4], target[:, 3:4])
    losses = {
        "loss_rgb": 0.75 * ipf_l1 + 0.25 * gb_l1,
        "loss_ssim": ssim_loss(output.reconstruction[:, :3], target[:, :3]) if weights.ssim else ipf_l1.detach()*0,
        "loss_edge": edge_loss(output.reconstruction, target),
        "loss_boundary": boundary_loss(output.reconstruction[:, 3:4], target[:, 3:4]),
        "loss_haar": haar_loss(output.reconstruction, target),
        "loss_flatness": intragranular_flatness_loss(output.reconstruction, target),
        "loss_kl": -0.5
        * (1 + output.log_variance - output.mean.square() - output.log_variance.exp()).mean(),
    }
    losses["loss_total"] = (
        weights.rgb * losses["loss_rgb"]
        + weights.ssim * losses["loss_ssim"]
        + weights.edge * losses["loss_edge"]
        + weights.boundary * losses["loss_boundary"]
        + weights.haar * losses["loss_haar"]
        + weights.flatness * losses["loss_flatness"]
        + weights.kl * kl_multiplier * losses["loss_kl"]
    )
    return losses


def simple_vae_loss(
    output: VAEOutput,
    target: torch.Tensor,
    edge_weight: float = 0.2,
    kl_weight: float = 1e-6,
    kl_multiplier: float = 1.0,
) -> dict[str, torch.Tensor]:
    """The lightweight VAE objective used by the original project.

    Keeping this path separate avoids evaluating SSIM, boundary, Haar and
    flatness losses when they are not part of the selected objective.
    """
    loss_rgb = F.l1_loss(output.reconstruction, target)
    loss_edge = edge_loss(output.reconstruction, target)
    loss_kl = -0.5 * (
        1 + output.log_variance - output.mean.square() - output.log_variance.exp()
    ).mean()
    zero = loss_rgb.detach() * 0.0
    return {
        "loss_rgb": loss_rgb,
        "loss_ssim": zero,
        "loss_edge": loss_edge,
        "loss_boundary": zero,
        "loss_haar": zero,
        "loss_flatness": zero,
        "loss_kl": loss_kl,
        "loss_total": loss_rgb + edge_weight * loss_edge + kl_weight * kl_multiplier * loss_kl,
    }
