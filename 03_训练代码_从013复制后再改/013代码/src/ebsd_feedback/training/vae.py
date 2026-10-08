from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from ..config import ConfigNode
from ..constants import canonical_alloy_id
from ..data import EbsdDataset, load_fold_manifest
from ..losses import (
    VaeLossWeights,
    boundary_aware_vae_loss,
    simple_vae_loss,
    soft_boundary_map,
)
from ..models.vae import (
    BoundaryAwareVAE,
    PatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from ..monitor import TrainingMonitor
from ..paths import FOLD_MODEL_ROOT, LOG_ROOT, IMAGE_CHANNELS
from ..utils import (
    atomic_json_dump,
    atomic_torch_save,
    seed_everything,
    worker_seed,
)
from .common import (
    autocast_context,
    backward,
    configure_torch_performance,
    format_image_batch,
    infinite_batches,
    make_cosine_scheduler,
    make_adamw,
    make_grad_scaler,
    maybe_compile,
    optimizer_step,
    use_channels_last,
)


VAE_METRICS = [
    "loss_total",
    "loss_rgb",
    "loss_ssim",
    "loss_edge",
    "loss_boundary",
    "loss_haar",
    "loss_flatness",
    "loss_kl",
    "loss_gan_generator",
    "loss_gan_discriminator",
    "loss_r1",
    "kl_multiplier",
    "lr",
    "grad_norm",
]


def _model_from_config(config: ConfigNode) -> BoundaryAwareVAE:
    return BoundaryAwareVAE(
        base_channels=int(config.model.base_channels),
        channel_multipliers=tuple(config.model.channel_multipliers),
        latent_channels=int(config.model.latent_channels),
    )


def _save_reconstruction(
    model: BoundaryAwareVAE,
    images: torch.Tensor,
    output_path: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        reconstruction = model(images[:4], sample=False).reconstruction
        target_boundary = images[:4, 3:4].repeat(1, 3, 1, 1)
        predicted_boundary = reconstruction[:, 3:4].repeat(1, 3, 1, 1)
        grid = make_grid(
            torch.cat([images[:4, :3], reconstruction[:, :3], target_boundary, predicted_boundary]),
            nrow=min(4, images.shape[0]),
            normalize=True,
            value_range=(-1, 1),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, output_path)
    model.train()


@torch.no_grad()
def estimate_latent_scale(
    model: BoundaryAwareVAE,
    records: pd.DataFrame,
    train_ids: list[str],
    config: ConfigNode,
    device: torch.device,
) -> float:
    dataset = EbsdDataset(
        records,
        train_ids,
        samples_per_alloy=1,
        view="full",
        image_height=int(config.data.image_height),
        image_width=int(config.data.image_width),
        patch_size=int(config.data.patch_size),
        random_augment=False,
        cache_images=bool(getattr(config.data, "cache_images", False)),
        include_metadata=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    latents = []
    model.eval()
    for batch in loader:
        image = format_image_batch(
            batch["image"].to(device),
            bool(getattr(getattr(config, "performance", None), "channels_last", False)),
        )
        latent, _, _ = model.encode(image, sample=False)
        latents.append(latent.float().cpu().flatten())
    standard_deviation = torch.cat(latents).std().item()
    if standard_deviation < 1e-6:
        raise RuntimeError("VAE潜变量标准差异常，不能建立扩散尺度")
    return 1.0 / standard_deviation


def train_vae(config: ConfigNode, holdout_id: str) -> Path:
    holdout_id = canonical_alloy_id(holdout_id)
    seed_everything(int(config.seed), bool(config.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(config.environment.precision)
    configure_torch_performance(config, device)
    performance = getattr(config, "performance", None)
    channels_last = bool(getattr(performance, "channels_last", False))
    fold_root = FOLD_MODEL_ROOT / holdout_id
    manifest_path = fold_root / "00_折准备" / "严格留一折清单.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"请先准备严格留一折: {manifest_path}")
    records = load_fold_manifest(manifest_path)
    train_ids = records.loc[records["role"] == "train", "alloy_id"].tolist()
    if holdout_id in train_ids:
        raise RuntimeError("防泄漏检查失败：留出合金进入VAE训练列表")
    dataset = EbsdDataset(
        records,
        train_ids,
        samples_per_alloy=int(config.train.samples_per_alloy),
        view="patch",
        image_height=int(config.data.image_height),
        image_width=int(config.data.image_width),
        patch_size=int(config.data.patch_size),
        random_augment=True,
        cache_images=bool(getattr(config.data, "cache_images", False)),
        include_metadata=False,
    )
    worker_count = int(config.data.num_workers)
    loader_options = {
        "num_workers": worker_count,
        "pin_memory": bool(config.data.pin_memory),
        "persistent_workers": bool(config.data.persistent_workers) and worker_count > 0,
        "worker_init_fn": worker_seed,
    }
    if worker_count > 0:
        loader_options["prefetch_factor"] = int(getattr(config.data, "prefetch_factor", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    batches = infinite_batches(loader)
    model = use_channels_last(_model_from_config(config).to(device), channels_last)
    discriminator = (
        use_channels_last(PatchDiscriminator().to(device), channels_last)
        if bool(config.adversarial.enabled)
        else None
    )
    optimizer = make_adamw(
        model.parameters(),
        device,
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
        betas=(0.9, 0.95),
    )
    discriminator_optimizer = (
        make_adamw(
            discriminator.parameters(),
            device,
            lr=float(config.adversarial.discriminator_learning_rate),
            betas=(0.5, 0.9),
            weight_decay=0.0,
        )
        if discriminator
        else None
    )
    scheduler = make_cosine_scheduler(
        optimizer, int(config.train.warmup_steps), int(config.train.max_steps)
    )
    scaler = make_grad_scaler(device, precision)
    loss_weights = VaeLossWeights(
        rgb=float(config.loss.rgb),
        ssim=float(config.loss.ssim),
        edge=float(config.loss.edge),
        boundary=float(config.loss.boundary),
        haar=float(config.loss.haar),
        flatness=float(config.loss.flatness),
        kl=float(config.loss.kl),
    )
    loss_mode = str(getattr(config.loss, "mode", "full")).lower()
    if loss_mode not in {"full", "simple"}:
        raise ValueError(f"未知 VAE 损失模式: {loss_mode}")
    model_dir = fold_root / "01_边界感知VAE"
    run_dir = LOG_ROOT / holdout_id / "01_边界感知VAE"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config.to_dict(), run_dir / "实际配置.json")
    monitor = TrainingMonitor(
        run_dir,
        f"{holdout_id}-VAE",
        int(config.train.max_steps),
        VAE_METRICS,
        int(config.monitor.log_every),
        int(config.monitor.smoothing_window),
        bool(getattr(config.monitor, "tensorboard", True)),
    )
    start_step = 0
    resume_path = model_dir / "最近断点.pt"
    if not bool(config.train.resume) or not resume_path.exists():
        resume_path = None
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if discriminator and state.get("discriminator"):
            discriminator.load_state_dict(state["discriminator"])
            discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        start_step = int(state["step"])
        print(f"从断点继续: {resume_path}，起始步 {start_step}", flush=True)

    # Keep the original module for checkpoint/state-dict compatibility; compile only
    # the callable used in the training loop.
    model_forward = (
        maybe_compile(model, config, "VAE")
        if bool(getattr(performance, "compile_vae", False))
        else model
    )

    accumulation = int(config.train.gradient_accumulation)
    max_steps = int(config.train.max_steps)
    last_images: torch.Tensor | None = None
    last_reconstruction: torch.Tensor | None = None
    try:
        for step in range(start_step + 1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated = {name: 0.0 for name in VAE_METRICS}
            adversarial_active = discriminator is not None and step >= int(config.adversarial.start_step)
            if discriminator:
                discriminator.requires_grad_(False)
            for _ in range(accumulation):
                batch = next(batches)
                images = format_image_batch(
                    batch["image"].to(device, non_blocking=True), channels_last
                )
                kl_multiplier = min(1.0, step / max(int(config.loss.kl_warmup_steps), 1))
                with autocast_context(device, precision):
                    output = model_forward(images)
                    if loss_mode == "simple":
                        losses = simple_vae_loss(
                            output,
                            images,
                            edge_weight=float(config.loss.edge),
                            kl_weight=float(config.loss.kl),
                            kl_multiplier=kl_multiplier,
                        )
                    else:
                        losses = boundary_aware_vae_loss(
                            output, images, loss_weights, kl_multiplier
                        )
                    gan_generator = torch.zeros((), device=device)
                    if adversarial_active:
                        gan_generator = generator_hinge_loss(discriminator(output.reconstruction))
                        losses["loss_total"] = losses["loss_total"] + float(
                            config.adversarial.weight
                        ) * gan_generator
                    scaled_loss = losses["loss_total"] / accumulation
                if not bool(torch.isfinite(scaled_loss)):
                    raise RuntimeError("VAE损失出现NaN/Inf，停止写入最终模型")
                backward(scaled_loss, scaler)
                for name, value in losses.items():
                    accumulated[name] += float(value.detach()) / accumulation
                accumulated["loss_gan_generator"] += float(gan_generator.detach()) / accumulation
                accumulated["kl_multiplier"] = kl_multiplier
                last_images = images.detach()
                last_reconstruction = output.reconstruction.detach()
            grad_norm = optimizer_step(
                optimizer, model, float(config.train.gradient_clip), scaler
            )
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            discriminator_loss = torch.zeros((), device=device)
            r1_loss = torch.zeros((), device=device)
            if adversarial_active and last_images is not None and last_reconstruction is not None:
                discriminator.requires_grad_(True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                calculate_r1 = step % int(config.adversarial.r1_every) == 0
                real_images = last_images.detach().requires_grad_(calculate_r1)
                with autocast_context(device, precision):
                    real_logits = discriminator(real_images)
                    fake_logits = discriminator(last_reconstruction)
                    discriminator_loss = discriminator_hinge_loss(real_logits, fake_logits)
                if calculate_r1:
                    gradients = torch.autograd.grad(
                        real_logits.sum(), real_images, create_graph=True, retain_graph=True
                    )[0]
                    r1_loss = gradients.float().square().flatten(1).sum(1).mean()
                discriminator_total = discriminator_loss + float(config.adversarial.r1_weight) * r1_loss
                discriminator_total.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                discriminator_optimizer.step()
            accumulated.update(
                loss_gan_discriminator=float(discriminator_loss.detach()),
                loss_r1=float(r1_loss.detach()),
                lr=float(scheduler.get_last_lr()[0]),
                grad_norm=grad_norm,
            )
            monitor.log(step, accumulated)

            if step % int(config.monitor.sample_every) == 0 and last_images is not None:
                _save_reconstruction(
                    model,
                    last_images,
                    run_dir / "重建样例" / f"step_{step:08d}.png",
                )
            checkpoint_every = int(config.monitor.checkpoint_every)
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                state = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "discriminator": discriminator.state_dict() if discriminator else None,
                    "discriminator_optimizer": discriminator_optimizer.state_dict()
                    if discriminator_optimizer
                    else None,
                    "model_config": config.model.to_dict(),
                    "holdout_id": holdout_id,
                }
                atomic_torch_save(state, model_dir / "最近断点.pt")

        latent_scale = estimate_latent_scale(model, records, train_ids, config, device)
        final_state: dict[str, Any] = {
            "step": max_steps,
            "model": model.state_dict(),
            "model_config": config.model.to_dict(),
            "latent_scale": latent_scale,
            "holdout_id": holdout_id,
            "train_ids": train_ids,
        }
        final_path = model_dir / "VAE_最终模型.pt"
        atomic_torch_save(final_state, final_path)
        (model_dir / "最近断点.pt").unlink(missing_ok=True)
        (model_dir / "VAE结果说明.json").write_text(
            json.dumps(
                {
                    "holdout_id": holdout_id,
                    "latent_scale": latent_scale,
                    "downsample_factor": model.downsample_factor,
                    "final_checkpoint": str(final_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        monitor.close("completed")
        return final_path
    except KeyboardInterrupt:
        emergency = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": config.model.to_dict(),
            "holdout_id": holdout_id,
        }
        if int(config.monitor.checkpoint_every) > 0:
            atomic_torch_save(emergency, model_dir / "最近断点.pt")
        monitor.close("interrupted")
        raise
    except Exception:
        monitor.close("failed")
        raise
