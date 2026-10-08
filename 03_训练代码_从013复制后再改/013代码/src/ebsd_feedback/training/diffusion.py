from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ..config import ConfigNode
from ..constants import CURVE_TARGET_COLUMNS, DESCRIPTOR_COLUMNS, canonical_alloy_id
from ..data import EbsdDataset, load_fold_manifest
from ..losses import boundary_loss, edge_loss, haar_loss, to_unit_range
from ..models.diffusion import ConditionalLatentUNet, DiffusionSchedule
from ..models.descriptor_head import DescriptorTargetTransform, LatentDescriptorHead
from .image_descriptor import load_frozen_proxy
from ..models.mechanics import EbsdMechanicsSurrogate, differentiable_curve_from_targets
from ..models.vae import BoundaryAwareVAE
from ..monitor import TrainingMonitor
from ..paths import FOLD_MODEL_ROOT, LOG_ROOT
from ..utils import (
    atomic_json_dump,
    atomic_torch_save,
    seed_everything,
    trainable_parameter_count,
    worker_seed,
)
from .common import (
    ExponentialMovingAverage,
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
from .descriptor_guidance import (
    descriptor_consistency,
    descriptor_error_metrics,
    fit_fold_transform,
)
from .orientation import label_path, load_frozen_orientation_head, orientation_stage_folder
from .orientation_guidance import orientation_consistency
from ..orientation import ORIENTATION_COLUMNS, load_orientation_labels


DIFFUSION_METRICS = [
    "loss_total",
    "loss_noise",
    "loss_image",
    "loss_edge",
    "loss_boundary",
    "loss_haar",
    "loss_mechanics",
    "loss_mechanics_unrolled",
    "loss_descriptor",
    "loss_descriptor_unrolled",
    "loss_overlay_alignment",
    "loss_orientation",
    "orientation_active_fraction",
    "descriptor_active_fraction",
    *[f"descriptor_mae_{name}" for name in DESCRIPTOR_COLUMNS],
    "mechanics_weight",
    "condition_gradient_norm",
    "lr",
    "grad_norm",
    "latent_abs_mean",
    "predicted_clean_abs_mean",
]


def build_condition(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            batch["composition"],
            batch["descriptor_condition"],
            batch["descriptor_std"],
            batch["curve_baseline"],
        ],
        dim=1,
    ).to(device, non_blocking=True)


def _load_vae(fold_root: Path, device: torch.device) -> tuple[BoundaryAwareVAE, float]:
    path = fold_root / "01_边界感知VAE" / "VAE_最终模型.pt"
    if not path.exists():
        raise FileNotFoundError(f"缺少VAE最终模型: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = BoundaryAwareVAE(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    return model, float(state["latent_scale"])


def _load_mechanics(fold_root: Path, device: torch.device) -> EbsdMechanicsSurrogate:
    path = fold_root / "02_力学代理" / "力学代理_最终模型.pt"
    if not path.exists():
        raise FileNotFoundError(f"缺少力学代理最终模型: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = EbsdMechanicsSurrogate(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    return model


def _set_feedback_train_scope(model: ConditionalLatentUNet, scope: str) -> None:
    if scope == "all":
        model.requires_grad_(True)
        return
    if scope != "condition_middle_up":
        raise ValueError(f"未知反馈训练范围: {scope}")
    model.requires_grad_(False)
    modules = [
        model.condition_encoder,
        model.middle1,
        model.middle_attention,
        model.middle2,
        model.up_blocks,
        model.upsamples,
        model.output,
    ]
    for module in modules:
        module.requires_grad_(True)


def _module_gradient_norm(module: torch.nn.Module) -> float:
    squared = torch.zeros((), device=next(module.parameters()).device)
    found = False
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
            found = True
    return float(squared.sqrt().cpu()) if found else 0.0


def _curve_feedback_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    config: ConfigNode,
    dense_grid: torch.Tensor | None = None,
    dense_target: torch.Tensor | None = None,
    dense_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized_error = (predicted - target) / scale.clamp_min(1e-6)
    element_loss = F.smooth_l1_loss(
        normalized_error, torch.zeros_like(normalized_error), reduction="none", beta=0.5
    )
    weights = torch.ones(len(CURVE_TARGET_COLUMNS), device=predicted.device)
    weights[CURVE_TARGET_COLUMNS.index("peak_stress_MPa")] = float(
        config.loss.mechanics_peak_multiplier
    )
    weights[CURVE_TARGET_COLUMNS.index("endpoint_strain_pct")] = float(
        config.loss.mechanics_endpoint_multiplier
    )
    weights[CURVE_TARGET_COLUMNS.index("endpoint_stress_MPa")] = float(
        config.loss.mechanics_endpoint_multiplier
    )
    weights[:3] *= float(
        getattr(config.loss, "mechanics_early_anchor_multiplier", 2.0)
    )
    anchor_loss = (element_loss * weights[None]).mean()
    if dense_grid is None or dense_target is None or dense_mask is None:
        return anchor_loss
    dense_prediction = differentiable_curve_from_targets(predicted, dense_grid)
    stress_scale = scale[[0, 1, 2, 3, 4, 5, 8]].mean().clamp_min(1e-6)
    dense_error = (dense_prediction - dense_target) / stress_scale
    dense_elements = F.smooth_l1_loss(
        dense_error,
        torch.zeros_like(dense_error),
        beta=0.5,
        reduction="none",
    )
    dense_loss = (dense_elements * dense_mask).sum() / dense_mask.sum().clamp_min(1)
    early_mask = dense_mask * (dense_grid <= 2.0)
    early_dense_loss = (dense_elements * early_mask).sum() / early_mask.sum().clamp_min(1)
    early_intervals = predicted.new_tensor([0.5, 0.5, 1.0])
    predicted_early = torch.cat(
        [predicted[:, :1], predicted[:, 1:3] - predicted[:, :2]], dim=1
    ) / early_intervals
    target_early = torch.cat(
        [target[:, :1], target[:, 1:3] - target[:, :2]], dim=1
    ) / early_intervals
    early_slope_loss = F.smooth_l1_loss(
        (predicted_early - target_early) / stress_scale,
        torch.zeros_like(target_early),
        beta=0.5,
    )
    return (
        anchor_loss
        + float(
        getattr(config.loss, "mechanics_dense_curve_multiplier", 0.5)
        ) * dense_loss
        + float(getattr(config.loss, "mechanics_early_dense_multiplier", 1.0))
        * early_dense_loss
        + float(getattr(config.loss, "mechanics_early_slope_multiplier", 1.0))
        * early_slope_loss
    )


@torch.no_grad()
def _save_sample(
    model: ConditionalLatentUNet,
    schedule: DiffusionSchedule,
    vae: BoundaryAwareVAE,
    latent_scale: float,
    condition: torch.Tensor,
    image_height: int,
    image_width: int,
    sampling_steps: int,
    guidance_scale: float,
    output_path: Path,
) -> None:
    model.eval()
    factor = vae.downsample_factor
    latent = schedule.ddim_sample(
        model,
        (1, vae.latent_channels, image_height // factor, image_width // factor),
        condition[:1],
        sampling_steps,
        guidance_scale,
        condition.device,
    )
    image = vae.decode(latent / latent_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image[:, :3], output_path, normalize=True, value_range=(-1, 1))
    save_image(image[:, 3:4], output_path.with_name(output_path.stem + "_GB.png"), normalize=True, value_range=(-1, 1))
    model.train()


def train_diffusion(config: ConfigNode, holdout_id: str) -> Path:
    holdout_id = canonical_alloy_id(holdout_id)
    is_feedback = str(config.stage) == "diffusion_feedback"
    seed_everything(int(config.seed), bool(config.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(config.environment.precision)
    configure_torch_performance(config, device)
    performance = getattr(config, "performance", None)
    channels_last = bool(getattr(performance, "channels_last", False))
    fold_root = FOLD_MODEL_ROOT / holdout_id
    records = load_fold_manifest(fold_root / "00_折准备" / "严格留一折清单.csv")
    train_ids = records.loc[records["role"] == "train", "alloy_id"].tolist()
    if holdout_id in train_ids:
        raise RuntimeError("防泄漏检查失败：留出合金进入扩散训练")
    orientation_enabled = bool(getattr(getattr(config, "orientation", None), "enabled", False))
    orientation_weight = float(getattr(config.loss, "orientation", 0.0))
    if orientation_enabled != (orientation_weight > 0):
        raise ValueError("orientation.enabled and positive loss.orientation must agree")
    orientation_head, orientation_state = None, None
    if orientation_enabled:
        labels = load_orientation_labels(label_path(config), train_ids)
        labels = labels[["alloy_id", *ORIENTATION_COLUMNS]].rename(
            columns={name: f"ori_{name}" for name in ORIENTATION_COLUMNS})
        records = records.merge(labels, on="alloy_id", how="left", validate="one_to_one")
        orientation_head, orientation_state = load_frozen_orientation_head(config, fold_root, train_ids, device)
        print("CTF orientation guidance enabled: frozen real-latent head, full-frame images", flush=True)
    dataset = EbsdDataset(
        records,
        train_ids,
        samples_per_alloy=int(config.train.samples_per_alloy),
        view="full",
        image_height=int(config.data.image_height),
        image_width=int(config.data.image_width),
        patch_size=int(config.data.patch_size),
        random_augment=not orientation_enabled,
        cache_images=bool(getattr(config.data, "cache_images", False)),
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
    vae, latent_scale = _load_vae(fold_root, device)
    latent_cache = {}
    if bool(getattr(config.performance, "cache_full_latents", False)):
        if dataset.random_augment or dataset.view != "full":
            raise ValueError("潜变量缓存仅允许完整确定性图像，不能冻结随机数据增强")
        print("缓存本折21张完整图的确定性VAE潜变量，仅省去重复编码；不缓存解码梯度。", flush=True)
        with torch.no_grad(), autocast_context(device, precision):
            for _, record in dataset.records.iterrows():
                frame = format_image_batch(dataset._image(record)[None].to(device), channels_last)
                encoded, _, _ = vae.encode(frame, sample=False)
                latent_cache[record["alloy_id"]] = (encoded * latent_scale).detach()
    mechanics = _load_mechanics(fold_root, device) if is_feedback else None
    condition_columns = (
        [column for column in records if column.startswith("comp_")]
        + [column for column in records if column.startswith("desc_cond_")]
        + [column for column in records if column.startswith("desc_std_")]
        + [f"curve_baseline_{name}" for name in CURVE_TARGET_COLUMNS]
    )
    condition_values = torch.tensor(
        records.loc[records["role"] == "train", condition_columns].to_numpy(dtype="float32")
    )
    model = ConditionalLatentUNet(
        latent_channels=int(config.model.latent_channels),
        condition_dim=len(condition_columns),
        base_channels=int(config.model.base_channels),
        channel_multipliers=tuple(config.model.channel_multipliers),
        attention_heads=int(config.model.attention_heads),
    )
    model = use_channels_last(model.to(device), channels_last)
    model.condition_encoder.set_statistics(
        condition_values.mean(0).to(device), condition_values.std(0).clamp_min(1e-6).to(device)
    )
    descriptor_weight = float(getattr(config.loss, "descriptor", 0.0))
    descriptor_head: LatentDescriptorHead | None = None
    descriptor_transform = None
    if descriptor_weight > 0.0:
        descriptor_head, descriptor_transform = load_frozen_proxy(fold_root, train_ids, device)
        print(
            f"已启用013冻结图像代理约束（解码IPF输入）: 权重={descriptor_weight:g}, "
            f"低噪声比例={float(getattr(config.loss, 'descriptor_low_noise_fraction', 0.3)):g}",
            flush=True,
        )
    schedule = DiffusionSchedule(int(config.diffusion.training_timesteps)).to(device)
    stage_folder = orientation_stage_folder(config, is_feedback)
    model_dir = fold_root / stage_folder
    run_dir = LOG_ROOT / holdout_id / stage_folder
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config.to_dict(), run_dir / "实际配置.json")
    start_step = 0
    base_path = fold_root / orientation_stage_folder(config, False) / "扩散_最终模型.pt"
    if orientation_enabled and not is_feedback and bool(config.orientation.initialize_from_baseline):
        baseline_path = fold_root / "03_基础条件扩散" / "扩散_最终模型.pt"
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
        if baseline.get("holdout_id") != holdout_id:
            raise ValueError("Baseline diffusion belongs to a different holdout fold")
        model.load_state_dict(baseline["model"])
        if descriptor_head is not None:
            if baseline.get("descriptor_head") is None or baseline.get("descriptor_transform") is None:
                raise ValueError("Expected the existing D2 baseline with its descriptor head")
            descriptor_head.load_state_dict(baseline["descriptor_head"])
            descriptor_transform = DescriptorTargetTransform.from_state_dict(baseline["descriptor_transform"])
        print(f"Warm-start CTF experiment from existing baseline: {baseline_path}", flush=True)
    if is_feedback and bool(config.train.load_base_checkpoint):
        if not base_path.exists():
            raise FileNotFoundError(f"反馈微调前缺少基础扩散模型: {base_path}")
        base_state = torch.load(base_path, map_location="cpu", weights_only=False)
        if orientation_enabled and (base_state.get("orientation_state") is None
                                   or base_state["orientation_state"]["labels_sha256"] != orientation_state["labels_sha256"]
                                   or base_state["orientation_state"].get("head_checkpoint_sha256") != orientation_state["head_checkpoint_sha256"]):
            raise ValueError("CTF base checkpoint/labels mismatch")
        model.load_state_dict(base_state["model"])
        if descriptor_head is not None and base_state.get("descriptor_head") is not None:
            descriptor_head.load_state_dict(base_state["descriptor_head"])
        if descriptor_transform is not None and base_state.get("descriptor_transform") is not None:
            descriptor_transform = DescriptorTargetTransform.from_state_dict(
                base_state["descriptor_transform"]
            )
        print(f"已载入基础扩散模型: {base_path}", flush=True)
        _set_feedback_train_scope(model, str(config.train.train_scope))
    ema = ExponentialMovingAverage(model, float(config.ema.decay))
    optimization_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    # Frozen image proxy remains differentiable w.r.t. decoded images only.
    optimizer = make_adamw(
        optimization_parameters,
        device,
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
        betas=(0.9, 0.95),
    )
    scheduler = make_cosine_scheduler(
        optimizer, int(config.train.warmup_steps), int(config.train.max_steps)
    )
    scaler = make_grad_scaler(device, precision)
    resume_path = model_dir / "最近断点.pt"
    if not bool(config.train.resume) or not resume_path.exists():
        resume_path = None
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if orientation_enabled and (state.get("orientation_state") is None
                                   or state["orientation_state"]["labels_sha256"] != orientation_state["labels_sha256"]
                                   or state["orientation_state"].get("head_checkpoint_sha256") != orientation_state["head_checkpoint_sha256"]):
            raise ValueError("CTF resume checkpoint/labels mismatch")
        model.load_state_dict(state["model"])
        if descriptor_head is not None and state.get("descriptor_head") is not None:
            descriptor_head.load_state_dict(state["descriptor_head"])
        if descriptor_transform is not None and state.get("descriptor_transform") is not None:
            descriptor_transform = DescriptorTargetTransform.from_state_dict(
                state["descriptor_transform"]
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema"])
        start_step = int(state["step"])
        print(f"从断点继续: {resume_path}", flush=True)
    # Keep the original module for checkpoint/state-dict compatibility; compile only
    # the callable used by denoising and the optional unrolled path.
    denoiser = maybe_compile(model, config, "扩散UNet")
    trainable, total = trainable_parameter_count(model)
    print(f"扩散参数: 可训练 {trainable:,} / 总计 {total:,}", flush=True)
    monitor = TrainingMonitor(
        run_dir,
        f"{holdout_id}-{'力学反馈' if is_feedback else '基础扩散'}",
        int(config.train.max_steps),
        DIFFUSION_METRICS,
        int(config.monitor.log_every),
        int(config.monitor.smoothing_window),
        bool(getattr(config.monitor, "tensorboard", True)),
    )
    accumulation = int(config.train.gradient_accumulation)
    max_steps = int(config.train.max_steps)
    last_condition: torch.Tensor | None = None
    try:
        for step in range(start_step + 1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            metrics = {name: 0.0 for name in DIFFUSION_METRICS}
            for micro_step in range(accumulation):
                batch = next(batches)
                image = format_image_batch(
                    batch["image"].to(device, non_blocking=True), channels_last
                )
                condition = build_condition(batch, device)
                with torch.no_grad(), autocast_context(device, precision):
                    if latent_cache:
                        latent = torch.cat([latent_cache[aid] for aid in batch["alloy_id"]])
                    else:
                        latent, _, _ = vae.encode(image, sample=False)
                        latent = latent * latent_scale
                noise = torch.randn_like(latent)
                timesteps = torch.randint(
                    0, schedule.timesteps, (latent.shape[0],), device=device, dtype=torch.long
                )
                noisy = schedule.add_noise(latent, noise, timesteps)
                drop_mask = torch.rand(latent.shape[0], device=device) < float(
                    config.model.condition_dropout
                )
                with autocast_context(device, precision):
                    predicted_noise = denoiser(noisy, timesteps, condition, drop_mask)
                    per_sample_noise = (predicted_noise - noise).float().square().flatten(1).mean(1)
                    snr_weight = schedule.min_snr_weight(
                        timesteps, float(config.diffusion.min_snr_gamma)
                    )
                    loss_noise = (per_sample_noise * snr_weight).mean()
                    predicted_clean = schedule.predict_clean(noisy, predicted_noise, timesteps).clamp(
                        -5.0, 5.0
                    )
                    loss_image = torch.zeros((), device=device)
                    loss_edge_value = torch.zeros((), device=device)
                    loss_boundary_value = torch.zeros((), device=device)
                    loss_haar_value = torch.zeros((), device=device)
                    loss_mechanics = torch.zeros((), device=device)
                    loss_mechanics_unrolled = torch.zeros((), device=device)
                    loss_descriptor = torch.zeros((), device=device)
                    descriptor_active = torch.zeros(
                        latent.shape[0], device=device, dtype=torch.bool
                    )
                    descriptor_predictions = None
                    loss_orientation = predicted_clean.sum() * 0.0
                    orientation_active = torch.zeros_like(drop_mask)
                    if orientation_head is not None:
                        loss_orientation, orientation_active = orientation_consistency(
                            predicted_clean, batch["orientation_true"].to(device), timesteps,
                            drop_mask, orientation_head, schedule.timesteps,
                            float(config.orientation.low_noise_fraction),
                        )
                    should_decode = step % int(config.loss.decode_every) == 0 or is_feedback or descriptor_head is not None
                    decoded = vae.decode(predicted_clean / latent_scale) if should_decode else None
                    loss_overlay_alignment = predicted_clean.sum() * 0.0
                    alignment_active = (timesteps < int(schedule.timesteps * float(config.loss.descriptor_low_noise_fraction))) & ~drop_mask
                    if decoded is not None and bool(alignment_active.any()):
                        unit = to_unit_range(decoded[alignment_active].float())
                        rgb_black = torch.sigmoid((.20 - unit[:, :3].amax(1, keepdim=True))*24)
                        loss_overlay_alignment = F.l1_loss(rgb_black, 1-unit[:, 3:4])
                    if descriptor_head is not None and descriptor_transform is not None:
                        loss_descriptor, descriptor_info = descriptor_consistency(
                            predicted_clean,
                            batch["descriptor_true"].to(device),
                            timesteps,
                            drop_mask,
                            descriptor_head,
                            descriptor_transform,
                            schedule.timesteps,
                            float(getattr(config.loss, "descriptor_low_noise_fraction", 0.3)),
                            float(getattr(config.loss, "descriptor_huber_beta", 0.5)),
                            decoded,
                        )
                        descriptor_active = descriptor_info["active"]
                        descriptor_predictions = descriptor_info["predictions"]
                    if should_decode:
                        loss_image = F.l1_loss(decoded, image)
                        loss_edge_value = edge_loss(decoded, image)
                        loss_boundary_value = boundary_loss(decoded, image)
                        loss_haar_value = haar_loss(decoded, image)
                        if mechanics is not None:
                            output = mechanics.predict_curve(
                                decoded,
                                batch["composition"].to(device),
                                batch["descriptor_condition"].to(device),
                                batch["curve_baseline"].to(device),
                            )
                            loss_mechanics = _curve_feedback_loss(
                                output["curve"],
                                batch["curve_true"].to(device),
                                mechanics.curve_scale,
                                config,
                                batch["curve_dense_strain"].to(device),
                                batch["curve_dense_stress"].to(device),
                                batch["curve_dense_mask"].to(device),
                            )
                    loss_descriptor_unrolled = predicted_clean.sum() * 0.0
                    unrolled_active = (
                        (is_feedback or float(getattr(config.loss, "unrolled_descriptor", 0.0)) > 0)
                        and micro_step == 0
                        and step >= int(config.loss.unrolled_start_step)
                        and step % int(config.loss.unrolled_every) == 0
                    )
                    if unrolled_active:
                        unrolled_latent = schedule.differentiable_ddim_sample(
                            denoiser,
                            tuple(latent.shape),
                            condition,
                            int(config.loss.unrolled_ddim_steps),
                            device,
                            use_checkpoint=True,
                        )
                        unrolled_image = vae.decode(unrolled_latent / latent_scale)
                        if descriptor_head is not None:
                            loss_descriptor_unrolled = F.smooth_l1_loss(
                                descriptor_head(unrolled_image[:, :3]).float(),
                                descriptor_transform.transform(batch["descriptor_condition"].to(device)), beta=.5)
                        if mechanics is not None:
                            unrolled_output = mechanics.predict_curve(unrolled_image,
                                batch["composition"].to(device), batch["descriptor_condition"].to(device),
                                batch["curve_baseline"].to(device))
                            loss_mechanics_unrolled = _curve_feedback_loss(unrolled_output["curve"],
                                batch["curve_true"].to(device), mechanics.curve_scale, config,
                                batch["curve_dense_strain"].to(device), batch["curve_dense_stress"].to(device),
                                batch["curve_dense_mask"].to(device))
                    mechanics_weight = 0.0
                    unrolled_mechanics_weight = 0.0
                    if is_feedback:
                        mechanics_weight = float(config.loss.mechanics) * min(
                            1.0, step / max(int(config.loss.mechanics_warmup_steps), 1)
                        )
                        unrolled_mechanics_weight = float(config.loss.unrolled_mechanics)
                    total_loss = (
                        float(config.loss.noise) * loss_noise
                        + float(config.loss.image) * loss_image
                        + float(config.loss.edge) * loss_edge_value
                        + float(config.loss.boundary) * loss_boundary_value
                        + float(config.loss.haar) * loss_haar_value
                        + mechanics_weight * loss_mechanics
                        + unrolled_mechanics_weight * loss_mechanics_unrolled * accumulation
                        + descriptor_weight * loss_descriptor
                        + float(getattr(config.loss, "unrolled_descriptor", 0.0)) * loss_descriptor_unrolled * accumulation
                        + float(getattr(config.loss, "overlay_alignment", 0.0)) * loss_overlay_alignment
                        + orientation_weight * loss_orientation
                    )
                if not bool(torch.isfinite(total_loss)):
                    raise RuntimeError("扩散损失出现NaN/Inf，停止写入最终模型")
                backward(total_loss / accumulation, scaler)
                values = {
                    "loss_total": total_loss,
                    "loss_noise": loss_noise,
                    "loss_image": loss_image,
                    "loss_edge": loss_edge_value,
                    "loss_boundary": loss_boundary_value,
                    "loss_haar": loss_haar_value,
                    "loss_mechanics": loss_mechanics,
                    "loss_mechanics_unrolled": loss_mechanics_unrolled,
                    "loss_descriptor": loss_descriptor,
                    "loss_descriptor_unrolled": loss_descriptor_unrolled,
                    "loss_overlay_alignment": loss_overlay_alignment,
                    "loss_orientation": loss_orientation,
                }
                for name, value in values.items():
                    metrics[name] += float(value.detach()) / accumulation
                metrics["mechanics_weight"] = mechanics_weight
                metrics["orientation_active_fraction"] += float(orientation_active.float().mean()) / accumulation
                metrics["descriptor_active_fraction"] += float(
                    descriptor_active.float().mean().detach()
                ) / accumulation
                if descriptor_predictions is not None and descriptor_transform is not None:
                    for key, value in descriptor_error_metrics(
                        descriptor_predictions,
                        batch["descriptor_true"].to(device),
                        descriptor_active,
                        descriptor_transform,
                    ).items():
                        metrics[key] += value / accumulation
                metrics["latent_abs_mean"] += float(latent.detach().abs().mean()) / accumulation
                metrics["predicted_clean_abs_mean"] += float(
                    predicted_clean.detach().abs().mean()
                ) / accumulation
                last_condition = condition.detach()
            condition_gradient_norm = _module_gradient_norm(model.condition_encoder)
            grad_norm = optimizer_step(
                optimizer, model, float(config.train.gradient_clip), scaler
            )
            scheduler.step()
            ema.update(model)
            metrics.update(
                condition_gradient_norm=condition_gradient_norm,
                grad_norm=grad_norm,
                lr=float(scheduler.get_last_lr()[0]),
            )
            monitor.log(step, metrics)

            if step % int(config.monitor.sample_every) == 0 and last_condition is not None:
                _save_sample(
                    denoiser,
                    schedule,
                    vae,
                    latent_scale,
                    last_condition,
                    int(config.data.image_height),
                    int(config.data.image_width),
                    min(int(config.diffusion.sampling_steps), 50),
                    float(config.diffusion.guidance_scale),
                    run_dir / "生成样例" / f"step_{step:08d}.png",
                )
            checkpoint_every = int(config.monitor.checkpoint_every)
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                state: dict[str, Any] = {
                    "orientation_state": orientation_state,
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "model_config": config.model.to_dict(),
                    "latent_scale": latent_scale,
                    "condition_columns": condition_columns,
                    "holdout_id": holdout_id,
                    "stage": str(config.stage),
                    "descriptor_head": descriptor_head.state_dict()
                    if descriptor_head is not None
                    else None,
                    "descriptor_head_config": descriptor_head.config()
                    if descriptor_head is not None
                    else None,
                    "descriptor_transform": descriptor_transform.state_dict()
                    if descriptor_transform is not None
                    else None,
                }
                atomic_torch_save(state, model_dir / "最近断点.pt")

        final_path = model_dir / "扩散_最终模型.pt"
        atomic_torch_save(
            {
                "orientation_state": orientation_state,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "model_config": config.model.to_dict(),
                "latent_scale": latent_scale,
                "condition_columns": condition_columns,
                "holdout_id": holdout_id,
                "stage": str(config.stage),
                "descriptor_head": descriptor_head.state_dict()
                if descriptor_head is not None
                else None,
                "descriptor_head_config": descriptor_head.config()
                if descriptor_head is not None
                else None,
                "descriptor_transform": descriptor_transform.state_dict()
                if descriptor_transform is not None
                else None,
                "sampling": {
                    "training_timesteps": int(config.diffusion.training_timesteps),
                    "sampling_steps": int(config.diffusion.sampling_steps),
                    "guidance_scale": float(config.diffusion.guidance_scale),
                },
            },
            final_path,
        )
        # Final weights supersede the rolling recovery checkpoint.
        (model_dir / "最近断点.pt").unlink(missing_ok=True)
        (model_dir / "扩散训练说明.json").write_text(
            json.dumps(
                {
                    "holdout_id": holdout_id,
                    "stage": str(config.stage),
                    "mechanics_feedback": is_feedback,
                    "descriptor_guidance": descriptor_head is not None,
                    "trainable_parameters": trainable,
                    "total_parameters": total,
                    "latent_scale": latent_scale,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        monitor.close("completed")
        return final_path
    except KeyboardInterrupt:
        if int(config.monitor.checkpoint_every) > 0:
            atomic_torch_save(
                {
                    "step": step,
                    "orientation_state": orientation_state,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "model_config": config.model.to_dict(),
                    "latent_scale": latent_scale,
                    "condition_columns": condition_columns,
                    "holdout_id": holdout_id,
                    "stage": str(config.stage),
                    "descriptor_head": descriptor_head.state_dict()
                    if descriptor_head is not None
                    else None,
                    "descriptor_head_config": descriptor_head.config()
                    if descriptor_head is not None
                    else None,
                    "descriptor_transform": descriptor_transform.state_dict()
                    if descriptor_transform is not None
                    else None,
                },
                model_dir / "最近断点.pt",
            )
        monitor.close("interrupted")
        raise
    except Exception:
        monitor.close("failed")
        raise
