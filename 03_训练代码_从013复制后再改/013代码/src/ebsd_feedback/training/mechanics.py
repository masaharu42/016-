from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import ConfigNode
from ..constants import CURVE_TARGET_COLUMNS, DESCRIPTOR_COLUMNS, canonical_alloy_id
from ..data import EbsdDataset, load_fold_manifest
from ..models.mechanics import (
    EbsdMechanicsSurrogate,
    differentiable_curve_from_targets,
    encode_curve_targets,
)
from ..monitor import TrainingMonitor
from ..paths import FOLD_MODEL_ROOT, LOG_ROOT
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
    optimizer_step,
    use_channels_last,
)


MECHANICS_METRICS = [
    "loss_total",
    "loss_fused_residual",
    "loss_image_residual",
    "loss_physical_curve",
    "loss_dense_curve",
    "loss_early_dense_curve",
    "loss_early_slope",
    "validation_loss",
    "validation_curve_rmse_normalized",
    "lr",
    "grad_norm",
    "phase",
]


def _statistics(records, train_ids: list[str]) -> dict[str, torch.Tensor]:
    rows = records[records["alloy_id"].isin(train_ids)]
    arrays = {
        "composition": rows[[column for column in rows if column.startswith("comp_")]].to_numpy(
            dtype=np.float32
        ),
        "descriptor": rows[
            [column for column in rows if column.startswith("desc_cond_")]
        ].to_numpy(dtype=np.float32),
        "baseline": rows[
            [f"curve_baseline_{name}" for name in CURVE_TARGET_COLUMNS]
        ].to_numpy(dtype=np.float32),
        "curve": rows[[f"curve_true_{name}" for name in CURVE_TARGET_COLUMNS]].to_numpy(
            dtype=np.float32
        ),
    }
    curve = torch.from_numpy(arrays["curve"])
    baseline = torch.from_numpy(arrays["baseline"])
    parameter_residual = encode_curve_targets(curve) - encode_curve_targets(baseline)
    return {
        "composition_mean": torch.from_numpy(arrays["composition"].mean(0)),
        "composition_std": torch.from_numpy(arrays["composition"].std(0)).clamp_min(1e-6),
        "descriptor_mean": torch.from_numpy(arrays["descriptor"].mean(0)),
        "descriptor_std": torch.from_numpy(arrays["descriptor"].std(0)).clamp_min(1e-6),
        "residual_mean": parameter_residual.mean(0),
        "residual_std": parameter_residual.std(0, unbiased=False).clamp_min(1e-3),
        "curve_scale": torch.from_numpy(arrays["curve"].std(0)).clamp_min(1e-6),
    }


def _compute_losses(
    model: EbsdMechanicsSurrogate,
    batch: dict[str, torch.Tensor],
    config: ConfigNode,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    image = format_image_batch(
        batch["image"].to(device, non_blocking=True),
        bool(getattr(getattr(config, "performance", None), "channels_last", False)),
    )
    composition = batch["composition"].to(device, non_blocking=True)
    descriptor = batch["descriptor_condition"].to(device, non_blocking=True)
    baseline = batch["curve_baseline"].to(device, non_blocking=True)
    curve = batch["curve_true"].to(device, non_blocking=True)
    output = model.predict_curve(image, composition, descriptor, baseline)
    true_parameters = encode_curve_targets(curve, model.endpoint_min_strain_pct)
    baseline_parameters = encode_curve_targets(baseline, model.endpoint_min_strain_pct)
    parameter_residual = true_parameters - baseline_parameters
    residual_normalized = (
        parameter_residual - model.residual_mean
    ) / model.residual_std.clamp_min(1e-6)
    fused = F.smooth_l1_loss(output["residual_normalized"], residual_normalized, beta=0.5)
    image_only = F.smooth_l1_loss(
        output["image_residual_normalized"], residual_normalized, beta=0.5
    )
    physical = F.smooth_l1_loss(
        (output["curve"] - curve) / model.curve_scale.clamp_min(1e-6),
        torch.zeros_like(curve),
        beta=0.5,
    )
    dense_grid = batch["curve_dense_strain"].to(device, non_blocking=True)
    dense_target = batch["curve_dense_stress"].to(device, non_blocking=True)
    dense_mask = batch["curve_dense_mask"].to(device, non_blocking=True)
    dense_prediction = differentiable_curve_from_targets(output["curve"], dense_grid)
    stress_scale = model.curve_scale[[0, 1, 2, 3, 4, 5, 8]].mean().clamp_min(1e-6)
    dense_error = (dense_prediction - dense_target) / stress_scale
    dense_elements = F.smooth_l1_loss(
        dense_error,
        torch.zeros_like(dense_error),
        beta=0.5,
        reduction="none",
    )
    dense = (dense_elements * dense_mask).sum() / dense_mask.sum().clamp_min(1)
    early_mask = dense_mask * (dense_grid <= 2.0)
    early_dense = (dense_elements * early_mask).sum() / early_mask.sum().clamp_min(1)
    early_intervals = curve.new_tensor([0.5, 0.5, 1.0])
    predicted_early = torch.cat(
        [output["curve"][:, :1], output["curve"][:, 1:3] - output["curve"][:, :2]],
        dim=1,
    ) / early_intervals
    target_early = torch.cat(
        [curve[:, :1], curve[:, 1:3] - curve[:, :2]], dim=1
    ) / early_intervals
    early_slope = F.smooth_l1_loss(
        (predicted_early - target_early) / stress_scale,
        torch.zeros_like(target_early),
        beta=0.5,
    )
    total = (
        float(config.loss.fused_residual) * fused
        + float(config.loss.image_only_residual) * image_only
        + float(config.loss.physical_curve) * physical
        + float(getattr(config.loss, "dense_curve", 0.5)) * dense
        + float(getattr(config.loss, "early_dense_curve", 0.75)) * early_dense
        + float(getattr(config.loss, "early_slope", 0.75)) * early_slope
    )
    return {
        "loss_total": total,
        "loss_fused_residual": fused,
        "loss_image_residual": image_only,
        "loss_physical_curve": physical,
        "loss_dense_curve": dense,
        "loss_early_dense_curve": early_dense,
        "loss_early_slope": early_slope,
    }, output


@torch.no_grad()
def _validate(
    model: EbsdMechanicsSurrogate,
    loader: DataLoader,
    config: ConfigNode,
    device: torch.device,
    precision: str,
) -> tuple[float, float]:
    model.eval()
    losses, squared_errors = [], []
    for batch in loader:
        with autocast_context(device, precision):
            batch_losses, output = _compute_losses(model, batch, config, device)
        curve = batch["curve_true"].to(device)
        normalized_error = (output["curve"] - curve) / model.curve_scale.clamp_min(1e-6)
        losses.append(float(batch_losses["loss_total"]))
        squared_errors.append(normalized_error.float().square().mean().item())
    model.train()
    return float(np.mean(losses)), float(np.sqrt(np.mean(squared_errors)))


def train_mechanics(config: ConfigNode, holdout_id: str) -> Path:
    holdout_id = canonical_alloy_id(holdout_id)
    seed_everything(int(config.seed), bool(config.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(config.environment.precision)
    configure_torch_performance(config, device)
    performance = getattr(config, "performance", None)
    channels_last = bool(getattr(performance, "channels_last", False))
    fold_root = FOLD_MODEL_ROOT / holdout_id
    manifest_path = fold_root / "00_折准备" / "严格留一折清单.csv"
    records = load_fold_manifest(manifest_path)
    outer_train_ids = records.loc[records["role"] == "train", "alloy_id"].tolist()
    if holdout_id in outer_train_ids:
        raise RuntimeError("防泄漏检查失败：留出合金进入力学代理")
    generator = np.random.default_rng(int(config.seed))
    shuffled = list(generator.permutation(outer_train_ids))
    validation_count = int(config.train.validation_alloy_count)
    validation_ids = sorted(shuffled[:validation_count])
    development_ids = sorted(shuffled[validation_count:])
    stats = _statistics(records, development_ids)
    model = EbsdMechanicsSurrogate(
        descriptor_dim=len(DESCRIPTOR_COLUMNS),
        image_feature_dim=int(config.model.image_feature_dim),
        image_base_channels=int(getattr(config.model, "image_base_channels", 32)),
        dropout=float(config.model.dropout),
        endpoint_min_strain_pct=float(
            getattr(config.model, "endpoint_min_strain_pct", 10.0)
        ),
    )
    model = use_channels_last(model.to(device), channels_last)
    model.set_statistics(**{key: value.to(device) for key, value in stats.items()})

    dataset_args = dict(
        records=records,
        view="full",
        image_height=int(config.data.image_height),
        image_width=int(config.data.image_width),
        patch_size=int(config.data.patch_size),
        cache_images=bool(getattr(config.data, "cache_images", False)),
    )
    train_dataset = EbsdDataset(
        alloy_ids=development_ids,
        samples_per_alloy=int(config.train.samples_per_alloy),
        random_augment=True,
        **dataset_args,
    )
    validation_dataset = EbsdDataset(
        alloy_ids=validation_ids,
        samples_per_alloy=1,
        random_augment=False,
        **dataset_args,
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    batches = infinite_batches(train_loader)
    total_steps = int(config.train.max_steps) + int(config.train.refit_all_steps)
    optimizer = make_adamw(
        model.parameters(),
        device,
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
        betas=(0.9, 0.95),
    )
    scheduler = make_cosine_scheduler(optimizer, int(config.train.warmup_steps), total_steps)
    scaler = make_grad_scaler(device, precision)
    model_dir = fold_root / "02_力学代理"
    run_dir = LOG_ROOT / holdout_id / "02_力学代理"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config.to_dict(), run_dir / "实际配置.json")
    monitor = TrainingMonitor(
        run_dir,
        f"{holdout_id}-力学代理",
        total_steps,
        MECHANICS_METRICS,
        int(config.monitor.log_every),
        int(config.monitor.smoothing_window),
        bool(getattr(config.monitor, "tensorboard", True)),
    )
    start_step = 0
    best_validation = float("inf")
    resume_path = model_dir / "最近断点.pt"
    if not bool(config.train.resume) or not resume_path.exists():
        resume_path = None
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        best_validation = float(state.get("best_validation", best_validation))
        print(f"从断点继续: {resume_path}", flush=True)

    best_path = model_dir / "内部验证最佳模型.pt"
    # Resume in the refit phase must continue using all 21 training alloys.
    # The 012 loop only switched loaders at the exact phase boundary.
    if start_step > int(config.train.max_steps):
        all_dataset = EbsdDataset(alloy_ids=outer_train_ids,
            samples_per_alloy=int(config.train.samples_per_alloy), random_augment=False, **dataset_args)
        batches = infinite_batches(DataLoader(all_dataset, batch_size=int(config.train.batch_size),
            shuffle=True, drop_last=True, **loader_options))
    try:
        for step in range(start_step + 1, total_steps + 1):
            if step == int(config.train.max_steps) + 1:
                if best_path.exists():
                    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
                all_dataset = EbsdDataset(
                    alloy_ids=outer_train_ids,
                    samples_per_alloy=int(config.train.samples_per_alloy),
                    random_augment=True,
                    **dataset_args,
                )
                all_loader = DataLoader(
                    all_dataset,
                    batch_size=int(config.train.batch_size),
                    shuffle=True,
                    num_workers=int(config.data.num_workers),
                    pin_memory=bool(config.data.pin_memory),
                    persistent_workers=bool(config.data.persistent_workers)
                    and int(config.data.num_workers) > 0,
                    worker_init_fn=worker_seed,
                    drop_last=True,
                )
                batches = infinite_batches(all_loader)
                print("内部验证完成，加载最佳模型并用外层训练侧21个合金短程重拟合。", flush=True)
            optimizer.zero_grad(set_to_none=True)
            batch = next(batches)
            with autocast_context(device, precision):
                losses, _ = _compute_losses(model, batch, config, device)
            if not bool(torch.isfinite(losses["loss_total"])):
                raise RuntimeError("力学代理损失出现NaN/Inf，停止写入最终模型")
            backward(losses["loss_total"], scaler)
            grad_norm = optimizer_step(
                optimizer, model, float(config.train.gradient_clip), scaler
            )
            scheduler.step()
            metrics = {name: float(value.detach()) for name, value in losses.items()}
            metrics.update(
                validation_loss=float("nan"),
                validation_curve_rmse_normalized=float("nan"),
                lr=float(scheduler.get_last_lr()[0]),
                grad_norm=grad_norm,
                phase=0.0 if step <= int(config.train.max_steps) else 1.0,
            )
            if (
                step <= int(config.train.max_steps)
                and step % int(config.train.validation_every) == 0
            ):
                validation_loss, validation_rmse = _validate(
                    model, validation_loader, config, device, precision
                )
                metrics["validation_loss"] = validation_loss
                metrics["validation_curve_rmse_normalized"] = validation_rmse
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    atomic_torch_save(
                        {
                            "step": step,
                            "model": model.state_dict(),
                            "best_validation": best_validation,
                            "validation_ids": validation_ids,
                        },
                        best_path,
                    )
            monitor.log(step, metrics)
            checkpoint_every = int(config.monitor.checkpoint_every)
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                state = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_validation": best_validation,
                    "development_ids": development_ids,
                    "validation_ids": validation_ids,
                    "holdout_id": holdout_id,
                    "model_config": config.model.to_dict(),
                }
                atomic_torch_save(state, model_dir / "最近断点.pt")

        final_path = model_dir / "力学代理_最终模型.pt"
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "model_config": config.model.to_dict(),
                "best_validation": best_validation,
                "development_ids": development_ids,
                "validation_ids": validation_ids,
                "refit_ids": outer_train_ids,
                "holdout_id": holdout_id,
            },
            final_path,
        )
        (model_dir / "最近断点.pt").unlink(missing_ok=True)
        (model_dir / "力学代理说明.json").write_text(
            json.dumps(
                {
                    "holdout_id": holdout_id,
                    "best_inner_validation_loss": best_validation,
                    "development_ids": development_ids,
                    "validation_ids": validation_ids,
                    "refit_all_training_ids": outer_train_ids,
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
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_validation": best_validation,
                    "development_ids": development_ids,
                    "validation_ids": validation_ids,
                    "holdout_id": holdout_id,
                    "model_config": config.model.to_dict(),
                },
                model_dir / "最近断点.pt",
            )
        monitor.close("interrupted")
        raise
    except Exception:
        monitor.close("failed")
        raise
