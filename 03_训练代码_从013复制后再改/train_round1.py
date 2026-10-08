# -*- coding: utf-8 -*-
"""ID03 hold-out, fold 1. Noise MSE only.

Hyperparameters taken from 013, not guessed:
  D:\\013_ipf_gb\\06_配置文件\\03_基础条件扩散.yaml
    train.learning_rate = 0.0001
    train.weight_decay = 0.0001
    train.gradient_clip = 1.0
    model.latent_channels = 4
    model.base_channels = 128
    model.channel_multipliers = [1, 2, 3, 4]
    model.attention_heads = 8
    model.condition_dropout = 0.15
    diffusion.training_timesteps = 1000
    diffusion.min_snr_gamma = 5.0
    ema.decay = 0.9999
  AdamW betas (0.9, 0.95) are not in that yaml. They are hardcoded in
    013代码/src/ebsd_feedback/training/diffusion.py
    and training/common.py make_adamw.
  The beta schedule is cosine_beta_schedule(timesteps, offset=0.008) in
    013代码/src/ebsd_feedback/models/diffusion.py
    DiffusionSchedule(training_timesteps). The yaml has no beta_start/beta_end.

The same yaml sets warmup_steps=1500 and max_steps=80000, and common.py then
applies a cosine learning-rate decay. This fold is capped at 4200 steps, so
that 1500-step warmup is not applied. Learning rate stays 0.0001.
torch.compile from 基础配置.yaml is also not turned on.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Grains have to be visible before this line changes. Do not set it nonzero
# in this fold: a nonzero weight would require decoding, which this fold does not do.
MECHANICS_LOSS_WEIGHT = 0.0

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
ADAMW_BETAS = (0.9, 0.95)
GRADIENT_CLIP = 1.0
TRAINING_TIMESTEPS = 1000
COSINE_BETA_OFFSET = 0.008
MIN_SNR_GAMMA = 5.0
EMA_DECAY = 0.9999
CONDITION_DROPOUT = 0.15
LOSS_NOISE = 1.0
LATENT_SCALE = 0.3440041526837966
LATENT_WIN = 64
HOLDOUT = "ID03"
SEED_BASE = 20261008
DRAWS_PER_ALLOY = 16
N_ALLOYS = 21
N_CROPS = N_ALLOYS * 256

LOSS_IMAGE = 0.0
LOSS_EDGE = 0.0
LOSS_BOUNDARY = 0.0
LOSS_HAAR = 0.0
LOSS_DESCRIPTOR = 0.0
LOSS_UNROLLED_DESCRIPTOR = 0.0
LOSS_OVERLAY = 0.0

COND_COLUMNS = (
    [f"comp_{n}" for n in ["Ni", "Fe", "Cr", "Nb", "Ta", "Co", "Mo", "Al", "Ti"]]
    + [f"desc_cond_{n}" for n in [
        "grain_size_median_um", "grain_size_log_spread", "area_weighted_aspect_ratio",
        "coarse_grain_area_fraction", "boundary_length_density_per_um"]]
    + [f"desc_std_{n}" for n in [
        "grain_size_median_um", "grain_size_log_spread", "area_weighted_aspect_ratio",
        "coarse_grain_area_fraction", "boundary_length_density_per_um"]]
    + [f"curve_baseline_{n}" for n in [
        "stress_at_0p5_pct_MPa", "stress_at_1p0_pct_MPa", "stress_at_2p0_pct_MPa",
        "stress_at_5p0_pct_MPa", "stress_at_10p0_pct_MPa", "peak_stress_MPa",
        "peak_strain_pct", "endpoint_strain_pct", "endpoint_stress_MPa"]]
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _import_unet():
    pkg = Path(__file__).resolve().parent / "013代码" / "src"
    if not pkg.is_dir():
        raise FileNotFoundError(f"缺少 013 的 U-Net 源码: {pkg}")
    sys.path.insert(0, str(pkg))
    from ebsd_feedback.models.diffusion import ConditionalLatentUNet, DiffusionSchedule, cosine_beta_schedule
    return ConditionalLatentUNet, DiffusionSchedule, cosine_beta_schedule


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def condition_matrix(rows: list[dict[str, str]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    missing = [name for name in COND_COLUMNS if name not in rows[0]]
    if missing:
        raise RuntimeError(f"conditions.csv 缺列: {missing}")
    by_id: dict[str, torch.Tensor] = {}
    stacked = []
    for row in rows:
        if row["alloy_id"] == HOLDOUT:
            raise RuntimeError("训练条件表里不该有 ID03")
        values = torch.tensor([float(row[name]) for name in COND_COLUMNS], dtype=torch.float32)
        if row["alloy_id"] in by_id:
            raise RuntimeError(f"条件表重复: {row['alloy_id']}")
        by_id[row["alloy_id"]] = values
        stacked.append(values)
    if len(by_id) != N_ALLOYS:
        raise RuntimeError(f"训练合金应为 {N_ALLOYS} 个，实际 {len(by_id)}")
    return by_id, torch.stack(stacked, dim=0)


def group_crops(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["alloy_id"] == HOLDOUT:
            raise RuntimeError("crops.csv 里不该有 ID03")
        groups[row["alloy_id"]].append(index)
    if len(groups) != N_ALLOYS or len(rows) != N_CROPS:
        raise RuntimeError(f"清单应为 {N_ALLOYS} 个合金、{N_CROPS} 行，实际合金 {len(groups)} 行 {len(rows)}")
    for alloy_id, indexes in groups.items():
        if len(indexes) != 256:
            raise RuntimeError(f"{alloy_id} 应有 256 个窗口，实际 {len(indexes)}")
    return groups


class LatentStore:
    def __init__(self, latent_dir: Path) -> None:
        self.latent_dir = latent_dir
        self.cache: dict[tuple[str, str], torch.Tensor] = {}

    def get(self, alloy_id: str, view: str) -> torch.Tensor:
        key = (alloy_id, view)
        if key not in self.cache:
            path = self.latent_dir / f"{alloy_id}_{view}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            array = np.load(path)
            if array.ndim != 3 or array.shape[0] != 4:
                raise RuntimeError(f"{path.name} 形状应为 4xH xW，实际 {array.shape}")
            self.cache[key] = torch.from_numpy(np.ascontiguousarray(array))
        return self.cache[key]


def crop_batch(store: LatentStore, rows: list[dict[str, str]]) -> torch.Tensor:
    crops = []
    for row in rows:
        latent = store.get(row["alloy_id"], row["view"])
        y = int(row["latent_y"])
        x = int(row["latent_x"])
        if y < 0 or x < 0 or y + LATENT_WIN > latent.shape[1] or x + LATENT_WIN > latent.shape[2]:
            raise RuntimeError(f"窗口越界 {row['alloy_id']} {row['view']} y={y} x={x} latent={tuple(latent.shape)}")
        crops.append(latent[:, y:y + LATENT_WIN, x:x + LATENT_WIN])
    return torch.stack(crops, dim=0)


def epoch_order(groups: dict[str, list[int]], rng: np.random.Generator, batch_size: int) -> np.ndarray:
    chosen = []
    for alloy_id in sorted(groups):
        pool = np.asarray(groups[alloy_id], dtype=np.int64)
        chosen.append(rng.choice(pool, size=DRAWS_PER_ALLOY, replace=False))
    order = np.concatenate(chosen)
    rng.shuffle(order)
    if len(order) != N_ALLOYS * DRAWS_PER_ALLOY:
        raise RuntimeError(f"一轮应有 {N_ALLOYS * DRAWS_PER_ALLOY} 条，实际 {len(order)}")
    if len(order) % batch_size != 0:
        raise RuntimeError(f"{len(order)} 不能被 batch {batch_size} 整除。16 或 8 可以。")
    return order


def noise_loss(schedule, predicted: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    per_sample = (predicted.float() - noise.float()).square().flatten(1).mean(1)
    return (per_sample * schedule.min_snr_weight(timesteps, MIN_SNR_GAMMA)).mean()


def build_model(ConditionalLatentUNet, values: torch.Tensor, device: torch.device):
    model = ConditionalLatentUNet(
        latent_channels=4,
        condition_dim=len(COND_COLUMNS),
        base_channels=128,
        channel_multipliers=(1, 2, 3, 4),
        attention_heads=8,
    ).to(device)
    model.condition_encoder.set_statistics(
        values.mean(0).to(device),
        values.std(0).clamp_min(1e-6).to(device),
    )
    return model


def save_checkpoint(path: Path, model, ema, optimizer, epoch: int, step: int, values: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "condition_mean": values.mean(0),
            "condition_std": values.std(0),
            "mechanics_loss_weight": MECHANICS_LOSS_WEIGHT,
            "latent_scale": LATENT_SCALE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adamw_betas": ADAMW_BETAS,
            "training_timesteps": TRAINING_TIMESTEPS,
            "cosine_beta_offset": COSINE_BETA_OFFSET,
            "min_snr_gamma": MIN_SNR_GAMMA,
            "condition_dropout": CONDITION_DROPOUT,
            "condition_columns": COND_COLUMNS,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    root = project_root()
    table_dir = root / "01_窗口清单_每个窗口的位置和描述符"
    parser = argparse.ArgumentParser(description="016 fold-1 patch diffusion, noise loss only")
    parser.add_argument("--latent-dir", type=Path, default=root / "02_整图潜变量_八个视角")
    parser.add_argument("--crops", type=Path, default=table_dir / "crops.csv")
    parser.add_argument("--val-strip", type=Path, default=table_dir / "val_strip.csv")
    parser.add_argument("--conditions", type=Path, default=table_dir / "conditions.csv")
    parser.add_argument("--log-dir", type=Path, default=root / "04_训练日志")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--smoke", action="store_true", help="One CPU/GPU forward of one real 64x64 window, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if MECHANICS_LOSS_WEIGHT != 0.0:
        raise RuntimeError("这一折 MECHANICS_LOSS_WEIGHT 必须是 0。非 0 会要解码，这一折不算。")
    if args.epochs > 200:
        raise RuntimeError("第一轮上限 200 轮，不要退回 80000 步")
    if args.batch_size not in (8, 16):
        raise RuntimeError("batch 只用 16。显存不够改成 8。不要改成别的数。")
    ConditionalLatentUNet, DiffusionSchedule, cosine_beta_schedule = _import_unet()
    probe = cosine_beta_schedule(TRAINING_TIMESTEPS, offset=COSINE_BETA_OFFSET)
    if probe.shape != (TRAINING_TIMESTEPS,):
        raise RuntimeError("余弦噪声日程长度不对")
    crops = read_table(args.crops)
    groups = group_crops(crops)
    cond_map, cond_values = condition_matrix(read_table(args.conditions))
    missing_cond = sorted(set(groups) - set(cond_map))
    if missing_cond:
        raise RuntimeError(f"这些合金没有 28 维条件: {missing_cond}")
    store = LatentStore(args.latent_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "bf16" if device.type == "cuda" else "fp32"
    print(
        f"device={device.type} precision={precision} batch={args.batch_size} "
        f"lr={LEARNING_RATE} betas={ADAMW_BETAS} timesteps={TRAINING_TIMESTEPS} "
        f"min_snr_gamma={MIN_SNR_GAMMA} mechanics_weight={MECHANICS_LOSS_WEIGHT}",
        flush=True,
    )
    if args.smoke:
        row = crops[0]
        latent = crop_batch(store, [row]).to(device)
        condition = cond_map[row["alloy_id"]].to(device)[None]
        if latent.shape != (1, 4, LATENT_WIN, LATENT_WIN):
            raise RuntimeError(f"smoke 窗口形状不对: {tuple(latent.shape)}")
        model = build_model(ConditionalLatentUNet, cond_values, device)
        schedule = DiffusionSchedule(TRAINING_TIMESTEPS).to(device)
        model.train()
        noise = torch.randn_like(latent)
        timesteps = torch.randint(0, schedule.timesteps, (1,), device=device)
        noisy = schedule.add_noise(latent, noise, timesteps)
        drop = torch.zeros(1, device=device, dtype=torch.bool)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else _null():
            predicted = model(noisy, timesteps, condition, drop)
        loss = noise_loss(schedule, predicted, noise, timesteps)
        n_param = sum(p.numel() for p in model.parameters())
        print(
            f"smoke_ok loss_noise={float(loss.detach()):.6f} "
            f"window={row['alloy_id']}_{row['view']}_y{row['latent_y']}_x{row['latent_x']} "
            f"pred={tuple(predicted.shape)} params={n_param} "
            f"beta0={float(schedule.betas[0]):.6e} beta_last={float(schedule.betas[-1]):.6e}",
            flush=True,
        )
        return
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    val_rows = [row for row in read_table(args.val_strip) if row["alloy_id"] != HOLDOUT]
    if not val_rows:
        raise RuntimeError("val_strip.csv 没有训练合金的右侧窗口")
    model = build_model(ConditionalLatentUNet, cond_values, device)
    schedule = DiffusionSchedule(TRAINING_TIMESTEPS).to(device)
    ema = ExponentialMovingAverage(model, EMA_DECAY)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAMW_BETAS,
        fused=device.type == "cuda",
    )
    steps_per_epoch = (N_ALLOYS * DRAWS_PER_ALLOY) // args.batch_size
    print(f"steps_per_epoch={steps_per_epoch} cap_steps={args.epochs * steps_per_epoch}", flush=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "这一折训练配置.json").write_text(
        json.dumps(
            {
                "holdout": HOLDOUT,
                "epochs_cap": args.epochs,
                "batch_size": args.batch_size,
                "steps_per_epoch": steps_per_epoch,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "adamw_betas": list(ADAMW_BETAS),
                "warmup_applied": False,
                "warmup_steps_in_yaml_for_80000": 1500,
                "gradient_clip": GRADIENT_CLIP,
                "training_timesteps": TRAINING_TIMESTEPS,
                "noise_schedule": "cosine_beta_schedule",
                "cosine_beta_offset": COSINE_BETA_OFFSET,
                "min_snr_gamma": MIN_SNR_GAMMA,
                "loss_noise_weight": LOSS_NOISE,
                "mechanics_loss_weight": MECHANICS_LOSS_WEIGHT,
                "weights_off": ["image", "edge", "boundary", "haar", "descriptor", "unrolled_descriptor", "overlay"],
                "condition_dropout": CONDITION_DROPOUT,
                "ema_decay": EMA_DECAY,
                "precision": precision,
                "yaml": "013/06_配置文件/03_基础条件扩散.yaml",
                "betas_source": "013代码/src/ebsd_feedback/training/diffusion.py AdamW betas=(0.9, 0.95)",
                "early_stop": "not inside this script; run sample_id03.py --mode memorization on epoch 50 and 100 checkpoints",
                "oom_fallback": "--batch-size 8 is 42 steps per epoch, still 200 epochs",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log_path = args.log_dir / "train_noise_loss.csv"
    fields = ["epoch", "step", "loss_noise", "loss_mechanics", "val_noise"]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    torch.manual_seed(args.seed)
    step = 0
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng([args.seed, epoch])
        order = epoch_order(groups, rng, args.batch_size)
        model.train()
        epoch_noise = []
        for start in range(0, len(order), args.batch_size):
            batch_rows = [crops[int(i)] for i in order[start:start + args.batch_size]]
            latent = crop_batch(store, batch_rows).to(device, non_blocking=True)
            condition = torch.stack([cond_map[row["alloy_id"]] for row in batch_rows], 0).to(device, non_blocking=True)
            noise = torch.randn_like(latent)
            timesteps = torch.randint(0, schedule.timesteps, (latent.shape[0],), device=device)
            noisy = schedule.add_noise(latent, noise, timesteps)
            drop = torch.rand(latent.shape[0], device=device) < CONDITION_DROPOUT
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else _null():
                predicted = model(noisy, timesteps, condition, drop)
                loss = LOSS_NOISE * noise_loss(schedule, predicted, noise, timesteps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            ema.update(model)
            step += 1
            loss_value = float(loss.detach())
            epoch_noise.append(loss_value)
            is_last = start + args.batch_size >= len(order)
            val_noise = ""
            if is_last:
                val_noise = f"{_val_noise(model, schedule, store, val_rows, cond_map, device, args.batch_size):.8f}"
                model.train()
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow({
                    "epoch": epoch,
                    "step": step,
                    "loss_noise": f"{loss_value:.8f}",
                    "loss_mechanics": "0.0",
                    "val_noise": val_noise,
                })
        print(
            f"epoch {epoch} train_noise {float(np.mean(epoch_noise)):.4f} val_noise {val_noise}",
            flush=True,
        )
        if epoch % 50 == 0 or epoch == args.epochs:
            save_checkpoint(args.log_dir / f"checkpoint_epoch{epoch:03d}.pt", model, ema, optimizer, epoch, step, cond_values)
            print(f"saved checkpoint_epoch{epoch:03d}.pt", flush=True)


def _null():
    import contextlib
    return contextlib.nullcontext()


@torch.no_grad()
def _val_noise(model, schedule, store, rows, cond_map, device, batch_size) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        latent = crop_batch(store, batch_rows).to(device, non_blocking=True)
        condition = torch.stack([cond_map[row["alloy_id"]] for row in batch_rows], 0).to(device, non_blocking=True)
        noise = torch.randn_like(latent)
        timesteps = torch.randint(0, schedule.timesteps, (latent.shape[0],), device=device)
        noisy = schedule.add_noise(latent, noise, timesteps)
        predicted = model(noisy, timesteps, condition, None)
        per_sample = (predicted.float() - noise.float()).square().flatten(1).mean(1)
        weighted = per_sample * schedule.min_snr_weight(timesteps, MIN_SNR_GAMMA)
        total += float(weighted.sum())
        count += int(weighted.shape[0])
    if count == 0:
        raise RuntimeError("验证条是空的")
    return total / count


if __name__ == "__main__":
    main()
