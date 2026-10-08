# -*- coding: utf-8 -*-
"""Continue ID03 patch diffusion from the round-1 noise checkpoint the user passes.

Round 1 (train_round1.py) stays noise-only. This script does not replace it.
Pass --checkpoint. On the server that file is 04_训练日志/checkpoint_epoch200.pt.
Do not resume checkpoint_epoch400.pt or checkpoint_epoch600.pt.
Each invocation trains at most 200 more epochs.

Image-structure terms are the four that training/diffusion.py adds on top of
the noise loss when it decodes (not the VAE-only SSIM, flatness, or KL terms):

  loss_image    = F.l1_loss(decoded, target)          # pixel
  loss_edge     = edge_loss(decoded, target)
  loss_boundary = boundary_loss(decoded, target)
  loss_haar     = haar_loss(decoded, target)

  total = loss.noise * loss_noise
        + loss.image * loss_image
        + loss.edge * loss_edge
        + loss.boundary * loss_boundary
        + loss.haar * loss_haar

Those functions live in 013代码/src/ebsd_feedback/losses.py and are called from
013代码/src/ebsd_feedback/training/diffusion.py. The yaml that filled
config.loss.image/edge/boundary/haar (06_配置文件/03_基础条件扩散.yaml) was not
copied into this tree, and diffusion.py does not hardcode the four numbers.
The only numeric coefficients in the copied source for the pixel, edge,
boundary, and Haar terms are VaeLossWeights in losses.py:

  rgb 1.0, edge 0.5, boundary 0.25, haar 0.25

loss.image multiplies a full four-channel L1. The pixel weight is
VaeLossWeights.rgb (1.0). Edge, boundary, and Haar stay on VaeLossWeights.
The 0.75/0.25 channel mix inside boundary_aware_vae_loss is not what
diffusion.py applies, so it is not used. The pixel weight is not 4.0.
VaeLossWeights.ssim (0.25), flatness (0.05), and kl (1e-6) are VAE-only.
simple_vae_loss's edge weight 0.2 is not used either: that path zeros
boundary and Haar, and diffusion.py evaluates all three.

descriptor, unrolled_descriptor, and overlay_alignment stay at the
getattr default of 0 in diffusion.py. This stage does not load a descriptor
head and does not add a local descriptor condition. orientation stays 0.
mechanics_weight is 0 and the mechanics surrogate is not constructed.
diffusion.py itself forces mechanics_weight to 0 unless stage is
diffusion_feedback.

The frozen VAE only decodes the predicted clean latent. The four structure
targets are the original 256x256 window, loaded from 06_可打开的窗口小图.
Those PNGs are the RGB crops export_preview_pngs.py wrote. The fourth channel
is packed from those same pixels by ebsd_feedback.data.image_to_tensor, which
is how 013 turns an RGB window into the four-channel training image. The clean
latent is not decoded to make the target. ID03 images are not read.

The four terms are not applied at every timestep. diffusion.py gates its other
image terms with descriptor_low_noise_fraction, whose code default is 0.3:
active when timesteps < int(training_timesteps * 0.3). Noise loss stays on
every step. A high timestep contributes no pixel, edge, boundary, or Haar loss.

Optimizer settings stay those of train_round1.py: AdamW lr 0.0001,
weight decay 0.0001, betas (0.9, 0.95), gradient clip 1.0, constant lr.
train_diffusion also steps a cosine schedule sized for warmup_steps 1500 and
max_steps 80000. That schedule is the 80000-step run, not a separate
structure-loss optimizer, so it is not turned on here. Adam moments are
restored from the checkpoint the user passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from train_round1 import (
    ADAMW_BETAS,
    CONDITION_DROPOUT,
    COSINE_BETA_OFFSET,
    DRAWS_PER_ALLOY,
    EMA_DECAY,
    GRADIENT_CLIP,
    HOLDOUT,
    LATENT_SCALE,
    LEARNING_RATE,
    LOSS_NOISE,
    MIN_SNR_GAMMA,
    N_ALLOYS,
    SEED_BASE,
    TRAINING_TIMESTEPS,
    WEIGHT_DECAY,
    ExponentialMovingAverage,
    LatentStore,
    _import_unet,
    _null,
    _val_noise,
    build_model,
    condition_matrix,
    crop_batch,
    epoch_order,
    group_crops,
    noise_loss,
    project_root,
    read_table,
)

MECHANICS_LOSS_WEIGHT = 0.0
LOSS_DESCRIPTOR = 0.0
LOSS_UNROLLED_DESCRIPTOR = 0.0
LOSS_OVERLAY = 0.0
LOSS_ORIENTATION = 0.0
ROUND1_EPOCHS = 200
MAX_EXTRA_EPOCHS = 200
FORBIDDEN_START_EPOCHS = (400, 600)
FORBIDDEN_CHECKPOINT_NAMES = ("checkpoint_epoch400.pt", "checkpoint_epoch600.pt")
# diffusion.py: getattr(config.loss, "descriptor_low_noise_fraction", 0.3)
# Overlay and descriptor terms use timesteps < int(timesteps * fraction).
LOW_NOISE_FRACTION = 0.3
# Existing spot-check folders. A finished run writes a different directory.
RESERVED_SAMPLE_DIRS = ("ID03抽查", "ID03抽查_第二轮", "ID03抽查_epoch600")
# A new log, so the earlier train_round2_loss.csv on the server stays intact.
LOSS_LOG_NAME = "train_round2_原图低噪声.csv"
CONFIG_NAME = "第二轮原图低噪声配置.json"


def _import_structure():
    pkg = Path(__file__).resolve().parent / "013代码" / "src"
    if not pkg.is_dir():
        raise FileNotFoundError(f"缺少 013 源码: {pkg}")
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from ebsd_feedback.data import image_to_tensor
    from ebsd_feedback.losses import VaeLossWeights, boundary_loss, edge_loss, haar_loss
    from ebsd_feedback.models.vae import BoundaryAwareVAE
    return VaeLossWeights, boundary_loss, edge_loss, haar_loss, BoundaryAwareVAE, image_to_tensor


VaeLossWeights, boundary_loss, edge_loss, haar_loss, BoundaryAwareVAE, image_to_tensor = _import_structure()
_STRUCTURE_SOURCE = VaeLossWeights()
# Full four-channel L1. Weight is VaeLossWeights.rgb, not the earlier 4.0 trial.
LOSS_IMAGE = float(_STRUCTURE_SOURCE.rgb)
LOSS_EDGE = float(_STRUCTURE_SOURCE.edge)
LOSS_BOUNDARY = float(_STRUCTURE_SOURCE.boundary)
LOSS_HAAR = float(_STRUCTURE_SOURCE.haar)


def structure_loss_weights() -> dict[str, float]:
    return {
        "image": LOSS_IMAGE,
        "edge": LOSS_EDGE,
        "boundary": LOSS_BOUNDARY,
        "haar": LOSS_HAAR,
    }


def image_structure_losses(decoded: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Same four calls as training/diffusion.py. target is the original window."""
    return {
        "image": F.l1_loss(decoded, target),
        "edge": edge_loss(decoded, target),
        "boundary": boundary_loss(decoded, target),
        "haar": haar_loss(decoded, target),
    }


def low_noise_mask(
    timesteps: torch.Tensor,
    total_timesteps: int,
    fraction: float = LOW_NOISE_FRACTION,
) -> torch.Tensor:
    """True where t is in the low-noise part of the schedule.

    Same comparison diffusion.py uses for overlay and descriptor terms:
    timesteps < int(training_timesteps * low_noise_fraction).
    """
    fraction = min(max(float(fraction), 0.0), 1.0)
    threshold = int(total_timesteps * fraction)
    if threshold <= 0:
        return torch.zeros_like(timesteps, dtype=torch.bool)
    return timesteps < threshold


def structure_losses_for_timesteps(
    decoded: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    total_timesteps: int,
) -> dict[str, torch.Tensor]:
    """Pixel, edge, boundary, and Haar only on low-noise rows. Others contribute 0."""
    active = low_noise_mask(timesteps, total_timesteps, LOW_NOISE_FRACTION)
    if not bool(active.any()):
        zero = decoded.sum() * 0.0
        return {name: zero for name in ("image", "edge", "boundary", "haar")}
    return image_structure_losses(decoded[active], target[active])


def window_png_path(window_dir: Path, row: dict[str, str]) -> Path:
    alloy = row["alloy_id"]
    if alloy == HOLDOUT:
        raise RuntimeError("训练不读 ID03 的窗口图")
    name = f"{alloy}_{row['view']}_y{int(row['pixel_y'])}_x{int(row['pixel_x'])}.png"
    return window_dir / alloy / name


def load_original_window(path: Path) -> torch.Tensor:
    """RGB PNG plus the fourth channel 013 packs from those same pixels."""
    if HOLDOUT in path.parts:
        raise RuntimeError(f"不读 ID03 的图: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"缺少训练窗口原图 {path}")
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        tensor = image_to_tensor(rgb)
    if tensor.shape != (4, 256, 256):
        raise RuntimeError(f"{path.name} 应为 4x256x256，实际 {tuple(tensor.shape)}")
    return tensor


def load_original_batch(window_dir: Path, rows: list[dict[str, str]]) -> torch.Tensor:
    return torch.stack([load_original_window(window_png_path(window_dir, row)) for row in rows])


def diffusion_structure_total(
    loss_noise: torch.Tensor,
    parts: dict[str, torch.Tensor],
) -> torch.Tensor:
    if MECHANICS_LOSS_WEIGHT != 0.0:
        raise RuntimeError("第二轮 MECHANICS_LOSS_WEIGHT 必须是 0，不跑力学代理。")
    if LOSS_DESCRIPTOR != 0.0 or LOSS_UNROLLED_DESCRIPTOR != 0.0 or LOSS_OVERLAY != 0.0 or LOSS_ORIENTATION != 0.0:
        raise RuntimeError("描述符、展开描述符、叠加、取向这一轮权重必须是 0。")
    expected = {"image", "edge", "boundary", "haar"}
    if set(parts) != expected:
        raise RuntimeError(f"结构损失只算像素、边缘、晶界、Haar，实际 {sorted(parts)}")
    return (
        LOSS_NOISE * loss_noise
        + LOSS_IMAGE * parts["image"]
        + LOSS_EDGE * parts["edge"]
        + LOSS_BOUNDARY * parts["boundary"]
        + LOSS_HAAR * parts["haar"]
    )


def resolve_epoch_range(start_epoch: int, extra_epochs: int) -> tuple[int, int]:
    """Start at the round-1 noise checkpoint and add at most 200 epochs."""
    if extra_epochs < 1 or extra_epochs > MAX_EXTRA_EPOCHS:
        raise RuntimeError("这一段最多再训 200 轮，不要退回 80000 步")
    if start_epoch in FORBIDDEN_START_EPOCHS:
        raise RuntimeError(
            "不要从第 400 轮或第 600 轮继续。把 --checkpoint 指到第一轮噪声断点，第 200 轮。"
        )
    if start_epoch != ROUND1_EPOCHS:
        raise RuntimeError(
            f"这一次从第一轮噪声断点第 {ROUND1_EPOCHS} 轮开始，实际第 {start_epoch} 轮。"
        )
    return start_epoch, start_epoch + extra_epochs


def assert_round1_checkpoint(path: Path, start_epoch: int) -> None:
    if path.name in FORBIDDEN_CHECKPOINT_NAMES or start_epoch in FORBIDDEN_START_EPOCHS:
        raise RuntimeError(
            "不要从 checkpoint_epoch400.pt 或 checkpoint_epoch600.pt 继续。"
            "把 --checkpoint 指到 04_训练日志/checkpoint_epoch200.pt。"
        )
    if start_epoch != ROUND1_EPOCHS:
        raise RuntimeError(
            f"载入的断点是第 {start_epoch} 轮。这一次要用第一轮噪声断点第 {ROUND1_EPOCHS} 轮。"
        )


def writes_checkpoints(smoke: bool) -> bool:
    return not smoke


def writes_id03_images(smoke: bool) -> bool:
    return not smoke


def heldout_sample_dir(log_dir: Path, end_epoch: int) -> Path:
    """A new folder under the log directory. Reserved names are never reused."""
    stem = f"ID03抽查_epoch{end_epoch:03d}"
    candidate = log_dir / stem
    suffix = 0
    while candidate.name in RESERVED_SAMPLE_DIRS or candidate.exists():
        suffix += 1
        candidate = log_dir / f"{stem}_{suffix}"
    return candidate


def check_batch_size(batch_size: int) -> None:
    if batch_size not in (8, 16):
        raise RuntimeError("batch 只用 16。显存不够改成 8。不要改成别的数。")


def load_frozen_vae(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"找不到冻结的 VAE: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    scale = float(state["latent_scale"])
    if abs(scale - LATENT_SCALE) > 1e-12:
        raise RuntimeError(f"VAE latent_scale {scale} != {LATENT_SCALE}")
    model = BoundaryAwareVAE(**state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    return model


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到断点 {path}。断点不在 git 里。把 --checkpoint 指到服务器 04_训练日志/ 下要接着训的那一份。"
        )
    return torch.load(path, map_location=device, weights_only=False)


def _check_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        if float(group["lr"]) != LEARNING_RATE:
            raise RuntimeError(f"断点学习率 {group['lr']} 不是 {LEARNING_RATE}")
        if float(group["weight_decay"]) != WEIGHT_DECAY:
            raise RuntimeError(f"断点 weight_decay {group['weight_decay']} 不是 {WEIGHT_DECAY}")
        betas = tuple(float(value) for value in group["betas"])
        if betas != ADAMW_BETAS:
            raise RuntimeError(f"断点 AdamW betas {betas} 不是 {ADAMW_BETAS}")


def _restore_ema(ema: ExponentialMovingAverage, saved: dict, device: torch.device) -> None:
    if float(saved["decay"]) != EMA_DECAY:
        raise RuntimeError(f"断点 EMA decay {saved['decay']} 不是 {EMA_DECAY}")
    missing = [name for name in ema.shadow if name not in saved["shadow"]]
    extra = [name for name in saved["shadow"] if name not in ema.shadow]
    if missing or extra:
        raise RuntimeError(f"EMA 参数对不上，缺 {missing[:3]} 多 {extra[:3]}")
    for name, value in saved["shadow"].items():
        ema.shadow[name].copy_(value.to(device=device, dtype=ema.shadow[name].dtype))


def save_checkpoint(path: Path, model, ema, optimizer, epoch: int, step: int, values: torch.Tensor, resumed_from: str) -> None:
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
            "round": 2,
            "resumed_from": resumed_from,
            "loss_noise_weight": LOSS_NOISE,
            "loss_image_weight": LOSS_IMAGE,
            "loss_edge_weight": LOSS_EDGE,
            "loss_boundary_weight": LOSS_BOUNDARY,
            "loss_haar_weight": LOSS_HAAR,
            "loss_descriptor_weight": LOSS_DESCRIPTOR,
            "loss_unrolled_descriptor_weight": LOSS_UNROLLED_DESCRIPTOR,
            "loss_overlay_weight": LOSS_OVERLAY,
            "low_noise_fraction": LOW_NOISE_FRACTION,
            "pixel_target": "original_window_png",
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    root = project_root()
    table_dir = root / "01_窗口清单_每个窗口的位置和描述符"
    parser = argparse.ArgumentParser(description="016 fold-1 patch diffusion, round 2 structure loss")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Round-1 noise checkpoint passed by the user. Not stored in git. On the server: 04_训练日志/checkpoint_epoch200.pt. Do not pass checkpoint_epoch400.pt or checkpoint_epoch600.pt.",
    )
    parser.add_argument("--latent-dir", type=Path, default=root / "02_整图潜变量_八个视角")
    parser.add_argument(
        "--window-dir",
        type=Path,
        default=root / "06_可打开的窗口小图",
        help="256x256 training-window PNGs. Pixel, edge, boundary, and Haar use these pixels.",
    )
    parser.add_argument("--crops", type=Path, default=table_dir / "crops.csv")
    parser.add_argument("--val-strip", type=Path, default=table_dir / "val_strip.csv")
    parser.add_argument("--conditions", type=Path, default=table_dir / "conditions.csv")
    parser.add_argument(
        "--vae",
        type=Path,
        default=root / "07_013各折已有权重" / "ID03" / "01_边界感知VAE" / "VAE_最终模型.pt",
    )
    parser.add_argument("--log-dir", type=Path, default=root / "04_训练日志")
    parser.add_argument("--epochs", type=int, default=MAX_EXTRA_EPOCHS, help="Extra epochs after the loaded checkpoint, at most 200")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One optimizer step from the checkpoint, then exit. Writes no checkpoint and no ID03 image",
    )
    return parser.parse_args()


def _autocast(precision: str):
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return _null()


def main() -> None:
    args = parse_args()
    if MECHANICS_LOSS_WEIGHT != 0.0:
        raise RuntimeError("第二轮 MECHANICS_LOSS_WEIGHT 必须是 0。")
    check_batch_size(args.batch_size)
    if args.epochs > MAX_EXTRA_EPOCHS:
        raise RuntimeError("第二轮最多再训 200 轮，不要退回 80000 步")
    weights = structure_loss_weights()
    if list(weights) != ["image", "edge", "boundary", "haar"]:
        raise RuntimeError(f"结构损失项不对: {list(weights)}")
    if weights["image"] != float(_STRUCTURE_SOURCE.rgb):
        raise RuntimeError("四通道像素 L1 权重必须是 VaeLossWeights.rgb，也就是 1.0")
    if not args.window_dir.is_dir():
        raise FileNotFoundError(f"缺少训练窗口原图目录 {args.window_dir}")
    if (weights["edge"], weights["boundary"], weights["haar"]) != (
        float(_STRUCTURE_SOURCE.edge),
        float(_STRUCTURE_SOURCE.boundary),
        float(_STRUCTURE_SOURCE.haar),
    ):
        raise RuntimeError("边缘、晶界、Haar 必须保持 VaeLossWeights，不能跟着像素项一起改")
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
    if any(row["alloy_id"] == HOLDOUT for row in crops):
        raise RuntimeError("crops.csv 里不该有 ID03")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "bf16" if device.type == "cuda" else "fp32"
    state = _load_checkpoint(args.checkpoint, device)
    start_epoch = int(state["epoch"])
    assert_round1_checkpoint(args.checkpoint, start_epoch)
    start_epoch, end_epoch = resolve_epoch_range(start_epoch, args.epochs)
    if float(state.get("mechanics_loss_weight", 0.0)) != 0.0:
        raise RuntimeError("载入的断点 mechanics_loss_weight 不是 0")
    if abs(float(state.get("latent_scale", LATENT_SCALE)) - LATENT_SCALE) > 1e-12:
        raise RuntimeError("断点 latent_scale 和这一折不一致")
    saved_mean = state["condition_mean"].detach().float().cpu()
    current_mean = cond_values.mean(0).float().cpu()
    if not torch.allclose(saved_mean, current_mean, atol=1e-5, rtol=1e-5):
        raise RuntimeError("断点里的条件均值和 conditions.csv 不一致")
    print(
        f"device={device.type} precision={precision} batch={args.batch_size} "
        f"resume_epoch={start_epoch} through_epoch={end_epoch} checkpoint={args.checkpoint} "
        f"lr={LEARNING_RATE} betas={ADAMW_BETAS} "
        f"loss_image={LOSS_IMAGE} loss_edge={LOSS_EDGE} loss_boundary={LOSS_BOUNDARY} loss_haar={LOSS_HAAR} "
        f"low_noise_fraction={LOW_NOISE_FRACTION} window_dir={args.window_dir} "
        f"mechanics_weight={MECHANICS_LOSS_WEIGHT}",
        flush=True,
    )
    store = LatentStore(args.latent_dir)
    vae = load_frozen_vae(args.vae, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model = build_model(ConditionalLatentUNet, cond_values, device)
    model.load_state_dict(state["model"])
    schedule = DiffusionSchedule(int(state.get("training_timesteps", TRAINING_TIMESTEPS))).to(device)
    ema = ExponentialMovingAverage(model, EMA_DECAY)
    _restore_ema(ema, state["ema"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAMW_BETAS,
        fused=device.type == "cuda",
    )
    optimizer.load_state_dict(state["optimizer"])
    _check_optimizer(optimizer)
    step = int(state["step"])
    if args.smoke:
        if writes_checkpoints(True) or writes_id03_images(True):
            raise RuntimeError("smoke 不写断点，也不写 ID03 图")
        _smoke_step(args, model, schedule, vae, optimizer, ema, store, crops, groups, cond_map, args.window_dir, device, precision, step)
        return
    val_rows = [row for row in read_table(args.val_strip) if row["alloy_id"] != HOLDOUT]
    if not val_rows:
        raise RuntimeError("val_strip.csv 没有训练合金的右侧窗口")
    steps_per_epoch = (N_ALLOYS * DRAWS_PER_ALLOY) // args.batch_size
    print(
        f"steps_per_epoch={steps_per_epoch} extra_epochs={end_epoch - start_epoch} "
        f"resume_step={step}",
        flush=True,
    )
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / CONFIG_NAME).write_text(
        json.dumps(_config_record(args, precision, steps_per_epoch, start_epoch, end_epoch), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_path = args.log_dir / LOSS_LOG_NAME
    fields = [
        "epoch", "step", "loss_total", "loss_noise", "loss_image", "loss_edge",
        "loss_boundary", "loss_haar", "loss_mechanics", "val_noise",
    ]
    if start_epoch == ROUND1_EPOCHS or not log_path.is_file():
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    torch.manual_seed(args.seed)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        rng = np.random.default_rng([args.seed, epoch])
        order = epoch_order(groups, rng, args.batch_size)
        model.train()
        epoch_total = []
        for start in range(0, len(order), args.batch_size):
            batch_rows = [crops[int(i)] for i in order[start:start + args.batch_size]]
            step, row = _optimizer_step(
                model, schedule, vae, optimizer, ema, store, batch_rows, cond_map, args.window_dir, device, precision, step,
            )
            epoch_total.append(float(row["loss_total"]))
            is_last = start + args.batch_size >= len(order)
            val_noise = ""
            if is_last:
                val_noise = f"{_val_noise(model, schedule, store, val_rows, cond_map, device, args.batch_size):.8f}"
                model.train()
            row["epoch"] = epoch
            row["val_noise"] = val_noise
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(
            f"epoch {epoch} train_total {float(np.mean(epoch_total)):.4f} val_noise {val_noise}",
            flush=True,
        )
        if writes_checkpoints(args.smoke) and (epoch % 50 == 0 or epoch == end_epoch):
            save_checkpoint(
                args.log_dir / f"checkpoint_epoch{epoch:03d}.pt",
                model, ema, optimizer, epoch, step, cond_values, str(args.checkpoint),
            )
            print(f"saved checkpoint_epoch{epoch:03d}.pt", flush=True)
    if writes_id03_images(args.smoke):
        final_checkpoint = args.log_dir / f"checkpoint_epoch{end_epoch:03d}.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError(f"训练结束但没有写出 {final_checkpoint.name}")
        sample_dir = heldout_sample_dir(args.log_dir, end_epoch)
        write_heldout_samples(final_checkpoint, args.vae, sample_dir, args.seed)


def _optimizer_step(model, schedule, vae, optimizer, ema, store, batch_rows, cond_map, window_dir, device, precision, step):
    latent = crop_batch(store, batch_rows).to(device, non_blocking=True)
    condition = torch.stack([cond_map[row["alloy_id"]] for row in batch_rows], 0).to(device, non_blocking=True)
    target = load_original_batch(window_dir, batch_rows).to(device, non_blocking=True)
    noise = torch.randn_like(latent)
    timesteps = torch.randint(0, schedule.timesteps, (latent.shape[0],), device=device)
    noisy = schedule.add_noise(latent, noise, timesteps)
    drop = torch.rand(latent.shape[0], device=device) < CONDITION_DROPOUT
    optimizer.zero_grad(set_to_none=True)
    with _autocast(precision):
        predicted = model(noisy, timesteps, condition, drop)
        loss_noise = noise_loss(schedule, predicted, noise, timesteps)
        predicted_clean = schedule.predict_clean(noisy, predicted, timesteps).clamp(-5.0, 5.0)
        # diffusion.py gates overlay and descriptor with timesteps < int(schedule.timesteps * 0.3).
        active = low_noise_mask(timesteps, int(schedule.timesteps))
        if bool(active.any()):
            decoded = vae.decode(predicted_clean[active] / LATENT_SCALE)
            parts = image_structure_losses(decoded, target[active].to(dtype=decoded.dtype))
        else:
            zero = predicted_clean.sum() * 0.0
            parts = {name: zero for name in ("image", "edge", "boundary", "haar")}
    total = diffusion_structure_total(loss_noise, parts)
    if not bool(torch.isfinite(total)):
        raise RuntimeError("第二轮损失出现 NaN/Inf")
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    ema.update(model)
    step += 1
    row = {
        "epoch": "",
        "step": step,
        "loss_total": f"{float(total.detach()):.8f}",
        "loss_noise": f"{float(loss_noise.detach()):.8f}",
        "loss_image": f"{float(parts['image'].detach()):.8f}",
        "loss_edge": f"{float(parts['edge'].detach()):.8f}",
        "loss_boundary": f"{float(parts['boundary'].detach()):.8f}",
        "loss_haar": f"{float(parts['haar'].detach()):.8f}",
        "loss_mechanics": "0.0",
        "val_noise": "",
    }
    return step, row


def write_heldout_samples(checkpoint: Path, vae_path: Path, out_dir: Path, seed: int) -> Path:
    """Same four DDIM windows as sample_id03.py --mode sample.

    Uses the holdout condition row only. Does not read an ID03 image.
    """
    from sample_id03 import GUIDANCE_SCALE, SAMPLING_STEPS, sample_id03 as sample_holdout
    namespace = argparse.Namespace(
        checkpoint=checkpoint,
        vae=vae_path,
        conditions=project_root() / "01_窗口清单_每个窗口的位置和描述符" / "conditions_ID03_holdout.csv",
        out_dir=out_dir,
        count=4,
        steps=SAMPLING_STEPS,
        guidance=GUIDANCE_SCALE,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_holdout(namespace, device)
    print(f"id03_sample_dir={out_dir}", flush=True)
    return out_dir


def _smoke_step(args, model, schedule, vae, optimizer, ema, store, crops, groups, cond_map, window_dir, device, precision, step) -> None:
    rng = np.random.default_rng([args.seed, int(args.epochs)])
    order = epoch_order(groups, rng, args.batch_size)
    batch_rows = [crops[int(i)] for i in order[: args.batch_size]]
    model.train()
    new_step, row = _optimizer_step(
        model, schedule, vae, optimizer, ema, store, batch_rows, cond_map, window_dir, device, precision, step,
    )
    print(
        f"smoke_ok step={new_step} loss_total={row['loss_total']} loss_noise={row['loss_noise']} "
        f"loss_image={row['loss_image']} loss_edge={row['loss_edge']} "
        f"loss_boundary={row['loss_boundary']} loss_haar={row['loss_haar']} "
        f"loss_mechanics={row['loss_mechanics']} window={batch_rows[0]['alloy_id']}_{batch_rows[0]['view']}",
        flush=True,
    )


def _config_record(args, precision: str, steps_per_epoch: int, start_epoch: int, end_epoch: int) -> dict:
    source = VaeLossWeights()
    return {
        "round": 2,
        "holdout": HOLDOUT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_note": "passed by the user; not in git. Resume 04_训练日志/checkpoint_epoch200.pt. Leave checkpoint_epoch400.pt and checkpoint_epoch600.pt unused.",
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "extra_epochs": end_epoch - start_epoch,
        "extra_epochs_cap": MAX_EXTRA_EPOCHS,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "adamw_betas": list(ADAMW_BETAS),
        "warmup_applied": False,
        "cosine_schedule_in_013_for_80000_steps": {
            "applied": False,
            "why": "train_diffusion sizes warmup_steps 1500 and max_steps 80000. Round 1 already left that off. The structure terms use the same AdamW, not a second optimizer.",
        },
        "gradient_clip": GRADIENT_CLIP,
        "training_timesteps": TRAINING_TIMESTEPS,
        "min_snr_gamma": MIN_SNR_GAMMA,
        "condition_dropout": CONDITION_DROPOUT,
        "ema_decay": EMA_DECAY,
        "precision": precision,
        "vae_frozen": True,
        "window_dir": str(args.window_dir),
        "pixel_target": "original 256x256 PNG under 06_可打开的窗口小图, packed by image_to_tensor. The stored clean latent is not decoded as the target. ID03 images are not read.",
        "low_noise_fraction": LOW_NOISE_FRACTION,
        "structure_timestep_gate": "pixel, edge, boundary, and Haar only when timesteps < int(training_timesteps * low_noise_fraction). Noise loss is every step.",
        "low_noise_fraction_source": "training/diffusion.py getattr(config.loss, descriptor_low_noise_fraction, 0.3), the gate on overlay and descriptor",
        "mechanics_loss_weight": MECHANICS_LOSS_WEIGHT,
        "mechanics_surrogate": "not constructed",
        "loss_weights": structure_loss_weights(),
        "loss_weight_source": "losses.py VaeLossWeights rgb 1.0, edge 0.5, boundary 0.25, haar 0.25. The pixel weight 4.0 trial is not used.",
        "loss_log": LOSS_LOG_NAME,
        "loss_formula_source": "013代码/src/ebsd_feedback/training/diffusion.py image L1, edge_loss, boundary_loss, haar_loss",
        "yaml_not_in_tree": "06_配置文件/03_基础条件扩散.yaml was not copied. diffusion.py reads config.loss.image/edge/boundary/haar and does not literalize them.",
        "vae_terms_not_in_diffusion_total": {
            "ssim": float(source.ssim),
            "flatness": float(source.flatness),
            "kl": float(source.kl),
            "simple_vae_edge_weight_not_used": 0.2,
        },
        "left_at_zero": {
            "descriptor": LOSS_DESCRIPTOR,
            "unrolled_descriptor": LOSS_UNROLLED_DESCRIPTOR,
            "overlay": LOSS_OVERLAY,
            "orientation": LOSS_ORIENTATION,
            "mechanics": MECHANICS_LOSS_WEIGHT,
        },
        "round1_script": "train_round1.py is unchanged and remains noise-only",
    }


if __name__ == "__main__":
    main()
