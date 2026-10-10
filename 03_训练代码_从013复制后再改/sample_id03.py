# -*- coding: utf-8 -*-
"""After training only. Do not import this from train_round1.py.

--mode sample: a few 256 windows for held-out ID03. Does not read ID03 images.
--mode memorization: the memorization check that training does not run.
  A few DDIM windows are decoded and compared with every catalog window.
  Stop the run yourself if max RGB Pearson correlation is near 1 (>= 0.98).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_round1 import (
    COND_COLUMNS,
    HOLDOUT,
    LATENT_SCALE,
    LATENT_WIN,
    TRAINING_TIMESTEPS,
    LatentStore,
    crop_batch,
    project_root,
    read_table,
)
from window_condition import CONDITION_DIM, batch_condition, holdout_condition, holdout_origins

CORR_NEAR_ONE = 0.98
GUIDANCE_SCALE = 2.0
SAMPLING_STEPS = 50


def _import_models():
    pkg = Path(__file__).resolve().parent / "013代码" / "src"
    if not pkg.is_dir():
        raise FileNotFoundError(f"缺少 013 源码: {pkg}")
    sys.path.insert(0, str(pkg))
    from ebsd_feedback.models.diffusion import ConditionalLatentUNet, DiffusionSchedule
    from ebsd_feedback.models.vae import BoundaryAwareVAE
    return ConditionalLatentUNet, DiffusionSchedule, BoundaryAwareVAE


def load_unet(checkpoint: Path, device: torch.device):
    ConditionalLatentUNet, DiffusionSchedule, _ = _import_models()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weight = state["model"]["condition_encoder.network.0.weight"]
    condition_dim = int(weight.shape[1])
    if condition_dim not in (len(COND_COLUMNS), CONDITION_DIM):
        raise RuntimeError(f"断点条件维度 {condition_dim} 不是 {len(COND_COLUMNS)} 也不是 {CONDITION_DIM}")
    model = ConditionalLatentUNet(
        latent_channels=4,
        condition_dim=condition_dim,
        base_channels=128,
        channel_multipliers=(1, 2, 3, 4),
        attention_heads=8,
    ).to(device).eval()
    model.load_state_dict(state["model"])
    copied = model.state_dict()
    for name, value in state["ema"]["shadow"].items():
        if tuple(value.shape) != tuple(copied[name].shape):
            raise RuntimeError(f"EMA {name} 形状 {tuple(value.shape)} 对不上 {tuple(copied[name].shape)}")
        copied[name].copy_(value)
    model.load_state_dict(copied)
    if condition_dim == len(COND_COLUMNS):
        mean = state["condition_mean"].to(device=device, dtype=torch.float32)
        std = state["condition_std"].to(device=device, dtype=torch.float32)
        model.condition_encoder.set_statistics(mean, std.clamp_min(1e-6))
    schedule = DiffusionSchedule(int(state.get("training_timesteps", TRAINING_TIMESTEPS))).to(device)
    return model, schedule, state


def load_vae(path: Path, device: torch.device):
    _, _, BoundaryAwareVAE = _import_models()
    state = torch.load(path, map_location="cpu", weights_only=False)
    scale = float(state["latent_scale"])
    if abs(scale - LATENT_SCALE) > 1e-12:
        raise RuntimeError(f"VAE latent_scale {scale} != {LATENT_SCALE}")
    model = BoundaryAwareVAE(**state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    return model


def save_rgb(image: torch.Tensor, path: Path) -> None:
    rgb = image[:3].detach().float().cpu()
    array = ((rgb + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def _max_rgb_corr(generated: torch.Tensor, real: torch.Tensor) -> float:
    """Pearson correlation of RGB pixels. The mask channel is left out."""
    left = generated[:, :3].float().reshape(generated.shape[0], -1)
    right = real[:, :3].float().reshape(real.shape[0], -1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    left = left / left.norm(dim=1, keepdim=True).clamp_min(1e-8)
    right = right / right.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return float((left @ right.T).max())


@torch.no_grad()
def sample_id03(args, device) -> None:
    model, schedule, _ = load_unet(args.checkpoint, device)
    vae = load_vae(args.vae, device)
    table = read_table(args.conditions)
    rows = [row for row in table if row["alloy_id"] == HOLDOUT]
    if len(rows) != 1:
        raise RuntimeError("需要 conditions_ID03_holdout.csv 里的那一行 ID03，不要用训练表")
    alloy = torch.tensor(
        [float(rows[0][name]) for name in COND_COLUMNS], dtype=torch.float32
    )
    if int(model.condition_encoder.input_mean.numel()) == CONDITION_DIM:
        condition = holdout_condition(rows[0], args.count).to(device)
        origins = holdout_origins(args.count)
        print(
            f"id03_window_origins={origins} "
            f"window_d50=desc_cond_grain_size_median_um condition_dim={CONDITION_DIM}",
            flush=True,
        )
    else:
        condition = alloy.repeat(args.count, 1).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = schedule.ddim_sample(
        model,
        (args.count, 4, LATENT_WIN, LATENT_WIN),
        condition,
        args.steps,
        args.guidance,
        device,
        generator,
    )
    image = vae.decode(latent / LATENT_SCALE).clamp(-1, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        save_rgb(image[index], args.out_dir / f"ID03_窗口_{index + 1:02d}.png")
    (args.out_dir / "说明.json").write_text(json.dumps({
        "holdout": HOLDOUT,
        "count": args.count,
        "seed": args.seed,
        "steps": args.steps,
        "guidance": args.guidance,
        "condition_dim": int(condition.shape[1]),
        "window_origins": [
            {"latent_x": origin_x, "latent_y": origin_y}
            for origin_x, origin_y in holdout_origins(args.count)
        ] if int(condition.shape[1]) == CONDITION_DIM else [],
        "note": "只用于最后看 ID03。不参与做清单，也不拿来调参。没有读取 ID03 的原图。34 维条件时窗口位置是 latent_x 0..128、latent_y 0..48、步长 16 的合法原点，--count 20 取前 20 个不重复位置，超过清单长度再从头循环。D50 和 log_spread 用留出合金条件里的描述符，不是窗口实测。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.count} windows to {args.out_dir}", flush=True)


@torch.no_grad()
def memorization_check(args, device) -> None:
    crops = read_table(args.crops)
    if any(row["alloy_id"] == HOLDOUT for row in crops):
        raise RuntimeError("背图检查只用训练清单，crops.csv 里不该有 ID03")
    cond_rows = read_table(args.train_conditions)
    cond_map = {}
    for row in cond_rows:
        if row["alloy_id"] == HOLDOUT:
            raise RuntimeError("背图检查不要用 ID03 的条件")
        cond_map[row["alloy_id"]] = torch.tensor(
            [float(row[name]) for name in COND_COLUMNS], dtype=torch.float32
        )
    model, schedule, _ = load_unet(args.checkpoint, device)
    vae = load_vae(args.vae, device)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(crops), size=args.count, replace=False)
    picked = [crops[int(i)] for i in pick]
    if int(model.condition_encoder.input_mean.numel()) == CONDITION_DIM:
        condition = batch_condition(picked, cond_map).to(device)
    else:
        condition = torch.stack([cond_map[row["alloy_id"]] for row in picked], 0).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = schedule.ddim_sample(
        model,
        (args.count, 4, LATENT_WIN, LATENT_WIN),
        condition,
        args.steps,
        args.guidance,
        device,
        generator,
    )
    generated = vae.decode(latent / LATENT_SCALE).float()
    store = LatentStore(args.latent_dir)
    best = -1.0
    compared = 0
    for start in range(0, len(crops), args.decode_batch):
        chunk = crops[start:start + args.decode_batch]
        real = vae.decode(crop_batch(store, chunk).to(device) / LATENT_SCALE).float()
        best = max(best, _max_rgb_corr(generated, real))
        compared += len(chunk)
        if compared % 512 == 0 or compared == len(crops):
            print(f"compared {compared}/{len(crops)} max_pixel_corr {best:.4f}", flush=True)
    stop = best >= CORR_NEAR_ONE
    report = {
        "checkpoint": str(args.checkpoint),
        "sampled_windows": args.count,
        "catalog_windows": compared,
        "max_pixel_corr": best,
        "near_one_threshold": CORR_NEAR_ONE,
        "channels": "rgb",
        "stop_recommended": stop,
        "note": "接近 1 表示生成窗口和某张训练窗口几乎一样，应停止。这个检查不在训练循环里，也不读 ID03 原图。",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"背图检查_{args.checkpoint.stem}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"max_pixel_corr {best:.4f} stop_recommended {stop} wrote {out}", flush=True)
    if stop:
        print("生成窗口和清单窗口的像素相关接近 1，请停掉训练。验证损失上升不要拿来停。", flush=True)


def parse_args() -> argparse.Namespace:
    root = project_root()
    table_dir = root / "01_窗口清单_每个窗口的位置和描述符"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sample", "memorization"), default="sample")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, default=root / "07_013各折已有权重" / "ID03" / "01_边界感知VAE" / "VAE_最终模型.pt")
    parser.add_argument("--conditions", type=Path, default=table_dir / "conditions_ID03_holdout.csv")
    parser.add_argument("--train-conditions", type=Path, default=table_dir / "conditions.csv")
    parser.add_argument("--crops", type=Path, default=table_dir / "crops.csv")
    parser.add_argument("--latent-dir", type=Path, default=root / "02_整图潜变量_八个视角")
    parser.add_argument("--out-dir", type=Path, default=root / "04_训练日志" / "ID03抽查")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=SAMPLING_STEPS)
    parser.add_argument(
        "--guidance",
        type=float,
        default=GUIDANCE_SCALE,
        help="DDIM classifier-free guidance. Default 2.0. Comparison settings are 1.5, 2.5, and 3.0.",
    )
    parser.add_argument("--decode-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.mode == "memorization" and args.out_dir == root / "04_训练日志" / "ID03抽查":
        args.out_dir = root / "04_训练日志"
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type} mode={args.mode}", flush=True)
    if args.mode == "sample":
        sample_id03(args, device)
    else:
        memorization_check(args, device)


if __name__ == "__main__":
    main()
