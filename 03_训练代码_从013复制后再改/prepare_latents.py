# -*- coding: utf-8 -*-
"""Encode each training alloy at 8 D4 orientations (ID04/ID17: 4). Never flips an encoded latent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from round1_common import (
    FULL_H, FULL_W, HOLDOUT, LATENT_SCALE, NO90, apply_ops, is_portrait_ops,
    load_resized, load_vae, to_tensor, views_for,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=str,
                        default=r"D:\013_IPF_GB混合方案项目\03_图像数据\IPF_GB")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"encode device={device.type} scale={LATENT_SCALE}", flush=True)
    vae, scale = load_vae(args.vae, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    images = sorted(args.image_dir.glob("ID*_真实EBSD_IPF加晶界.bmp"))
    alloy_ids = []
    for path in images:
        alloy_id = path.name.split("_", 1)[0]
        if alloy_id == HOLDOUT:
            continue
        alloy_ids.append(alloy_id)
        rgb_i, mask_i = load_resized(path)
        source = str(Path(args.source_root) / alloy_id / path.name)
        for view, ops in views_for(alloy_id):
            rgb_v = apply_ops(rgb_i, ops)
            mask_v = apply_ops(mask_i, ops)
            tensor = to_tensor(rgb_v, mask_v).unsqueeze(0).to(device)
            with torch.no_grad():
                latent, _, _ = vae.encode(tensor, sample=False)
            latent = (latent * scale).squeeze(0).detach().float().cpu().numpy()
            if latent.dtype != np.float32:
                latent = latent.astype(np.float32)
            name = f"{alloy_id}_{view}.npy"
            np.save(args.out_dir / name, latent)
            eligible = (not is_portrait_ops(ops)) and latent.shape == (4, 112, 192)
            files.append({
                "alloy_id": alloy_id,
                "view": view,
                "ops": list(ops),
                "shape": list(latent.shape),
                "scale": scale,
                "scale_multiplied_into_file": True,
                "source_image": source,
                "file": name,
                "crop_eligible": bool(eligible),
                "reason_if_not_eligible": (
                    "" if eligible else
                    "90度后画布变为448×768，潜变量4×192×112。右侧再留256像素后，剩不到一个256窗口。"
                ),
            })
            print(f"{name} {tuple(latent.shape)}", flush=True)
        rgb_i.close()
        mask_i.close()
    missing = [f"ID{i:02d}" for i in range(1, 23) if f"ID{i:02d}" not in alloy_ids and f"ID{i:02d}" != HOLDOUT]
    index = {
        "holdout": HOLDOUT,
        "latent_scale": scale,
        "scale_multiplied_into_file": True,
        "resized_hw": [FULL_H, FULL_W],
        "device": device.type,
        "no90_alloys": sorted(NO90),
        "missing_images": missing,
        "file_count": len(files),
        "note": (
            "普通合金8个视角，ID04和ID17不做90度，只有4个。"
            "先把图缩到768×448，再对图像做旋转或翻转，然后分别编码。"
            "不会把已经编好的潜变量再翻转。"
            "不做90度的视角是4×112×192。"
            "做了90度的视角是4×192×112，因为768×448转90度后是448×768，VAE下采样4倍。"
            "这些90度文件照样保存，但不进入256条训练窗口：右侧256像素验证带留出后，宽度不够再放一个256窗口。"
        ),
        "files": files,
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(files)} latents, missing={missing}", flush=True)


if __name__ == "__main__":
    main()
