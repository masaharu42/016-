# -*- coding: utf-8 -*-
"""One fixed-seed catalog: 256 training windows per alloy, plus one right-edge val strip per view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from round1_common import (
    COND_COLUMNS, CURVE_TRUE_COLUMNS, DESC_NAMES, FULL_H, FULL_W, HOLDOUT, LATENT_WIN,
    PIXEL_WIN, SEED, apply_ops, desc_parent, load_resized, segment, training_positions,
    truncation_ratio, views_for,
)


def _row(alloy_id, view, y, x, stats, trunc):
    row = {
        "alloy_id": alloy_id,
        "view": view,
        "latent_y": int(y),
        "latent_x": int(x),
        "pixel_y": int(y) * 4,
        "pixel_x": int(x) * 4,
    }
    for name, value in zip(DESC_NAMES, stats):
        row[name] = value
    row["truncation_ratio"] = trunc
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    by_key = {(item["alloy_id"], item["view"]): item for item in index["files"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    manifest["alloy_id"] = manifest["alloy_id"].astype(str)
    train = manifest[manifest["role"] == "train"].sort_values("alloy_id")
    if len(train) != 21:
        raise RuntimeError(f"训练合金不是21行: {len(train)}")
    missing_cols = [c for c in COND_COLUMNS + CURVE_TRUE_COLUMNS if c not in manifest.columns]
    if missing_cols:
        raise RuntimeError(f"折清单缺列: {missing_cols}")
    hold = manifest[manifest["alloy_id"] == HOLDOUT]
    if len(hold) != 1:
        raise RuntimeError("折清单里没有唯一的 ID03 行")
    cond = train[["alloy_id", *COND_COLUMNS]].copy()
    cond.to_csv(args.out_dir / "conditions.csv", index=False)
    hold[["alloy_id", *COND_COLUMNS]].to_csv(args.out_dir / "conditions_ID03_holdout.csv", index=False)
    train[["alloy_id", *CURVE_TRUE_COLUMNS]].to_csv(
        args.out_dir / "curve_true_不进网络条件.csv", index=False
    )
    print(f"conditions {len(cond)} x {len(COND_COLUMNS)}", flush=True)

    crops = []
    strips = []
    rngs = np.random.SeedSequence(args.seed).spawn(len(train))
    images = {p.name.split("_", 1)[0]: p for p in args.image_dir.glob("ID*_真实EBSD_IPF加晶界.bmp")}
    for alloy_id, child in zip(train["alloy_id"].tolist(), rngs):
        path = images.get(alloy_id)
        if path is None:
            raise FileNotFoundError(alloy_id)
        rgb_i, mask_i = load_resized(path)
        pool = []
        cache = {}
        for view, ops in views_for(alloy_id):
            meta = by_key[(alloy_id, view)]
            shape = tuple(meta["shape"])
            latent_h, latent_w = shape[1], shape[2]
            rgb_v = np.asarray(apply_ops(rgb_i, ops).convert("RGB"), dtype=np.uint8)
            if rgb_v.shape[0] != latent_h * 4 or rgb_v.shape[1] != latent_w * 4:
                raise RuntimeError(f"{alloy_id} {view} image {rgb_v.shape} vs latent {shape}")
            lab, table, areas = segment(rgb_v)
            cache[view] = (lab, table, areas, latent_h, latent_w)
            positions = training_positions(latent_h, latent_w) if meta["crop_eligible"] else []
            for y, x in positions:
                pool.append((view, y, x))
            sy, sx = 0, latent_w - LATENT_WIN
            stats = desc_parent(lab, table, sy * 4, sx * 4, PIXEL_WIN, PIXEL_WIN)
            trunc = truncation_ratio(lab, areas, sy * 4, sx * 4, PIXEL_WIN, PIXEL_WIN)
            strips.append(_row(alloy_id, view, sy, sx, stats, trunc))
        if len(pool) < 256:
            raise RuntimeError(f"{alloy_id} 合法窗口只有 {len(pool)} 个")
        rng = np.random.default_rng(child)
        pick = rng.choice(len(pool), size=256, replace=False)
        for k in pick.tolist():
            view, y, x = pool[k]
            lab, table, areas, _, _ = cache[view]
            stats = desc_parent(lab, table, y * 4, x * 4, PIXEL_WIN, PIXEL_WIN)
            trunc = truncation_ratio(lab, areas, y * 4, x * 4, PIXEL_WIN, PIXEL_WIN)
            crops.append(_row(alloy_id, view, y, x, stats, trunc))
        rgb_i.close()
        mask_i.close()
        print(f"{alloy_id} pool={len(pool)} picked=256", flush=True)
    crop_df = pd.DataFrame(crops)
    # Stable order: the draw is random, the file is sorted so a later shuffle is explicit.
    crop_df = crop_df.sort_values(["alloy_id", "view", "latent_y", "latent_x"]).reset_index(drop=True)
    if len(crop_df) != 21 * 256:
        raise RuntimeError(len(crop_df))
    # Reject any training window that touches the right strip.
    bad = crop_df[crop_df["latent_x"] + LATENT_WIN > 128]
    if len(bad):
        raise RuntimeError(f"训练窗口进入了右侧验证带: {len(bad)}")
    crop_df.to_csv(args.out_dir / "crops.csv", index=False)
    strip_df = pd.DataFrame(strips).sort_values(["alloy_id", "view"]).reset_index(drop=True)
    strip_df.to_csv(args.out_dir / "val_strip.csv", index=False)
    meta = {
        "seed": args.seed,
        "generator": "numpy.random.SeedSequence.spawn per alloy, then choice without replacement",
        "windows_per_alloy": 256,
        "crop_rows": int(len(crop_df)),
        "val_strip_rows": int(len(strip_df)),
        "descriptors_are_model_inputs": False,
        "parent_grain": "centroid inside window, size is the full grain; area weights use in-window pixels; border is not a boundary",
        "right_strip": "latent_x = width-64, latent_y = 0, one 64x64 window per encoded view",
        "portrait_views_in_crops": False,
        "resized_hw": [FULL_H, FULL_W],
    }
    (args.out_dir / "crops_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"crops {len(crop_df)} strips {len(strip_df)}", flush=True)


if __name__ == "__main__":
    main()
