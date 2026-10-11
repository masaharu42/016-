# -*- coding: utf-8 -*-
"""Append a 22-D window and full-view condition to the 28-D alloy vector.

The 22 numbers are inputs, not a loss. An empty or non-finite descriptor is 0
with a presence bit of 0. There is no window-minus-global delta.

Layout (CONDITION_DIM = 28 + 22 = 50):

  0-1    window_x, window_y          latent_x/128, latent_y/48
  2-11   window_{DESC_NAMES}         value, present, five descriptors
  12-21  global_{DESC_NAMES}         same five, measured on the whole view
"""
from __future__ import annotations

import math

import torch

from round1_common import DESC_NAMES
from train_round1 import COND_COLUMNS

ALLOY_CONDITION_DIM = len(COND_COLUMNS)
# Landscape latent is 112 x 192. A 64 window's origin is at most x=128, y=48.
WINDOW_X_SPAN = 128.0
WINDOW_Y_SPAN = 48.0
# Same five names as crops.csv. Holdout fills both blocks from desc_cond_*.
HOLDOUT_DESC_SOURCES = {
    "D50": "desc_cond_grain_size_median_um",
    "log_spread": "desc_cond_grain_size_log_spread",
    "area_weighted_aspect": "desc_cond_area_weighted_aspect_ratio",
    "coarse_area_fraction": "desc_cond_coarse_grain_area_fraction",
    "boundary_length_density": "desc_cond_boundary_length_density_per_um",
}
if tuple(HOLDOUT_DESC_SOURCES) != tuple(DESC_NAMES):
    raise RuntimeError(f"留出描述符映射和 DESC_NAMES 不一致: {tuple(DESC_NAMES)}")


def _descriptor_feature_names(prefix: str) -> tuple[str, ...]:
    names: list[str] = []
    for desc in DESC_NAMES:
        names.append(f"{prefix}_{desc}")
        names.append(f"{prefix}_{desc}_present")
    return tuple(names)


WINDOW_FEATURE_NAMES = (
    "window_x",
    "window_y",
    *_descriptor_feature_names("window"),
    *_descriptor_feature_names("global"),
)
GLOBAL_COLUMNS = tuple(f"global_{name}" for name in DESC_NAMES)
CONDITION_DIM = ALLOY_CONDITION_DIM + len(WINDOW_FEATURE_NAMES)
if CONDITION_DIM != 50 or len(WINDOW_FEATURE_NAMES) != 22:
    raise RuntimeError(f"条件应为 50 维（28+22），实际 {CONDITION_DIM}")
if any("delta" in name.lower() or "Δ" in name for name in WINDOW_FEATURE_NAMES):
    raise RuntimeError("条件用绝对量，不加 Δ")
# A 64 window fits a 112×192 landscape latent at x 0..128 and y 0..48.
# Step 16 gives 9×4 = 36 distinct origins. The first 20 are a coarser
# spread (x every 32, every y), with the four corners listed first.
_HOLDOUT_X = tuple(range(0, 129, 16))
_HOLDOUT_Y = tuple(range(0, 49, 16))
_HOLDOUT_CORNERS = ((0, 0), (64, 0), (0, 48), (64, 48))
_HOLDOUT_SPREAD_X = (0, 32, 64, 96, 128)


def _build_holdout_origins() -> tuple[tuple[int, int], ...]:
    spread = [(x, y) for x in _HOLDOUT_SPREAD_X for y in _HOLDOUT_Y]
    first = list(_HOLDOUT_CORNERS) + [origin for origin in spread if origin not in _HOLDOUT_CORNERS]
    rest = [(x, y) for x in _HOLDOUT_X for y in _HOLDOUT_Y if (x, y) not in first]
    origins = tuple(first + rest)
    if len(origins) < 20 or len(set(origins)) != len(origins):
        raise RuntimeError(f"ID03 抽样位置不够或有重复: {len(origins)}")
    return origins


HOLDOUT_SAMPLE_ORIGINS = _build_holdout_origins()


def holdout_origins(count: int) -> list[tuple[int, int]]:
    """First `count` sample origins. Past the list, cycle from the start."""
    if count < 1:
        raise RuntimeError("ID03 抽样数量至少是 1")
    return [HOLDOUT_SAMPLE_ORIGINS[index % len(HOLDOUT_SAMPLE_ORIGINS)] for index in range(count)]


def finite_field(text: object) -> tuple[float, float]:
    """Return (value, 1) when the cell is a finite number, else (0, 0)."""
    if text is None:
        return 0.0, 0.0
    raw = str(text).strip()
    if raw == "":
        return 0.0, 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0, 0.0
    if not math.isfinite(value):
        return 0.0, 0.0
    return value, 1.0


def require_catalog_columns(rows: list[dict[str, str]]) -> None:
    """Training and val tables must already carry window and global descriptors."""
    if not rows:
        raise RuntimeError("清单是空的")
    missing = [name for name in (*DESC_NAMES, *GLOBAL_COLUMNS) if name not in rows[0]]
    if missing:
        raise RuntimeError(
            f"清单缺少描述符列 {missing}。"
            "整图列 global_* 由 prepare_crops.py 在每个视角上写一次。"
            "重跑前会把还没有这些列的 crops.csv 备份成 crops_备份_无整图描述符.csv。"
        )


def window_features(row: dict[str, str]) -> torch.Tensor:
    """22 numbers: origin, five window descriptors, five full-view descriptors."""
    values = [
        float(row["latent_x"]) / WINDOW_X_SPAN,
        float(row["latent_y"]) / WINDOW_Y_SPAN,
    ]
    for name in DESC_NAMES:
        value, present = finite_field(row.get(name, ""))
        values.append(value)
        values.append(present)
    for name in DESC_NAMES:
        value, present = finite_field(row.get(f"global_{name}", ""))
        values.append(value)
        values.append(present)
    tensor = torch.tensor(values, dtype=torch.float32)
    if int(tensor.numel()) != len(WINDOW_FEATURE_NAMES):
        raise RuntimeError(f"窗口特征应为 {len(WINDOW_FEATURE_NAMES)} 个数，实际 {int(tensor.numel())}")
    return tensor


def _column_std(column: torch.Tensor) -> torch.Tensor:
    if column.numel() < 2:
        return column.new_tensor(1.0)
    deviation = column.std(unbiased=True)
    if not torch.isfinite(deviation) or float(deviation) <= 0.0:
        return column.new_tensor(1.0)
    return deviation


def window_feature_statistics(rows: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std for the encoder. Physical stats use only present rows."""
    stacked = torch.stack([window_features(row) for row in rows])
    mean = torch.zeros(len(WINDOW_FEATURE_NAMES))
    std = torch.ones(len(WINDOW_FEATURE_NAMES))
    for index, name in enumerate(WINDOW_FEATURE_NAMES):
        if name in ("window_x", "window_y") or name.endswith("_present"):
            column = stacked[:, index]
            mean[index] = column.mean()
            std[index] = _column_std(column)
    for index, name in enumerate(WINDOW_FEATURE_NAMES):
        if not name.endswith("_present"):
            continue
        present = stacked[:, index] > 0.5
        if int(present.sum()) >= 2:
            column = stacked[present, index - 1]
            mean[index - 1] = column.mean()
            std[index - 1] = _column_std(column)
    return mean, std.clamp_min(1e-6)


def batch_condition(rows: list[dict[str, str]], alloy_by_id: dict[str, torch.Tensor]) -> torch.Tensor:
    """28 alloy numbers, then the 22 window and global numbers. Raw, not normalized."""
    alloy = torch.stack([alloy_by_id[row["alloy_id"]].detach().float().cpu() for row in rows])
    window = torch.stack([window_features(row) for row in rows])
    if alloy.shape[1] != ALLOY_CONDITION_DIM or window.shape[1] != len(WINDOW_FEATURE_NAMES):
        raise RuntimeError(f"条件拼接维度不对: alloy {tuple(alloy.shape)} window {tuple(window.shape)}")
    return torch.cat([alloy, window], dim=1)


def _holdout_descriptor_row(alloy_row: dict[str, str], origin_x: int, origin_y: int) -> dict[str, str]:
    """Window and global descriptors both come from the holdout desc_cond_* fields.

    ID03 images are not read. A missing alloy descriptor stays masked at 0.
    """
    row = {
        "alloy_id": alloy_row.get("alloy_id", "ID03"),
        "latent_x": str(origin_x),
        "latent_y": str(origin_y),
    }
    for name, source in HOLDOUT_DESC_SOURCES.items():
        value = alloy_row.get(source, "")
        row[name] = value
        row[f"global_{name}"] = value
    return row


def holdout_condition(alloy_row: dict[str, str], count: int) -> torch.Tensor:
    """ID03 samples. Positions are the step-16 latent origins.

    crops.csv has no ID03 rows. Both the window block and the global block use
    conditions_ID03_holdout.csv desc_cond_* values, not a measured window.
    """
    alloy = torch.tensor([float(alloy_row[name]) for name in COND_COLUMNS], dtype=torch.float32)
    alloy_id = alloy_row.get("alloy_id", "ID03")
    rows = [
        _holdout_descriptor_row(alloy_row, origin_x, origin_y)
        for origin_x, origin_y in holdout_origins(count)
    ]
    return batch_condition(rows, {alloy_id: alloy})
