# -*- coding: utf-8 -*-
"""Append window position and grain-size fields to the 28-D alloy condition.

These fields are condition inputs. They are not a loss. An empty or non-finite
descriptor becomes 0 with a presence bit of 0, so a missing cell does not crash.
"""
from __future__ import annotations

import math

import torch

from train_round1 import COND_COLUMNS

ALLOY_CONDITION_DIM = len(COND_COLUMNS)
# Landscape latent is 112 x 192. A 64 window's origin is at most x=128, y=48.
WINDOW_X_SPAN = 128.0
WINDOW_Y_SPAN = 48.0
WINDOW_FEATURE_NAMES = (
    "window_x",
    "window_y",
    "window_d50",
    "window_d50_present",
    "window_log_spread",
    "window_log_spread_present",
)
CONDITION_DIM = ALLOY_CONDITION_DIM + len(WINDOW_FEATURE_NAMES)
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


def window_features(row: dict[str, str]) -> torch.Tensor:
    """Six numbers: normalized latent origin, D50, log_spread, and two masks."""
    x = float(row["latent_x"]) / WINDOW_X_SPAN
    y = float(row["latent_y"]) / WINDOW_Y_SPAN
    d50, d50_present = finite_field(row.get("D50", ""))
    spread, spread_present = finite_field(row.get("log_spread", ""))
    return torch.tensor(
        [x, y, d50, d50_present, spread, spread_present],
        dtype=torch.float32,
    )


def _column_std(column: torch.Tensor) -> torch.Tensor:
    if column.numel() < 2:
        return column.new_tensor(1.0)
    deviation = column.std(unbiased=True)
    if not torch.isfinite(deviation) or float(deviation) <= 0.0:
        return column.new_tensor(1.0)
    return deviation


def window_feature_statistics(rows: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std for the encoder. Descriptor stats use only finite rows."""
    stacked = torch.stack([window_features(row) for row in rows])
    mean = torch.zeros(len(WINDOW_FEATURE_NAMES))
    std = torch.ones(len(WINDOW_FEATURE_NAMES))
    for index in (0, 1, 3, 5):
        column = stacked[:, index]
        mean[index] = column.mean()
        std[index] = _column_std(column)
    for value_index, mask_index in ((2, 3), (4, 5)):
        present = stacked[:, mask_index] > 0.5
        if int(present.sum()) >= 2:
            column = stacked[present, value_index]
            mean[value_index] = column.mean()
            std[value_index] = _column_std(column)
    return mean, std.clamp_min(1e-6)


def batch_condition(rows: list[dict[str, str]], alloy_by_id: dict[str, torch.Tensor]) -> torch.Tensor:
    """28 alloy numbers, then the six window numbers. Raw, not normalized."""
    alloy = torch.stack([alloy_by_id[row["alloy_id"]].detach().float().cpu() for row in rows])
    window = torch.stack([window_features(row) for row in rows])
    if alloy.shape[1] != ALLOY_CONDITION_DIM or window.shape[1] != len(WINDOW_FEATURE_NAMES):
        raise RuntimeError(f"条件拼接维度不对: alloy {tuple(alloy.shape)} window {tuple(window.shape)}")
    return torch.cat([alloy, window], dim=1)


def holdout_condition(alloy_row: dict[str, str], count: int) -> torch.Tensor:
    """ID03 samples. Positions are the step-16 latent origins.

    Window D50 and log_spread are the alloy condition fields already in the
    holdout csv. ID03 images are not read, and crops.csv has no ID03 rows.
    A missing alloy descriptor stays masked at 0.
    """
    alloy = torch.tensor([float(alloy_row[name]) for name in COND_COLUMNS], dtype=torch.float32)
    alloy_id = alloy_row.get("alloy_id", "ID03")
    rows = []
    for origin_x, origin_y in holdout_origins(count):
        rows.append({
            "alloy_id": alloy_id,
            "latent_x": str(origin_x),
            "latent_y": str(origin_y),
            "D50": alloy_row.get("desc_cond_grain_size_median_um", ""),
            "log_spread": alloy_row.get("desc_cond_grain_size_log_spread", ""),
        })
    return batch_condition(rows, {alloy_id: alloy})
