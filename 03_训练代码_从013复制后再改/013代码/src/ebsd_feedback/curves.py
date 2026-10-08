from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .constants import CURVE_TARGET_COLUMNS


def constrain_curve_targets_numpy(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=float).copy()
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.shape[1] != 9:
        raise ValueError(f"力学目标必须为9维，实际为: {values.shape}")
    values[:, :5] = np.maximum.accumulate(np.maximum(values[:, :5], 0.0), axis=1)
    values[:, 5] = np.maximum(values[:, 5], values[:, :5].max(axis=1))
    values[:, 7] = np.maximum(values[:, 7], 10.0)
    values[:, 6] = np.clip(values[:, 6], 0.5, values[:, 7])
    values[:, 8] = np.clip(values[:, 8], 0.0, values[:, 5])
    return values[0] if one_dimensional else values


def reconstruct_curve(targets: np.ndarray, strain_step: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    values = dict(zip(CURVE_TARGET_COLUMNS, constrain_curve_targets_numpy(targets)))
    endpoint = max(float(values["endpoint_strain_pct"]), 10.0)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    fixed = [
        (0.5, "stress_at_0p5_pct_MPa"),
        (1.0, "stress_at_1p0_pct_MPa"),
        (2.0, "stress_at_2p0_pct_MPa"),
        (5.0, "stress_at_5p0_pct_MPa"),
        (10.0, "stress_at_10p0_pct_MPa"),
    ]
    for strain, name in fixed:
        if strain < endpoint - 1e-6 and np.isfinite(values[name]):
            points.append((strain, max(float(values[name]), 0.0)))
    peak_strain = float(values["peak_strain_pct"])
    if 0 < peak_strain < endpoint and np.isfinite(values["peak_stress_MPa"]):
        points.append((peak_strain, max(float(values["peak_stress_MPa"]), 0.0)))
    points.append((endpoint, max(float(values["endpoint_stress_MPa"]), 0.0)))
    grouped: dict[float, list[float]] = {}
    for strain, stress in points:
        grouped.setdefault(round(strain, 6), []).append(stress)
    strains = np.array(sorted(grouped), dtype=float)
    stresses = np.array([np.mean(grouped[value]) for value in strains], dtype=float)
    if len(strains) < 2:
        return strains, stresses
    # Grouping rounds near-identical anchors; use that exact final anchor for interpolation.
    endpoint = float(strains[-1])
    grid = np.arange(0.0, endpoint, strain_step)
    if len(grid) == 0 or not np.isclose(grid[-1], endpoint):
        grid = np.append(grid, endpoint)
    prediction = PchipInterpolator(strains, stresses, extrapolate=False)(grid)
    if not np.isfinite(prediction).all():
        raise ValueError("曲线重建产生非有限值，请检查9个力学目标")
    return grid, np.maximum(prediction, 0.0)
