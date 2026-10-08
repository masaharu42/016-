from __future__ import annotations

from .paths import IMAGE_MODE

COMPOSITION_COLUMNS = ["Ni", "Fe", "Cr", "Nb", "Ta", "Co", "Mo", "Al", "Ti"]

GB_DESCRIPTOR_COLUMNS = [
    "grain_size_median_um",
    "grain_size_log_spread",
    "area_weighted_aspect_ratio",
    "coarse_grain_area_fraction",
    "boundary_length_density_per_um",
]

DESCRIPTOR_COLUMNS = [
    *GB_DESCRIPTOR_COLUMNS,
    *(["csl3_boundary_fraction"] if IMAGE_MODE == "GB_CSL3" else []),
]

CURVE_TARGET_COLUMNS = [
    "stress_at_0p5_pct_MPa",
    "stress_at_1p0_pct_MPa",
    "stress_at_2p0_pct_MPa",
    "stress_at_5p0_pct_MPa",
    "stress_at_10p0_pct_MPa",
    "peak_stress_MPa",
    "peak_strain_pct",
    "endpoint_strain_pct",
    "endpoint_stress_MPa",
]

STRESS_TARGET_INDICES = [0, 1, 2, 3, 4, 5, 8]
STRAIN_TARGET_INDICES = [6, 7]

DENSE_CURVE_STRAIN_STEP_PCT = 0.05
DENSE_CURVE_MAX_STRAIN_PCT = 40.0


def canonical_alloy_id(value: str | int) -> str:
    text = str(value).strip().upper().replace("ALLOY", "").replace("ID", "")
    return f"ID{int(float(text)):02d}"
