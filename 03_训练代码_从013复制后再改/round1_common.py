# -*- coding: utf-8 -*-
"""Shared geometry for round-1 patch diffusion. Does not modify 013."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import expand_labels

LATENT_SCALE = 0.3440041526837966
HOLDOUT = "ID03"
SEED = 20261008
FULL_H = 448
FULL_W = 768
LATENT_WIN = 64
PIXEL_WIN = 256
# Rightmost 256 px of a 768-wide map is latent x 128..191.
RIGHT_LATENT = 128
# Landscape latent is 112 x 192, so y of a 64-window is 0..48.
Y_MAX = 48
NO90 = {"ID04", "ID17"}
VIEWS = [
    ("v0_rot0", ()),
    ("v1_rot90", ("rot90",)),
    ("v2_rot180", ("rot180",)),
    ("v3_rot270", ("rot270",)),
    ("v4_flipLR", ("flipLR",)),
    ("v5_flipLR_rot90", ("flipLR", "rot90")),
    ("v6_flipLR_rot180", ("flipLR", "rot180")),
    ("v7_flipLR_rot270", ("flipLR", "rot270")),
]
_TRANS = {
    "rot90": Image.Transpose.ROTATE_90,
    "rot180": Image.Transpose.ROTATE_180,
    "rot270": Image.Transpose.ROTATE_270,
    "flipLR": Image.Transpose.FLIP_LEFT_RIGHT,
}
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
CURVE_TRUE_COLUMNS = [c.replace("curve_baseline_", "curve_true_") for c in COND_COLUMNS if c.startswith("curve_baseline_")]
DESC_NAMES = ["D50", "log_spread", "area_weighted_aspect", "coarse_area_fraction", "boundary_length_density"]


def views_for(alloy_id: str):
    if alloy_id in NO90:
        return [(n, ops) for n, ops in VIEWS if "rot90" not in ops and "rot270" not in ops]
    return list(VIEWS)


def is_portrait_ops(ops) -> bool:
    return any(op in ("rot90", "rot270") for op in ops)


def apply_ops(image: Image.Image, ops) -> Image.Image:
    for op in ops:
        image = image.transpose(_TRANS[op])
    return image


def load_resized(path: Path):
    """RGB Lanczos, interior-mask nearest. Channel A matches 013 pack_image (white = interior)."""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    interior = ((rgb.max(axis=2) >= 51).astype(np.uint8) * 255)
    rgb_i = Image.fromarray(rgb, mode="RGB").resize((FULL_W, FULL_H), Image.Resampling.LANCZOS)
    mask_i = Image.fromarray(interior, mode="L").resize((FULL_W, FULL_H), Image.Resampling.NEAREST)
    return rgb_i, mask_i


def to_tensor(rgb_i: Image.Image, mask_i: Image.Image) -> torch.Tensor:
    rgb = np.asarray(rgb_i, dtype=np.float32)
    mask = np.asarray(mask_i, dtype=np.float32)
    packed = np.concatenate([rgb, mask[..., None]], axis=2)
    packed = packed / 127.5 - 1.0
    return torch.from_numpy(packed).permute(2, 0, 1).contiguous()


def segment(rgb: np.ndarray, min_px: int = 10):
    """Same parent-grain segmentation as patchdesc.py."""
    gb = rgb.max(axis=2) < 51
    lab, _ = ndimage.label(~gb)
    lab = expand_labels(lab, distance=4)
    if (lab == 0).any():
        idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = lab[tuple(idx)]
    props = regionprops(lab)
    table = {}
    for p in props:
        area = int(p.area)
        if area < 1:
            continue
        ev = np.clip(np.array(p.inertia_tensor_eigvals), 0, None) + 1.0 / 12.0
        ar = float(np.sqrt(ev.max() / ev.min()))
        table[int(p.label)] = dict(area=area, ecd=float(np.sqrt(4 * area / np.pi)), ar=ar,
                                   cy=float(p.centroid[0]), cx=float(p.centroid[1]))
    areas = np.bincount(lab.ravel())
    return lab, table, areas


def boundary_length(lab_crop: np.ndarray) -> float:
    n = (lab_crop[:, 1:] != lab_crop[:, :-1]).sum() + (lab_crop[1:, :] != lab_crop[:-1, :]).sum()
    return float(n * np.pi / 4.0)


def _stats(ecd_num, a_w, ecd_w, ar_w, length, area_tot):
    if len(ecd_num) < 3 or a_w.sum() <= 0:
        return [float("nan")] * 5
    d10, d50, d90 = np.percentile(ecd_num, [10, 50, 90])
    return [
        float(d50),
        float((np.log(d90) - np.log(d10)) / 2.5631),
        float((a_w * ar_w).sum() / a_w.sum()),
        float(a_w[ecd_w >= 20].sum() / a_w.sum()),
        float(length / area_tot),
    ]


def desc_parent(lab, table, y0, x0, h, w, min_px=10):
    crop = lab[y0:y0 + h, x0:x0 + w]
    ids, cnt = np.unique(crop, return_counts=True)
    keep = [(int(i), int(c)) for i, c in zip(ids, cnt) if int(i) in table and table[int(i)]["area"] >= min_px]
    if not keep:
        return [float("nan")] * 5
    a_in = np.array([c for _, c in keep], dtype=np.float64)
    e_full = np.array([table[i]["ecd"] for i, _ in keep], dtype=np.float64)
    ar_full = np.array([table[i]["ar"] for i, _ in keep], dtype=np.float64)
    cen = [table[i]["ecd"] for i, _ in keep if y0 <= table[i]["cy"] < y0 + h and x0 <= table[i]["cx"] < x0 + w]
    return _stats(np.array(cen, dtype=np.float64), a_in, e_full, ar_full, boundary_length(crop), float(a_in.sum()))


def truncation_ratio(lab, areas, y0, x0, h, w) -> float:
    """Fraction of window pixels whose parent grain also exists outside the window."""
    crop = lab[y0:y0 + h, x0:x0 + w]
    ids, cnt = np.unique(crop, return_counts=True)
    cut = 0
    for i, c in zip(ids.tolist(), cnt.tolist()):
        if areas[int(i)] > int(c):
            cut += int(c)
    return float(cut) / float(h * w)


def training_positions(latent_h: int, latent_w: int):
    """Integer origins whose 64-window stays inside the map and out of the right 256 px.

    On a 112x192 latent this is y in 0..48 and x+64 <= 128. A 90-degree latent is
    192x112, so a 64-wide window cannot also leave 64 columns on the right.
    """
    if latent_h < LATENT_WIN or latent_w < LATENT_WIN:
        return []
    ys = range(0, min(Y_MAX, latent_h - LATENT_WIN) + 1)
    xs = range(0, min(RIGHT_LATENT - LATENT_WIN, latent_w - 2 * LATENT_WIN) + 1)
    # min(...) can be negative when the map is too narrow.
    if min(RIGHT_LATENT - LATENT_WIN, latent_w - 2 * LATENT_WIN) < 0:
        return []
    return [(y, x) for y in ys for x in xs]


def load_vae(vae_path: Path, device: torch.device):
    import sys
    pkg = Path(__file__).resolve().parent / "013代码" / "src"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    else:
        alt = Path("/workspace/p016_pkg/src")
        if alt.is_dir() and str(alt) not in sys.path:
            sys.path.insert(0, str(alt))
    from ebsd_feedback.models.vae import BoundaryAwareVAE
    state = torch.load(vae_path, map_location="cpu", weights_only=False)
    scale = float(state["latent_scale"])
    if abs(scale - LATENT_SCALE) > 1e-12:
        raise RuntimeError(f"latent_scale {scale} != {LATENT_SCALE}")
    model = BoundaryAwareVAE(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    return model, scale


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
