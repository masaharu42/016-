"""Cubic CTF to fixed-bin IPF direction fractions, NOT a full ODF.

Bunge passive sample-to-crystal convention: g = (Rz(phi1) Rx(Phi)
Rz(phi2)).T. Cubic m-3m directions reduce by absolute value and sorting.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

SCHEMA = "cubic_ipf_ratio_triangle_4_v1"
ORIENTATION_COLUMNS = tuple(f"ipf_bin_{j}_{i}" for j in range(4) for i in range(j + 1))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ctf(path: str | Path) -> tuple[pd.DataFrame, dict]:
    metadata = {}
    with Path(path).open(encoding="utf-8-sig", errors="replace") as handle:
        if handle.readline().strip() != "Channel Text File":
            raise ValueError(f"Not a Channel Text File: {path}")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing CTF Phase table: {path}")
            fields = line.rstrip("\r\n").split("\t")
            if fields[0] == "Phases":
                count = int(fields[1])
                phases = [handle.readline().rstrip().split("\t") for _ in range(count)]
                if count != 1 or len(phases[0]) < 5 or int(phases[0][3]) != 11:
                    raise ValueError("This version supports one cubic m-3m phase only")
                metadata["phase_name"] = phases[0][2]
                metadata["laue_group"] = 11
            elif fields[0] == "Phase":
                frame = pd.read_csv(handle, sep="\t", names=fields)
                break
            elif len(fields) == 2:
                metadata[fields[0]] = fields[1]
    required = {"Phase", "X", "Y", "Euler1", "Euler2", "Euler3"}
    if not required.issubset(frame.columns) or "laue_group" not in metadata:
        raise ValueError(f"Incomplete CTF fields/phases: {path}")
    for name in ("XCells", "YCells"):
        metadata[name] = int(metadata[name])
    for name in ("XStep", "YStep"):
        metadata[name] = float(metadata[name])
        if metadata[name] <= 0:
            raise ValueError("CTF step must be positive")
    if len(frame) != metadata["XCells"] * metadata["YCells"]:
        raise ValueError("CTF row count does not match rectangular grid")
    coordinates = frame[["X", "Y"]].to_numpy(float)
    steps = np.array([metadata["XStep"], metadata["YStep"]])
    grid = (coordinates - np.nanmin(coordinates, axis=0)) / steps
    if not np.isfinite(grid).all() or not np.allclose(grid, np.rint(grid), atol=0.002):
        raise ValueError("Invalid CTF grid coordinates")
    grid = np.rint(grid).astype(int)
    if (grid[:, 0].max() >= metadata["XCells"] or grid[:, 1].max() >= metadata["YCells"]
            or len(np.unique(grid, axis=0)) != len(frame)):
        raise ValueError("Duplicate/out-of-range CTF coordinates")
    frame["grid_x"], frame["grid_y"] = grid[:, 0], grid[:, 1]
    return frame, metadata


def cubic_ipf_bins(eulers_deg: np.ndarray, axis: str = "Z") -> np.ndarray:
    if axis not in ("X", "Y", "Z"):
        raise ValueError("IPF reference axis must be X, Y or Z")
    eulers = np.asarray(eulers_deg, dtype=float)
    if eulers.ndim != 2 or eulers.shape[1] != 3 or not np.isfinite(eulers).all():
        raise ValueError("Expected finite [N,3] Bunge Euler angles")
    reference = np.eye(3)[("X", "Y", "Z").index(axis)]
    directions = Rotation.from_euler("ZXZ", eulers, degrees=True).inv().apply(reference)
    reduced = np.sort(np.abs(directions), axis=1)
    ratio = reduced[:, :2] / reduced[:, 2:3].clip(1e-12)
    cells = np.minimum((ratio * 4).astype(int), 3)
    i, j = cells[:, 0], cells[:, 1]
    return j * (j + 1) // 2 + i


def extract_orientation_stats(path: str | Path, axis: str = "Z", max_mad: float | None = None):
    frame, metadata = read_ctf(path)
    eulers = frame[["Euler1", "Euler2", "Euler3"]].to_numpy(float)
    phase = frame["Phase"].to_numpy(float)
    if not np.isin(phase, [0, 1]).all():
        raise ValueError("Unsupported phase IDs")
    valid = (phase == 1) & np.isfinite(eulers).all(axis=1)
    valid &= (eulers[:, 1] >= 0) & (eulers[:, 1] <= 180)
    if "Error" in frame:
        valid &= frame["Error"].to_numpy(float) == 0
    if max_mad is not None:
        if "MAD" not in frame or max_mad <= 0:
            raise ValueError("Positive max_mad and a MAD field are required")
        mad = frame["MAD"].to_numpy(float)
        valid &= np.isfinite(mad) & (mad <= max_mad)
    if not valid.any():
        raise ValueError(f"No valid indexed orientations: {path}")
    bins = cubic_ipf_bins(eulers[valid], axis)
    fractions = np.bincount(bins, minlength=len(ORIENTATION_COLUMNS)) / len(bins)
    audit = {"valid_pixel_ratio": float(valid.mean()), "valid_pixels": int(valid.sum()),
             "ipf_axis": axis, "schema": SCHEMA, "max_mad": max_mad, **metadata}
    for field in ("MAD", "BC", "BS"):
        if field in frame:
            quality = frame.loc[valid, field].to_numpy(float)
            finite = quality[np.isfinite(quality)]
            audit[f"mean_{field}"] = float(finite.mean()) if len(finite) else None
    # Discrete registration aid, not the software's IPF color key.
    bin_map = np.full((metadata["YCells"], metadata["XCells"]), -1, dtype=int)
    bin_map[frame.loc[valid, "grid_y"], frame.loc[valid, "grid_x"]] = bins
    return dict(zip(ORIENTATION_COLUMNS, fractions.tolist())), audit, bin_map


def load_orientation_labels(path: Path, train_ids: list[str], require_review: bool = True) -> pd.DataFrame:
    from .constants import canonical_alloy_id

    frame = pd.read_csv(path, dtype={"alloy_id": str})
    frame["alloy_id"] = frame["alloy_id"].map(canonical_alloy_id)
    frame = frame[frame["alloy_id"].isin(train_ids)].copy()
    if frame["alloy_id"].duplicated().any() or set(frame["alloy_id"]) != set(train_ids):
        raise ValueError("Missing/duplicate training alloy orientation labels")
    if not (frame["schema"] == SCHEMA).all() or frame["ipf_axis"].nunique() != 1:
        raise ValueError("Incompatible orientation schema or mixed IPF axes")
    if require_review and not frame["registration_reviewed"].astype(str).str.lower().isin(["true", "1"]).all():
        raise ValueError("Review CTF/image ROI and IPF axis first; registration_reviewed must be true")
    values = frame[list(ORIENTATION_COLUMNS)].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1, atol=1e-5):
        raise ValueError("IPF fractions must be finite, nonnegative and sum to one")
    return frame.set_index("alloy_id").loc[train_ids].reset_index()


def rgb_histogram(image: np.ndarray) -> np.ndarray:
    """Independent RGB distribution, explicitly not crystallographic orientation."""
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Expected H,W,3 RGB image")
    if np.issubdtype(values.dtype, np.integer):
        values = values.astype(float) / 255.0
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Expected RGB in [0,1] or uint8")
    counts, _ = np.histogramdd(values.reshape(-1, 3), bins=[4, 4, 4], range=[(0, 1)] * 3)
    return (counts / counts.sum()).ravel().astype(np.float32)
