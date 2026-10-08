from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .constants import (
    COMPOSITION_COLUMNS,
    CURVE_TARGET_COLUMNS,
    DESCRIPTOR_COLUMNS,
    canonical_alloy_id,
)
from .data import AlloyRepository
from .curves import constrain_curve_targets_numpy
from .paths import FOLD_MODEL_ROOT


@dataclass
class Prediction:
    mean: np.ndarray
    std: np.ndarray


def _make_gp(feature_count: int, seed: int, restarts: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(
            length_scale=np.ones(feature_count),
            length_scale_bounds=(0.15, 20.0),
            nu=2.5,
        )
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=False,
        n_restarts_optimizer=restarts,
        random_state=seed,
    )


def fit_bundle(
    x: np.ndarray,
    descriptors: np.ndarray,
    curves: np.ndarray,
    seed: int,
    restarts: int,
) -> dict[str, Any]:
    x_scaler = StandardScaler().fit(x)
    descriptor_scaler = StandardScaler().fit(descriptors)
    curve_scaler = StandardScaler().fit(curves)
    x_scaled = x_scaler.transform(x)
    y_groups = {
        "descriptor_models": descriptor_scaler.transform(descriptors),
        "curve_models": curve_scaler.transform(curves),
    }
    bundle: dict[str, Any] = {
        "x_scaler": x_scaler,
        "descriptor_scaler": descriptor_scaler,
        "curve_scaler": curve_scaler,
        "composition_columns": COMPOSITION_COLUMNS,
        "descriptor_columns": DESCRIPTOR_COLUMNS,
        "curve_columns": CURVE_TARGET_COLUMNS,
    }
    for group_name, targets in y_groups.items():
        models = []
        for target_index in range(targets.shape[1]):
            model = _make_gp(x.shape[1], seed + target_index, restarts)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_scaled, targets[:, target_index])
            models.append(model)
        bundle[group_name] = models
    return bundle


def _predict_models(
    models: list[GaussianProcessRegressor],
    output_scaler: StandardScaler,
    x_scaled: np.ndarray,
) -> Prediction:
    means, stds = [], []
    for index, model in enumerate(models):
        mean, std = model.predict(x_scaled, return_std=True)
        means.append(mean * output_scaler.scale_[index] + output_scaler.mean_[index])
        stds.append(std * output_scaler.scale_[index])
    return Prediction(np.column_stack(means), np.column_stack(stds))


def predict_bundle(bundle: dict[str, Any], composition: np.ndarray) -> dict[str, Prediction]:
    x_scaled = bundle["x_scaler"].transform(np.asarray(composition, dtype=np.float64))
    curve_prediction = _predict_models(
        bundle["curve_models"], bundle["curve_scaler"], x_scaled
    )
    curve_prediction.mean = constrain_curve_targets_numpy(curve_prediction.mean)
    return {
        "descriptors": _predict_models(
            bundle["descriptor_models"], bundle["descriptor_scaler"], x_scaled
        ),
        "curves": curve_prediction,
    }


def prepare_strict_fold(
    holdout_id: str,
    seed: int = 20260819,
    inner_restarts: int = 1,
    final_restarts: int = 4,
) -> Path:
    repository = AlloyRepository()
    holdout_id = canonical_alloy_id(holdout_id)
    train_ids, _ = repository.split_ids(holdout_id)
    table = repository.table.copy().set_index("alloy_id", drop=False)
    x_all = table[COMPOSITION_COLUMNS].to_numpy(dtype=np.float64)
    desc_all = table[DESCRIPTOR_COLUMNS].to_numpy(dtype=np.float64)
    curve_all = table[CURVE_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    ids = table["alloy_id"].tolist()
    train_indices = [ids.index(value) for value in train_ids]
    holdout_index = ids.index(holdout_id)

    desc_mean = np.full_like(desc_all, np.nan)
    desc_std = np.full_like(desc_all, np.nan)
    curve_mean = np.full_like(curve_all, np.nan)
    curve_std = np.full_like(curve_all, np.nan)

    # Every training alloy receives an inner-LOO prediction. This avoids teaching the
    # diffusion model with descriptor conditions fitted on its own EBSD label.
    for inner_position, validation_index in enumerate(train_indices):
        print(f"[{holdout_id} GPR准备] 内层留一 {inner_position+1}/{len(train_indices)}，当前 {ids[validation_index]}", flush=True)
        inner_train = [index for index in train_indices if index != validation_index]
        inner_bundle = fit_bundle(
            x_all[inner_train],
            desc_all[inner_train],
            curve_all[inner_train],
            seed + 1000 + inner_position,
            inner_restarts,
        )
        prediction = predict_bundle(inner_bundle, x_all[[validation_index]])
        desc_mean[validation_index] = prediction["descriptors"].mean[0]
        desc_std[validation_index] = prediction["descriptors"].std[0]
        curve_mean[validation_index] = prediction["curves"].mean[0]
        curve_std[validation_index] = prediction["curves"].std[0]

    print(f"[{holdout_id} GPR准备] 用21个训练合金拟合外层GPR，随后预测留出成分", flush=True)
    final_bundle = fit_bundle(
        x_all[train_indices],
        desc_all[train_indices],
        curve_all[train_indices],
        seed,
        final_restarts,
    )
    holdout_prediction = predict_bundle(final_bundle, x_all[[holdout_index]])
    desc_mean[holdout_index] = holdout_prediction["descriptors"].mean[0]
    desc_std[holdout_index] = holdout_prediction["descriptors"].std[0]
    curve_mean[holdout_index] = holdout_prediction["curves"].mean[0]
    curve_std[holdout_index] = holdout_prediction["curves"].std[0]

    records = []
    for index, alloy_id in enumerate(ids):
        if alloy_id not in train_ids and alloy_id != holdout_id:
            continue
        source = table.loc[alloy_id]
        row: dict[str, Any] = {
            "alloy_id": alloy_id,
            "role": "holdout" if alloy_id == holdout_id else "train",
            "image_path": source["image_path"],
        }
        row.update({f"comp_{name}": source[name] for name in COMPOSITION_COLUMNS})
        row.update({f"desc_true_{name}": source[name] for name in DESCRIPTOR_COLUMNS})
        row.update({f"desc_cond_{name}": desc_mean[index, j] for j, name in enumerate(DESCRIPTOR_COLUMNS)})
        row.update({f"desc_std_{name}": desc_std[index, j] for j, name in enumerate(DESCRIPTOR_COLUMNS)})
        row.update({f"curve_true_{name}": source[name] for name in CURVE_TARGET_COLUMNS})
        row.update({f"curve_baseline_{name}": curve_mean[index, j] for j, name in enumerate(CURVE_TARGET_COLUMNS)})
        row.update({f"curve_baseline_std_{name}": curve_std[index, j] for j, name in enumerate(CURVE_TARGET_COLUMNS)})
        row.update(
            {
                f"curve_residual_{name}": source[name] - curve_mean[index, j]
                for j, name in enumerate(CURVE_TARGET_COLUMNS)
            }
        )
        records.append(row)

    output_dir = FOLD_MODEL_ROOT / holdout_id / "00_折准备"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "严格留一折清单.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    joblib.dump(final_bundle, output_dir / "成分到描述符与曲线_GPR.joblib")
    metadata = {
        "holdout_id": holdout_id,
        "train_ids": train_ids,
        "training_alloy_count": len(train_ids),
        "inner_condition_rule": "21个训练合金分别使用内层留一预测",
        "holdout_condition_rule": "仅使用21个训练合金拟合后预测",
        "seed": seed,
        "manifest": str(manifest_path),
    }
    (output_dir / "折说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path
