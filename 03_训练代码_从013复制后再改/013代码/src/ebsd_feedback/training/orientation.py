from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from ..constants import canonical_alloy_id
from ..data import image_to_tensor, load_fold_manifest, resize_full_frame
from ..models.orientation_head import LatentOrientationHead
from ..models.vae import BoundaryAwareVAE
from ..orientation import ORIENTATION_COLUMNS, SCHEMA, file_sha256, load_orientation_labels
from ..paths import FOLD_MODEL_ROOT, LOG_ROOT, resolve_project_path
from ..utils import atomic_json_dump, atomic_torch_save, seed_everything
from .orientation_guidance import js_divergence

ORIENTATION_STAGE = "06_CTF取向头"


def label_path(config) -> Path:
    return resolve_project_path(config.orientation.labels)


def orientation_stage_folder(config, feedback: bool) -> str:
    if not bool(getattr(getattr(config, "orientation", None), "enabled", False)):
        return "04_力学反馈扩散" if feedback else "03_基础条件扩散"
    return "08_CTF取向反馈扩散" if feedback else "07_CTF取向基础扩散"


def train_orientation_head(config, holdout_id: str) -> Path:
    holdout_id = canonical_alloy_id(holdout_id)
    seed_everything(int(config.seed), bool(config.deterministic))
    fold_root = FOLD_MODEL_ROOT / holdout_id
    records = load_fold_manifest(fold_root / "00_折准备" / "严格留一折清单.csv")
    train_ids = sorted(records.loc[records.role == "train", "alloy_id"].tolist())
    if holdout_id in train_ids or len(train_ids) < 5:
        raise ValueError("Invalid outer fold: holdout leakage or too few training alloys")
    labels = load_orientation_labels(label_path(config), train_ids)
    source_path = fold_root / "01_边界感知VAE" / "VAE_最终模型.pt"
    state = torch.load(source_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = BoundaryAwareVAE(**state["model_config"]).to(device)
    vae.load_state_dict(state["model"])
    vae.eval().requires_grad_(False)
    rows = records.set_index("alloy_id")
    latents = []
    with torch.no_grad():
        for alloy_id in train_ids:
            image_path = resolve_project_path(rows.loc[alloy_id, "image_path"])
            if file_sha256(image_path) != labels.set_index("alloy_id").loc[alloy_id, "image_sha256"]:
                raise ValueError(f"Image changed since CTF registration review: {alloy_id}")
            with Image.open(image_path) as source:
                image = image_to_tensor(resize_full_frame(source, int(config.data.image_height),
                                                        int(config.data.image_width)))[None].to(device)
            latent, _, _ = vae.encode(image, sample=False)
            latents.append((latent * float(state["latent_scale"])).detach())
            print(f"[{holdout_id}] encode real image {alloy_id}", flush=True)
    latent = torch.cat(latents)
    targets = torch.tensor(labels[list(ORIENTATION_COLUMNS)].to_numpy("float32"), device=device)
    del vae
    # Inner validation never uses the outer holdout; IDs, not crops, are split.
    generator = torch.Generator().manual_seed(int(config.seed))
    order = torch.randperm(len(train_ids), generator=generator).tolist()
    n_val = max(3, len(train_ids) // 5)
    val_indices, fit_indices = order[:n_val], order[n_val:]
    baseline = targets[fit_indices].mean(0, keepdim=True).expand(n_val, -1)
    baseline_js = float(js_divergence(baseline, targets[val_indices]).mean())
    model_config = {"latent_channels": latent.shape[1], "hidden_dim": int(config.orientation.hidden_dim)}

    def new_head():
        torch.manual_seed(int(config.seed))
        head = LatentOrientationHead(**model_config).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(config.orientation.head_learning_rate),
                                      weight_decay=0.01)
        return head, optimizer

    head, optimizer = new_head()
    log_dir = LOG_ROOT / holdout_id / ORIENTATION_STAGE
    log_dir.mkdir(parents=True, exist_ok=True)
    logs, best_js, best_step = [], float("inf"), 0
    max_steps = int(config.orientation.head_steps)
    batch_size = min(int(config.orientation.head_batch_size), len(fit_indices))
    for step in range(1, max_steps + 1):
        head.train()
        indices = torch.tensor(fit_indices, device=device)[torch.randperm(len(fit_indices), device=device)[:batch_size]]
        optimizer.zero_grad(set_to_none=True)
        loss = js_divergence(head(latent[indices]), targets[indices]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == max_steps:
            head.eval()
            with torch.no_grad():
                score = float(js_divergence(head(latent[val_indices]), targets[val_indices]).mean())
            logs.append({"step": step, "train_js": float(loss.detach()), "validation_js": score,
                         "mean_baseline_js": baseline_js})
            pd.DataFrame(logs).to_csv(log_dir / "head_validation.csv", index=False)
            print(f"[{holdout_id}] head {step}/{max_steps} val_JS={score:.6f} baseline={baseline_js:.6f}", flush=True)
            if score < best_js:
                best_js, best_step = score, step
    passed = best_js < baseline_js
    report = {"holdout_id": holdout_id, "train_ids": train_ids,
              "inner_fit_ids": [train_ids[i] for i in fit_indices],
              "inner_validation_ids": [train_ids[i] for i in val_indices],
              "best_step": best_step, "validation_js": best_js, "mean_baseline_js": baseline_js,
              "passed": passed, "rule": "inner validation JS must beat training-mean baseline",
              "limitation": "One small inner split; proxy validation, not generated orientation measurement"}
    atomic_json_dump(report, log_dir / "head_validation.json")
    if not passed:
        raise RuntimeError(
            f"取向头内部验证未优于训练均值基线，已阻止后续扩散训练。"
            f"验证JS={best_js:.6f}，基线JS={baseline_js:.6f}；报告：{log_dir / 'head_validation.json'}")
    # Refit on all outer-training alloys for the selected number of steps.
    head, optimizer = new_head()
    for step in range(1, best_step + 1):
        indices = torch.randperm(len(train_ids), device=device)[:batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = js_divergence(head(latent[indices]), targets[indices]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == best_step:
            print(f"[{holdout_id}] head refit {step}/{best_step}, JS={float(loss.detach()):.6f}", flush=True)
    checkpoint = {"model": head.state_dict(), "model_config": head.config(), "schema": SCHEMA,
                  "holdout_id": holdout_id, "train_ids": train_ids, "validation": report,
                  "labels_sha256": file_sha256(label_path(config)), "vae_sha256": file_sha256(source_path),
                  "ipf_axis": labels.ipf_axis.iloc[0], "latent_scale": float(state["latent_scale"]),
                  "image_height": int(config.data.image_height), "image_width": int(config.data.image_width)}
    output = fold_root / ORIENTATION_STAGE / "orientation_head_final.pt"
    atomic_torch_save(checkpoint, output)
    return output


def load_frozen_orientation_head(config, fold_root: Path, train_ids: list[str], device):
    path = fold_root / ORIENTATION_STAGE / "orientation_head_final.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["schema"] != SCHEMA or state["holdout_id"] != fold_root.name
            or set(state["train_ids"]) != set(train_ids) or not state["validation"]["passed"]):
        raise ValueError("Orientation head fold/schema/validation mismatch")
    if (state["labels_sha256"] != file_sha256(label_path(config))
            or state["vae_sha256"] != file_sha256(fold_root / "01_边界感知VAE" / "VAE_最终模型.pt")
            or state["image_height"] != int(config.data.image_height)
            or state["image_width"] != int(config.data.image_width)):
        raise ValueError("Orientation labels/VAE/image size changed; retrain orientation head")
    head = LatentOrientationHead(**state["model_config"]).to(device)
    head.load_state_dict(state["model"])
    head.eval().requires_grad_(False)
    state["head_checkpoint_sha256"] = file_sha256(path)
    return head, state
