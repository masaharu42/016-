"""Fold-local image proxy calibration. Outer holdout never selects weights/steps."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image

from ..constants import DESCRIPTOR_COLUMNS, canonical_alloy_id
from ..data import AlloyRepository, load_fold_manifest, image_to_tensor, resize_full_frame
from ..models.vae import BoundaryAwareVAE
from ..models.descriptor_head import LatentDescriptorHead, DescriptorTargetTransform
from ..paths import FOLD_MODEL_ROOT, LOG_ROOT
from ..monitor import TrainingMonitor
from ..utils import atomic_torch_save, atomic_json_dump, seed_everything
from .common import autocast_context, configure_torch_performance, make_adamw, make_cosine_scheduler

FOLDER = "02b_图像描述符代理"
FINAL_NAME = "图像描述符_最终模型.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_proxy(fold: Path, train_ids, device):
    path = fold / FOLDER / FINAL_NAME
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "013_image_proxy_v1" or state["holdout_id"] != fold.name:
        raise ValueError("013描述符代理检查点格式/留出折错误")
    if set(state["train_ids"]) != set(train_ids) or fold.name in train_ids:
        raise ValueError("描述符代理训练ID不符")
    if state["vae_sha256"] != sha256(fold / "01_边界感知VAE" / "VAE_最终模型.pt"):
        raise ValueError("描述符代理使用的VAE已变化，请在新的实验目录重训")
    head = LatentDescriptorHead(**state["model_config"]).to(device)
    head.load_state_dict(state["model"])
    return head.eval().requires_grad_(False), DescriptorTargetTransform.from_state_dict(state["transform"])


def train_image_descriptor(config, holdout_id: str):
    holdout = canonical_alloy_id(holdout_id)
    seed_everything(int(config.seed), bool(config.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_performance(config, device)
    fold = FOLD_MODEL_ROOT / holdout
    records = load_fold_manifest(fold / "00_折准备" / "严格留一折清单.csv")
    train = records.loc[records.role == "train"].copy().reset_index(drop=True)
    train_ids = train.alloy_id.tolist()
    if holdout in train_ids or len(train_ids) != 21:
        raise ValueError("需要21个训练合金，不能含外层留出ID")
    directory, log = fold / FOLDER, LOG_ROOT / holdout / FOLDER
    directory.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config.to_dict(), log / "实际配置.json")
    vae_path = fold / "01_边界感知VAE" / "VAE_最终模型.pt"
    vae_hash = sha256(vae_path)
    state = torch.load(vae_path, map_location="cpu", weights_only=False)
    if state["holdout_id"] != holdout:
        raise ValueError("VAE不属于当前折")
    vae = BoundaryAwareVAE(**state["model_config"]).to(device).eval().requires_grad_(False)
    vae.load_state_dict(state["model"])
    del state
    repo = AlloyRepository()
    height, width = int(config.data.image_height), int(config.data.image_width)
    precision = str(config.environment.precision)
    images, targets = [], []
    print("准备描述符代理的21张完整真实图和VAE重建图，不读取留出图作训练。", flush=True)
    with torch.no_grad():
        for i, aid in enumerate(train_ids):
            with Image.open(repo.row(aid)["image_path"]) as source:
                x = image_to_tensor(resize_full_frame(source, height, width))[None].to(device)
            with autocast_context(device, precision):
                z, _, _ = vae.encode(x, sample=False)
                reconstruction = vae.decode(z)
            images.append(torch.stack([x[0, :3].cpu(), reconstruction[0, :3].float().cpu()]))
            targets.append(train.loc[i, ["desc_true_" + n for n in DESCRIPTOR_COLUMNS]].to_numpy(np.float32))
    pixels = torch.stack(images)  # [21,2,3,H,W]; real/reconstruction share one label
    values = torch.tensor(np.stack(targets))
    rng = np.random.default_rng(int(config.seed))
    order = rng.permutation(len(train_ids))
    validation = order[:int(config.train.validation_alloy_count)].tolist()
    development = order[int(config.train.validation_alloy_count):].tolist()
    architecture = {"latent_channels": 3, "hidden_dim": int(config.model.hidden_dim), "output_dim": len(DESCRIPTOR_COLUMNS)}

    def fit_phase(name, indices, total_steps, validate):
        seed_everything(int(config.seed), bool(config.deterministic))
        transform = DescriptorTargetTransform.fit(values[indices])
        model = LatentDescriptorHead(**architecture).to(device)
        optimizer = make_adamw(model.parameters(), device, float(config.train.learning_rate), float(config.train.weight_decay))
        scheduler = make_cosine_scheduler(optimizer, min(100, total_steps//5), total_steps)
        checkpoint = directory / f"{name}_最近断点.pt"
        start, best, best_step = 0, float("inf"), total_steps
        if bool(config.train.resume) and checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["vae_sha256"] != vae_hash or saved["indices"] != indices or saved["total_steps"] != total_steps:
                raise ValueError("代理断点数据/步数与当前运行不符")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start, best, best_step = saved["step"], saved["best"], saved["best_step"]
            torch.set_rng_state(saved["rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        transformed = transform.transform(values).to(device)
        monitor = TrainingMonitor(log / name, f"{holdout}-描述符代理-{name}", total_steps,
            ["loss_total", "validation_loss", "lr", "grad_norm"], int(config.monitor.log_every), 20, False)
        step = start
        def save():
            atomic_torch_save({"step": step, "best": best, "best_step": best_step, "indices": indices,
                "total_steps": total_steps, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "vae_sha256": vae_hash, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, checkpoint)
        try:
            for step in range(start+1, total_steps+1):
                # All alloys equally likely; real/reconstruction pair shares no independent sample claim.
                index = torch.tensor(indices)[torch.randint(len(indices), (int(config.train.batch_size),))]
                version = torch.randint(2, index.shape)
                batch = pixels[index, version].to(device)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                with autocast_context(device, precision):
                    prediction = model(batch)
                    loss = F.smooth_l1_loss(prediction.float(), transformed[index.to(device)], beta=.5)
                if not torch.isfinite(loss):
                    raise RuntimeError("描述符代理损失非有限值")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.gradient_clip))
                optimizer.step()
                scheduler.step()
                metrics = {"loss_total": float(loss.detach()), "lr": scheduler.get_last_lr()[0], "grad_norm": float(norm)}
                if validate and (step % int(config.train.validation_every) == 0 or step == total_steps):
                    model.eval()
                    errors = []
                    with torch.no_grad(), autocast_context(device, precision):
                        for j in validation:
                            pred = model(pixels[j].to(device)).float()
                            errors.append(float(F.smooth_l1_loss(pred, transformed[j:j+1].expand_as(pred), beta=.5)))
                    score = float(np.mean(errors))
                    metrics["validation_loss"] = score
                    if score < best:
                        best, best_step = score, step
                monitor.log(step, metrics, force=step == 1)
                if step % int(config.monitor.checkpoint_every) == 0:
                    save()
            save()
            monitor.close("completed")
        except KeyboardInterrupt:
            save()
            monitor.close("interrupted")
            raise
        except Exception:
            monitor.close("failed")
            raise
        return model, transform, best_step, best

    selection = directory / "内层选步数.json"
    if selection.exists():
        import json
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        if chosen["vae_sha256"] != vae_hash:
            raise ValueError("内层选步记录使用其他VAE")
        selected_steps, best = chosen["selected_steps"], chosen["validation_loss"]
    else:
        model, _, selected_steps, best = fit_phase("内层验证", development, int(config.train.max_steps), True)
        del model
        atomic_json_dump({"selected_steps": selected_steps, "validation_loss": best, "vae_sha256": vae_hash,
            "development_ids": [train_ids[i] for i in development], "validation_ids": [train_ids[i] for i in validation],
            "limitation": "VAE was fitted on outer-training 21; inner proxy selection is conditional on this frozen VAE, not fully nested VAE validation"}, selection)
    print(f"内层选择 {selected_steps} 步，重置图像代理并在21个训练合金上拟合。", flush=True)
    model, transform, _, _ = fit_phase("21合金重拟合", list(range(21)), selected_steps, False)
    model.eval()
    # Outer holdout enters only this post-fit reporting loop.
    report = []
    with torch.no_grad():
        for aid in [*train_ids, holdout]:
            with Image.open(repo.row(aid)["image_path"]) as source:
                x = image_to_tensor(resize_full_frame(source, height, width))[None].to(device)
            z, _, _ = vae.encode(x, sample=False)
            recon = vae.decode(z)
            if aid == holdout:
                from torchvision.utils import save_image
                panels = torch.cat([x[:, :3], recon[:, :3], x[:, 3:4].repeat(1,3,1,1), recon[:, 3:4].repeat(1,3,1,1)])
                save_image(panels, log / "留出图IPF_GB重建对照.png", nrow=2, normalize=True, value_range=(-1,1))
                rgb_mse = float(((x[:, :3]-recon[:, :3])/2).square().mean())
                real_boundary, pred_boundary = x[:,3:4]<0, recon[:,3:4]<0
                dice = float(2*(real_boundary & pred_boundary).sum()/(real_boundary.sum()+pred_boundary.sum()).clamp_min(1))
                atomic_json_dump({"role": "postfit holdout reporting ONLY; no selection", "IPF_PSNR_dB": -10*np.log10(max(rgb_mse,1e-12)),
                    "GB_Dice_at_0p5": dice}, log / "留出图VAE重建指标.json")
            for kind, tensor in (("real", x), ("reconstruction", recon)):
                pred = transform.inverse(model(tensor[:, :3]))[0].cpu().numpy()
                for j, column in enumerate(DESCRIPTOR_COLUMNS):
                    true = float(repo.row(aid)[column])
                    report.append({"alloy_id": aid, "role": "holdout" if aid == holdout else "train", "image": kind,
                        "descriptor": column, "truth": true, "proxy_prediction": float(pred[j]),
                        "signed_relative_error_pct": float(100*(pred[j]-true)/abs(true)) if abs(true)>1e-12 else float("nan")})
    pd.DataFrame(report).to_csv(log / "真实与重建图代理校验.csv", index=False, encoding="utf-8-sig")
    final = directory / FINAL_NAME
    atomic_torch_save({"schema": "013_image_proxy_v1", "model": model.state_dict(), "model_config": architecture,
        "transform": transform.state_dict(), "holdout_id": holdout, "train_ids": train_ids,
        "selected_steps": selected_steps, "vae_sha256": vae_hash, "image_hw": [height, width]}, final)
    return final
