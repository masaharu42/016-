from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import save_image
from PIL import Image

from .config import ConfigNode
from .constants import COMPOSITION_COLUMNS, CURVE_TARGET_COLUMNS, DESCRIPTOR_COLUMNS, canonical_alloy_id
from .curves import reconstruct_curve
from .data import AlloyRepository, image_to_tensor, resize_full_frame
from .folds import predict_bundle
from .models.diffusion import ConditionalLatentUNet, DiffusionSchedule
from .models.descriptor_head import DescriptorTargetTransform, LatentDescriptorHead
from .models.mechanics import EbsdMechanicsSurrogate
from .models.vae import BoundaryAwareVAE
from .paths import FOLD_MODEL_ROOT, MODELING_DATA_ROOT, PREDICTION_ROOT
from .training.common import ExponentialMovingAverage
from .training.orientation import label_path, orientation_stage_folder
from .training.orientation_guidance import js_divergence
from .models.orientation_head import LatentOrientationHead
from .orientation import ORIENTATION_COLUMNS, SCHEMA, file_sha256, load_orientation_labels, rgb_histogram


def _composition_from_input(
    holdout_id: str, composition_csv: str | Path | None
) -> tuple[np.ndarray, str]:
    if composition_csv:
        frame = pd.read_csv(composition_csv)
        missing = [column for column in COMPOSITION_COLUMNS if column not in frame]
        if missing or len(frame) != 1:
            raise ValueError(f"新成分CSV必须恰好一行且包含9元素，缺少: {missing}")
        return frame[COMPOSITION_COLUMNS].to_numpy(dtype=np.float64), "新输入成分"
    row = AlloyRepository().row(holdout_id)
    return row[COMPOSITION_COLUMNS].to_numpy(dtype=np.float64)[None], f"{holdout_id}成分"


def generate_three_images(
    config: ConfigNode,
    holdout_id: str,
    composition_csv: str | Path | None = None,
) -> Path:
    holdout_id = canonical_alloy_id(holdout_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_root = FOLD_MODEL_ROOT / holdout_id
    bundle = joblib.load(fold_root / "00_折准备" / "成分到描述符与曲线_GPR.joblib")
    composition, source_name = _composition_from_input(holdout_id, composition_csv)
    prediction = predict_bundle(bundle, composition)
    descriptor_mean = prediction["descriptors"].mean.astype(np.float32)
    descriptor_std = prediction["descriptors"].std.astype(np.float32)
    curve_baseline = prediction["curves"].mean.astype(np.float32)
    condition = torch.from_numpy(
        np.concatenate([composition, descriptor_mean, descriptor_std, curve_baseline], axis=1).astype(
            np.float32
        )
    ).to(device)

    vae_state = torch.load(
        fold_root / "01_边界感知VAE" / "VAE_最终模型.pt",
        map_location="cpu",
        weights_only=False,
    )
    vae = BoundaryAwareVAE(**vae_state["model_config"]).to(device)
    vae.load_state_dict(vae_state["model"])
    vae.eval().requires_grad_(False)
    latent_scale = float(vae_state["latent_scale"])

    mechanics_state = torch.load(
        fold_root / "02_力学代理" / "力学代理_最终模型.pt",
        map_location="cpu",
        weights_only=False,
    )
    mechanics = EbsdMechanicsSurrogate(**mechanics_state["model_config"]).to(device)
    mechanics.load_state_dict(mechanics_state["model"])
    mechanics.eval().requires_grad_(False)

    orientation_enabled = bool(getattr(getattr(config, "orientation", None), "enabled", False))
    requested_stage = str(getattr(config.inference, "checkpoint_stage", "feedback"))
    if requested_stage not in {"base", "feedback"}:
        raise ValueError("checkpoint_stage必须为base或feedback")
    diffusion_path = fold_root / orientation_stage_folder(config, requested_stage == "feedback") / "扩散_最终模型.pt"
    diffusion_state = torch.load(diffusion_path, map_location="cpu", weights_only=False)
    orientation_head = None
    orientation_state = diffusion_state.get("orientation_state")
    if orientation_enabled:
        if (not orientation_state or orientation_state["schema"] != SCHEMA
                or orientation_state["holdout_id"] != holdout_id):
            raise ValueError("CTF checkpoint is missing or belongs to another fold/schema")
        if (orientation_state["vae_sha256"] != file_sha256(fold_root / "01_边界感知VAE" / "VAE_最终模型.pt")
                or orientation_state["image_height"] != int(config.data.image_height)
                or orientation_state["image_width"] != int(config.data.image_width)):
            raise ValueError("VAE or image size differs from orientation checkpoint")
        orientation_head = LatentOrientationHead(**orientation_state["model_config"]).to(device)
        orientation_head.load_state_dict(orientation_state["model"])
        orientation_head.eval().requires_grad_(False)
    model_config = diffusion_state["model_config"]
    diffusion = ConditionalLatentUNet(
        latent_channels=int(model_config["latent_channels"]),
        condition_dim=condition.shape[1],
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=tuple(model_config["channel_multipliers"]),
        attention_heads=int(model_config["attention_heads"]),
    ).to(device)
    diffusion.load_state_dict(diffusion_state["model"])
    ema = ExponentialMovingAverage(diffusion, float(config.ema.decay))
    ema.load_state_dict(diffusion_state["ema"])
    ema.copy_to(diffusion)
    diffusion.eval().requires_grad_(False)
    descriptor_head = None
    descriptor_transform = None
    if diffusion_state.get("descriptor_head") is not None:
        descriptor_config = diffusion_state.get("descriptor_head_config") or {
            "latent_channels": 3,
            "hidden_dim": 256,
            "output_dim": len(DESCRIPTOR_COLUMNS),
        }
        descriptor_head = LatentDescriptorHead(**descriptor_config).to(device)
        descriptor_head.load_state_dict(diffusion_state["descriptor_head"])
        descriptor_head.eval().requires_grad_(False)
        if diffusion_state.get("descriptor_transform") is not None:
            descriptor_transform = DescriptorTargetTransform.from_state_dict(
                diffusion_state["descriptor_transform"]
            )
    schedule = DiffusionSchedule(int(config.diffusion.training_timesteps)).to(device)

    output_count = int(config.inference.output_images)
    seeds = list(config.inference.seeds)
    if len(seeds) < output_count:
        raise ValueError("推理种子数量少于输出图像数")
    output_name = Path(composition_csv).stem if composition_csv else holdout_id
    stage_name = str(getattr(config.inference, "checkpoint_stage", "feedback"))
    output_dir = PREDICTION_ROOT / holdout_id / stage_name
    if composition_csv:
        output_dir = output_dir / output_name
    if orientation_enabled:
        output_dir = PREDICTION_ROOT / "CTF_orientation_v1" / holdout_id
        if composition_csv:
            output_dir = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_targets, full_curves, generated_images = [], [], []
    orientation_predictions = []
    composition_tensor = torch.from_numpy(composition.astype(np.float32)).to(device)
    descriptor_tensor = torch.from_numpy(descriptor_mean).to(device)
    baseline_tensor = torch.from_numpy(curve_baseline).to(device)
    with torch.no_grad():
        for index in range(output_count):
            generator = torch.Generator(device=device).manual_seed(int(seeds[index]))
            latent = schedule.ddim_sample(
                diffusion,
                (
                    1,
                    vae.latent_channels,
                    int(config.data.image_height) // vae.downsample_factor,
                    int(config.data.image_width) // vae.downsample_factor,
                ),
                condition,
                int(config.inference.sampling_steps),
                float(config.inference.guidance_scale),
                device,
                generator,
            )
            image = vae.decode(latent / latent_scale)
            if orientation_head is not None:
                predicted_orientation = orientation_head(latent)[0].cpu()
                reencoded, _, _ = vae.encode(image, sample=False)
                reencoded_orientation = orientation_head(reencoded * latent_scale)[0].cpu()
                orientation_predictions.append((predicted_orientation, reencoded_orientation))
                pd.DataFrame({"statistic": ORIENTATION_COLUMNS,
                              "latent_proxy_fraction": predicted_orientation.numpy(),
                              "reencoded_image_proxy_fraction": reencoded_orientation.numpy()}).to_csv(
                    output_dir / f"生成EBSD_{index + 1:02d}_取向头代理预测.csv", index=False, encoding="utf-8-sig")
            print(f"[{holdout_id} {stage_name}] 生成 {index+1}/{output_count}，seed={seeds[index]}", flush=True)
            generated_images.append(image[:, :3].float().cpu()[0].permute(1, 2, 0).clamp(-1, 1))
            image_path = output_dir / f"生成EBSD_{index + 1:02d}.png"
            save_image(image[:, :3], image_path, normalize=True, value_range=(-1, 1))
            save_image(image[:, 3:4], output_dir / f"生成GB_{index+1:02d}.png", normalize=True, value_range=(-1, 1))
            np.save(output_dir / f"生成四通道_{index+1:02d}.npy", image[0].float().cpu().numpy())
            mechanics_output = mechanics.predict_curve(
                image, composition_tensor, descriptor_tensor, baseline_tensor
            )
            targets = mechanics_output["curve"].float().cpu().numpy()[0]
            all_targets.append(targets)
            if descriptor_head is not None and descriptor_transform is not None:
                generated_descriptors = descriptor_transform.inverse(
                    descriptor_head(image[:, :3])
                ).float().cpu().numpy()[0]
                pd.DataFrame(
                    {"descriptor": DESCRIPTOR_COLUMNS, "predicted_value": generated_descriptors,
                     "input_gpr_mean": descriptor_mean[0], "input_gpr_std": descriptor_std[0],
                     "signed_error_to_condition": generated_descriptors-descriptor_mean[0],
                     "interpretation": "frozen image proxy, NOT independent physical measurement"}
                ).to_csv(
                    output_dir / f"生成EBSD_{index + 1:02d}_{len(DESCRIPTOR_COLUMNS)}个描述符.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            pd.DataFrame(
                {"target": CURVE_TARGET_COLUMNS, "predicted_value": targets}
            ).to_csv(output_dir / f"生成EBSD_{index + 1:02d}_9个曲线目标.csv", index=False, encoding="utf-8-sig")
            strain, stress = reconstruct_curve(targets)
            full_curves.append((strain, stress))
            pd.DataFrame({"engineering_strain_pct": strain, "stress_MPa": stress}).to_csv(
                output_dir / f"生成EBSD_{index + 1:02d}_应力应变曲线.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # Outer-test CTF labels are read only after ALL generation is complete.
    orientation_evaluation = "not_requested"
    if orientation_enabled and composition_csv is None:
        try:
            true_rows = load_orientation_labels(label_path(config), [holdout_id])
            if true_rows.ipf_axis.iloc[0] != orientation_state["ipf_axis"]:
                raise ValueError("Evaluation IPF axis differs from checkpoint")
            truth = torch.tensor(true_rows[list(ORIENTATION_COLUMNS)].to_numpy("float32")[0])
            true_image_path = AlloyRepository().row(holdout_id)["image_path"]
            if file_sha256(true_image_path) != true_rows.image_sha256.iloc[0]:
                raise ValueError("Held-out image changed after registration review")
            with Image.open(true_image_path) as source:
                real_tensor = image_to_tensor(resize_full_frame(source, int(config.data.image_height),
                                                               int(config.data.image_width)))[None].to(device)
            with torch.no_grad():
                real_latent, _, _ = vae.encode(real_tensor, sample=False)
                real_proxy = orientation_head(real_latent * latent_scale)[0].cpu()
            pd.DataFrame({"statistic": ORIENTATION_COLUMNS, "true_ctf_fraction": truth.numpy(),
                          "real_image_proxy_fraction": real_proxy.numpy(),
                          "absolute_error_percentage_points": ((real_proxy - truth).abs() * 100).numpy()}).to_csv(
                output_dir / "真实EBSD_取向头校验.csv", index=False, encoding="utf-8-sig")
            comparison, scores = [], []
            for index, (latent_prediction, image_prediction) in enumerate(orientation_predictions, 1):
                for column, actual, predicted, reencoded in zip(ORIENTATION_COLUMNS, truth, latent_prediction, image_prediction):
                    comparison.append({"image": index, "statistic": column,
                                       "true_ctf_fraction": float(actual), "latent_proxy_fraction": float(predicted),
                                       "reencoded_image_proxy_fraction": float(reencoded),
                                       "absolute_error_percentage_points": 100 * float(abs(predicted - actual))})
                scores.append({"image": index, "latent_proxy_JS": float(js_divergence(latent_prediction, truth)),
                               "reencoded_image_proxy_JS": float(js_divergence(image_prediction, truth)),
                               "latent_proxy_TV": float(abs(latent_prediction - truth).sum() / 2)})
            pd.DataFrame(comparison).to_csv(output_dir / "取向头与真实CTF对比.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(scores).to_csv(output_dir / "取向代理评价.csv", index=False, encoding="utf-8-sig")
            orientation_evaluation = "proxy_only_not_measured_orientation"
        except (FileNotFoundError, ValueError) as exc:
            orientation_evaluation = f"not_evaluated: {exc}"
            print(orientation_evaluation, flush=True)

    common_end = min(curve[0][-1] for curve in full_curves)
    common_strain = np.arange(0.0, common_end, 0.05)
    if len(common_strain) == 0 or not np.isclose(common_strain[-1], common_end):
        common_strain = np.append(common_strain, common_end)
    stress_matrix = np.vstack(
        [np.interp(common_strain, strain, stress) for strain, stress in full_curves]
    )
    summary = pd.DataFrame(
        {
            "engineering_strain_pct": common_strain,
            "stress_median_MPa": np.median(stress_matrix, axis=0),
            "stress_min_MPa": np.min(stress_matrix, axis=0),
            "stress_max_MPa": np.max(stress_matrix, axis=0),
        }
    )
    summary.to_csv(output_dir / "三张图_曲线中位数与范围.csv", index=False, encoding="utf-8-sig")
    figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
    for index, (strain, stress) in enumerate(full_curves):
        axis.plot(strain, stress, alpha=0.45, label=f"Generated {index + 1}")
        axis.scatter(strain[-1], stress[-1], s=24, zorder=3)
    axis.plot(common_strain, summary["stress_median_MPa"], color="black", linewidth=2, label="Median")
    axis.fill_between(
        common_strain,
        summary["stress_min_MPa"],
        summary["stress_max_MPa"],
        color="gray",
        alpha=0.2,
        label="Generated range",
    )
    axis.set_xlabel("Engineering strain (%)")
    axis.set_ylabel("Stress (MPa)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "三张生成图对应曲线.png")
    plt.close(figure)

    if composition_csv is None:
        real_curve_table = pd.read_csv(
            MODELING_DATA_ROOT
            / "04_应力应变曲线"
            / "22个合金中位数曲线长表.csv"
        )
        real_curve_table["alloy_id"] = real_curve_table["alloy_id"].map(
            canonical_alloy_id
        )
        real_rows = real_curve_table[real_curve_table["alloy_id"] == holdout_id]
        real_strain = real_rows["engineering_strain_pct"].to_numpy(dtype=float)
        real_stress = real_rows["median_stress_MPa"].to_numpy(dtype=float)
        comparison_rows = []
        comparison_figure, comparison_axis = plt.subplots(figsize=(8, 5), dpi=180)
        comparison_axis.plot(
            real_strain, real_stress, color="black", linewidth=2.4, label="Real median"
        )
        for index, (strain, stress) in enumerate(full_curves):
            comparison_axis.plot(
                strain, stress, linewidth=1.5, alpha=0.75, label=f"Generated {index + 1}"
            )
            comparison_axis.scatter(strain[-1], stress[-1], s=26, zorder=3)
            evaluation_end = min(float(real_strain[-1]), float(strain[-1]))
            evaluation_grid = real_strain[real_strain <= evaluation_end]
            predicted_on_real_grid = np.interp(evaluation_grid, strain, stress)
            actual_on_grid = np.interp(evaluation_grid, real_strain, real_stress)
            error = predicted_on_real_grid - actual_on_grid
            comparison_rows.append(
                {
                    "generated_image": index + 1,
                    "evaluation_end_strain_pct": evaluation_end,
                    "curve_RMSE_MPa": float(np.sqrt(np.mean(error**2))),
                    "curve_MAE_MPa": float(np.mean(np.abs(error))),
                    "predicted_endpoint_strain_pct": float(strain[-1]),
                    "real_endpoint_strain_pct": float(real_strain[-1]),
                }
            )
        comparison_axis.set_xlabel("Engineering strain (%)")
        comparison_axis.set_ylabel("Stress (MPa)")
        comparison_axis.grid(alpha=0.2)
        comparison_axis.legend()
        comparison_figure.tight_layout()
        comparison_figure.savefig(output_dir / "真实与生成应力应变曲线对比.png")
        plt.close(comparison_figure)
        pd.DataFrame(comparison_rows).to_csv(
            output_dir / "真实与生成曲线评价指标.csv",
            index=False,
            encoding="utf-8-sig",
        )

        real_row = AlloyRepository().row(holdout_id)
        truth = real_row[DESCRIPTOR_COLUMNS].to_numpy(float)
        pd.DataFrame({"descriptor": DESCRIPTOR_COLUMNS, "truth": truth,
            "gpr_mean": descriptor_mean[0], "gpr_std": descriptor_std[0],
            "signed_relative_error_pct": np.divide((descriptor_mean[0]-truth)*100, np.abs(truth),
                out=np.full_like(truth, np.nan), where=np.abs(truth)>1e-12)}).to_csv(
                output_dir / "GPR描述符与真实值.csv", index=False, encoding="utf-8-sig")
        descriptor_evaluations = []
        if descriptor_head is not None:
            for i in range(output_count):
                proxy_table = pd.read_csv(output_dir / f"生成EBSD_{i+1:02d}_{len(DESCRIPTOR_COLUMNS)}个描述符.csv")
                for j, n in enumerate(DESCRIPTOR_COLUMNS):
                    predicted = float(proxy_table.predicted_value.iloc[j])
                    descriptor_evaluations.append({"image": i+1, "seed": seeds[i], "descriptor": n,
                        "proxy_prediction": predicted, "truth": truth[j], "gpr_mean": descriptor_mean[0,j],
                        "signed_error_to_truth": predicted-truth[j],
                        "signed_relative_error_to_truth_pct": 100*(predicted-truth[j])/abs(truth[j]) if abs(truth[j])>1e-12 else np.nan,
                        "signed_relative_error_to_condition_pct": 100*(predicted-descriptor_mean[0,j])/abs(descriptor_mean[0,j]) if abs(descriptor_mean[0,j])>1e-12 else np.nan})
            pd.DataFrame(descriptor_evaluations).to_csv(output_dir / "图像描述符代理对比.csv", index=False, encoding="utf-8-sig")
        target_truth = real_row[CURVE_TARGET_COLUMNS].to_numpy(float)
        pd.DataFrame([{"image": i+1, "target": n, "predicted": float(targets[j]), "truth": target_truth[j],
            "signed_relative_error_pct": 100*(targets[j]-target_truth[j])/abs(target_truth[j]) if abs(target_truth[j])>1e-12 else np.nan}
            for i, targets in enumerate(all_targets) for j,n in enumerate(CURVE_TARGET_COLUMNS)]).to_csv(
                output_dir / "9个曲线参数与真实值.csv", index=False, encoding="utf-8-sig")
        real_image = plt.imread(real_row["image_path"])
        if orientation_enabled:
            real_histogram = torch.from_numpy(rgb_histogram(real_image[..., :3]))
            rgb_scores = []
            for index, generated in enumerate(generated_images, 1):
                generated_histogram = torch.from_numpy(rgb_histogram((generated.numpy() + 1) / 2))
                rgb_scores.append({"image": index, "RGB_histogram_JS": float(js_divergence(generated_histogram, real_histogram)),
                                   "RGB_histogram_TV": float((generated_histogram - real_histogram).abs().sum() / 2),
                                   "note": "Image-only color distribution, not an ODF/IPF orientation measurement"})
            pd.DataFrame(rgb_scores).to_csv(output_dir / "独立RGB颜色分布评价.csv", index=False, encoding="utf-8-sig")
        image_figure, image_axes = plt.subplots(
            1,
            output_count + 1,
            figsize=(4 * (output_count + 1), 3.2),
            dpi=180,
        )
        image_axes[0].imshow(real_image)
        image_axes[0].set_title("Real EBSD")
        for index, generated in enumerate(generated_images):
            rendered = (generated.numpy() + 1.0) / 2.0
            if rendered.shape[-1] == 1:
                rendered = rendered[..., 0]
            image_axes[index + 1].imshow(rendered, cmap="gray" if rendered.ndim == 2 else None)
            image_axes[index + 1].set_title(f"Generated {index + 1}")
        for image_axis in image_axes:
            image_axis.axis("off")
        image_figure.tight_layout()
        image_figure.savefig(output_dir / "真实与生成EBSD对比.png")
        plt.close(image_figure)
    metadata = {
        "source": source_name,
        "outer_holdout_fold": holdout_id,
        "composition": dict(zip(COMPOSITION_COLUMNS, composition[0].tolist())),
        "descriptor_prediction_mean": dict(zip(DESCRIPTOR_COLUMNS, descriptor_mean[0].tolist())),
        "descriptor_prediction_std": dict(zip(DESCRIPTOR_COLUMNS, descriptor_std[0].tolist())),
        "seeds": seeds[:output_count],
        "selection_rule": "固定三个种子直接输出，未使用真实EBSD或真实曲线挑选",
        "evaluation_rule": "真实EBSD和真实曲线只在生成完成后用于评价，不参与训练、选图或权重更新",
        "diffusion_checkpoint": str(diffusion_path),
        "descriptor_head_enabled": descriptor_head is not None,
        "descriptor_head_kind": "frozen decoded-IPF image proxy; not direct EBSD measurement",
        "orientation_head_enabled": orientation_enabled,
        "orientation_evaluation": orientation_evaluation,
        "orientation_limitation": "Proxy predictions, NOT measured orientations/ODF of generated PNG",
    }
    (output_dir / "预测说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir
