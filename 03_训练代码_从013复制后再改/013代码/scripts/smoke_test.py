"""Isolated tiny CPU integration test, NOT a scientific experiment or training run.

Uses synthetic fold conditions with real files to validate all stage interfaces.
Creates only disposable mock checkpoints under a temporary directory.
"""
from pathlib import Path
import os
import sys
import tempfile
import copy
import json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "05_代码" / "src"))


def main():
    import torch
    import pandas as pd
    import numpy as np
    import joblib
    from unittest.mock import patch
    torch.set_num_threads(2)
    import ebsd_feedback.paths as paths
    from ebsd_feedback.config import load_config, ConfigNode
    from ebsd_feedback.constants import COMPOSITION_COLUMNS, DESCRIPTOR_COLUMNS, CURVE_TARGET_COLUMNS
    from ebsd_feedback.data import AlloyRepository
    with patch("torch.cuda.is_available", return_value=False), tempfile.TemporaryDirectory(prefix="013_smoke_", ignore_cleanup_errors=True) as temp:
        paths.FOLD_MODEL_ROOT = Path(temp) / "models"
        paths.LOG_ROOT = Path(temp) / "logs"
        paths.PREDICTION_ROOT = Path(temp) / "predictions"
        from ebsd_feedback.folds import fit_bundle, predict_bundle
        from ebsd_feedback.training.vae import train_vae
        from ebsd_feedback.training.mechanics import train_mechanics
        from ebsd_feedback.training.image_descriptor import train_image_descriptor
        from ebsd_feedback.training.diffusion import train_diffusion
        from ebsd_feedback.inference import generate_three_images
        from ebsd_feedback.monitor import TrainingMonitor
        repo = AlloyRepository()
        held = "ID17"
        train_ids, _ = repo.split_ids(held)
        train = repo.table[repo.table.alloy_id.isin(train_ids)]
        bundle = fit_bundle(train[COMPOSITION_COLUMNS].to_numpy(), train[DESCRIPTOR_COLUMNS].to_numpy(),
                            train[CURVE_TARGET_COLUMNS].to_numpy(), seed=42, restarts=0)
        pred = predict_bundle(bundle,repo.table[COMPOSITION_COLUMNS].to_numpy())
        rows = []
        for i, (_, record) in enumerate(repo.table.iterrows()):
            r = {"alloy_id": record.alloy_id, "role": "holdout" if record.alloy_id == held else "train", "image_path": record.image_path}
            for prefix, names, vector in (("comp_", COMPOSITION_COLUMNS, record[COMPOSITION_COLUMNS]),
                ("desc_true_", DESCRIPTOR_COLUMNS, record[DESCRIPTOR_COLUMNS]),
                ("desc_cond_", DESCRIPTOR_COLUMNS, pred["descriptors"].mean[i]),
                ("desc_std_", DESCRIPTOR_COLUMNS, pred["descriptors"].std[i]),
                ("curve_true_", CURVE_TARGET_COLUMNS, record[CURVE_TARGET_COLUMNS]),
                ("curve_baseline_", CURVE_TARGET_COLUMNS, pred["curves"].mean[i])):
                r.update({prefix+n:float(v) for n,v in zip(names,vector)})
            r.update({"curve_residual_"+n: r["curve_true_"+n]-r["curve_baseline_"+n] for n in CURVE_TARGET_COLUMNS})
            rows.append(r)
        prepared = paths.FOLD_MODEL_ROOT / held / "00_折准备"
        prepared.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(prepared / "严格留一折清单.csv",index=False)
        joblib.dump(bundle, prepared / "成分到描述符与曲线_GPR.joblib")

        def tiny(name):
            c = load_config(name).to_dict()
            c["performance"].update(compile=False, channels_last=False, allow_tf32=False, matmul_precision="highest")
            c["data"].update(image_height=64,image_width=96,patch_size=32,num_workers=0,persistent_workers=False,pin_memory=False)
            c["monitor"].update(log_every=1,checkpoint_every=1,sample_every=1,tensorboard=False)
            c.setdefault("train",{}).update(max_steps=2,batch_size=2,samples_per_alloy=2,warmup_steps=1)
            if c["stage"] == "vae":
                c["model"].update(base_channels=8,channel_multipliers=[1,2])
            if c["stage"] == "mechanics":
                c["train"].update(refit_all_steps=2,validation_every=1)
                c["model"].update(image_base_channels=16,image_feature_dim=64)
            if c["stage"] == "image_descriptor":
                c["train"].update(validation_every=1)
                c["model"].update(hidden_dim=32)
            if c["stage"].startswith("diffusion") or c["stage"]=="inference":
                c["model"].update(base_channels=8,channel_multipliers=[1,2],attention_heads=2)
                c["diffusion"].update(training_timesteps=10,sampling_steps=3)
                c["loss"].update(unrolled_start_step=1,unrolled_every=1,unrolled_ddim_steps=3,descriptor_low_noise_fraction=1.)
            if "inference" in c:
                c["inference"].update(output_images=1,seeds=[42],sampling_steps=3)
            return ConfigNode(c)
        def interrupt_and_resume(function, config, interrupt_at=1):
            original_log = TrainingMonitor.log
            interrupted = False
            def log_then_interrupt(self, step, *args, **kwargs):
                nonlocal interrupted
                original_log(self, step, *args, **kwargs)
                if step == interrupt_at and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt("intentional test interruption")
            try:
                with patch.object(TrainingMonitor, "log", log_then_interrupt):
                    function(config, held)
            except KeyboardInterrupt:
                pass
            assert interrupted
            return function(config, held)

        interrupt_and_resume(train_vae, tiny("01_边界感知VAE.yaml"))
        interrupt_and_resume(train_mechanics, tiny("02_力学代理.yaml"), interrupt_at=3)
        interrupt_and_resume(train_image_descriptor, tiny("02b_图像描述符代理.yaml"))
        interrupt_and_resume(train_diffusion, tiny("03_基础条件扩散.yaml"))
        base = generate_three_images(tiny("05b_基础扩散三图推理.yaml"),held)
        interrupt_and_resume(train_diffusion, tiny("04_力学反馈微调.yaml"))
        final = generate_three_images(tiny("05_三图推理.yaml"),held)
        assert base != final and (base / "预测说明.json").exists() and (final / "图像描述符代理对比.csv").exists()
        from PIL import Image
        assert Image.open(final/"生成EBSD_01.png").mode=="RGB"
        assert Image.open(final/"生成GB_01.png").size==(96,64)
        assert np.load(final/"生成四通道_01.npy").shape==(4,64,96)
        print("PASS: isolated CPU pipeline including INTERRUPT/RESUME VAE / mechanics refit / frozen proxy / base diffusion / feedback / both inference outputs")


if __name__=="__main__":
    main()
