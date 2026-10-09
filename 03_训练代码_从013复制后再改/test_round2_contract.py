# -*- coding: utf-8 -*-
"""Contract checks for round 2. Does not read ID03 or run the mechanics surrogate."""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "013代码" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

import train_round1
import train_round2
import window_condition
from ebsd_feedback.losses import VaeLossWeights, boundary_loss, edge_loss, haar_loss, ssim_loss
from ebsd_feedback.models.descriptor_head import DescriptorTargetTransform, LatentDescriptorHead
from ebsd_feedback.training.descriptor_guidance import descriptor_consistency
from ebsd_feedback.training.image_descriptor import sha256
from ebsd_feedback.models.vae import BoundaryAwareVAE


class Round2Contract(unittest.TestCase):
    def test_round1_noise_only_constants_unchanged(self):
        self.assertEqual(train_round1.MECHANICS_LOSS_WEIGHT, 0.0)
        self.assertEqual(train_round1.LOSS_NOISE, 1.0)
        self.assertEqual(train_round1.LOSS_IMAGE, 0.0)
        self.assertEqual(train_round1.LOSS_EDGE, 0.0)
        self.assertEqual(train_round1.LOSS_BOUNDARY, 0.0)
        self.assertEqual(train_round1.LOSS_HAAR, 0.0)
        self.assertEqual(train_round1.LOSS_DESCRIPTOR, 0.0)
        self.assertEqual(train_round1.LOSS_UNROLLED_DESCRIPTOR, 0.0)
        self.assertEqual(train_round1.LOSS_OVERLAY, 0.0)
        source = (ROOT / "train_round1.py").read_text(encoding="utf-8")
        self.assertIn("MECHANICS_LOSS_WEIGHT = 0.0", source)
        self.assertIn("LOSS_IMAGE = 0.0", source)
        self.assertNotIn("train_round2", source)
        self.assertNotIn("import sample_id03", source)

    def test_round2_uses_vae_pixel_weight_and_keeps_mechanics_off(self):
        source = VaeLossWeights()
        self.assertEqual(float(source.rgb), 1.0)
        self.assertEqual(train_round2.LOSS_IMAGE, 1.0)
        self.assertEqual(train_round2.LOSS_IMAGE, float(source.rgb))
        self.assertNotEqual(train_round2.LOSS_IMAGE, 4.0)
        self.assertEqual(train_round2.LOSS_EDGE, float(source.edge))
        self.assertEqual(train_round2.LOSS_BOUNDARY, 0.5)
        self.assertNotEqual(train_round2.LOSS_BOUNDARY, float(source.boundary))
        self.assertEqual(train_round2.LOSS_HAAR, float(source.haar))
        self.assertEqual(train_round2.LOSS_SSIM, 0.0)
        self.assertNotEqual(train_round2.LOSS_SSIM, float(source.ssim))
        self.assertEqual(train_round2.LOSS_DESCRIPTOR, 0.0)
        self.assertEqual(list(train_round2.structure_loss_weights()), list(train_round2.STRUCTURE_TERM_KEYS))
        checked = train_round2.check_structure_weights()
        self.assertEqual(checked["boundary"], 0.5)
        self.assertEqual(checked["ssim"], 0.0)
        self.assertEqual(checked["descriptor"], 0.0)
        self.assertEqual(checked["grain_coherence"], 0.1)
        self.assertEqual(train_round2.LOSS_GRAIN_COHERENCE, 0.1)
        for ssim_weight, descriptor_weight in ((0.25, 0.25), (0.1, 0.1), (0.1, 0.0), (0.0, 0.1), (0.0, 0.25)):
            rejected = dict(checked)
            rejected["ssim"] = ssim_weight
            rejected["descriptor"] = descriptor_weight
            with self.assertRaises(RuntimeError) as still_on:
                train_round2.check_structure_weights(rejected)
            self.assertIn("必须是 0", str(still_on.exception))
        old_boundary = dict(checked)
        old_boundary["boundary"] = 0.25
        with self.assertRaises(RuntimeError) as quarter_boundary:
            train_round2.check_structure_weights(old_boundary)
        self.assertIn("0.5", str(quarter_boundary.exception))
        off_coherence = dict(checked)
        off_coherence["grain_coherence"] = 0.0
        with self.assertRaises(RuntimeError) as missing_coherence:
            train_round2.check_structure_weights(off_coherence)
        self.assertIn("0.1", str(missing_coherence.exception))
        with self.assertRaises(RuntimeError) as stale:
            train_round2.check_structure_weights({
                "image": 1.0,
                "edge": 0.5,
                "boundary": 0.25,
                "haar": 0.25,
            })
        self.assertIn("结构损失项不对", str(stale.exception))
        main_source = inspect.getsource(train_round2.main)
        self.assertIn("check_structure_weights()", main_source)
        self.assertNotIn('["image", "edge", "boundary", "haar"]', main_source)
        self.assertEqual(train_round2.structure_loss_weights(), {
            "image": 1.0,
            "edge": 0.5,
            "boundary": 0.5,
            "haar": 0.25,
            "ssim": 0.0,
            "descriptor": 0.0,
            "grain_coherence": 0.1,
        })
        self.assertEqual(train_round2.LOSS_NOISE, 1.0)
        self.assertEqual(train_round2.MECHANICS_LOSS_WEIGHT, 0.0)
        self.assertEqual(train_round2.LOSS_UNROLLED_DESCRIPTOR, 0.0)
        self.assertEqual(train_round2.LOSS_OVERLAY, 0.0)
        self.assertEqual(train_round2.LOSS_ORIENTATION, 0.0)
        self.assertEqual(float(source.boundary), 0.25)
        self.assertEqual(float(source.ssim), 0.25)
        self.assertEqual(float(source.flatness), 0.05)
        self.assertEqual(float(source.kl), 1e-6)
        text = (ROOT / "train_round2.py").read_text(encoding="utf-8")
        self.assertNotIn("EbsdMechanicsSurrogate", text)
        self.assertNotIn("intragranular_flatness_loss(", text)
        self.assertIs(train_round2.edge_loss, edge_loss)
        self.assertIs(train_round2.boundary_loss, boundary_loss)
        self.assertIs(train_round2.haar_loss, haar_loss)
        self.assertIs(train_round2.ssim_loss, ssim_loss)
        self.assertIs(train_round2.descriptor_consistency, descriptor_consistency)
        self.assertIn(
            "ssim_loss(decoded[:, :3], target[:, :3])",
            inspect.getsource(train_round2.image_structure_losses),
        )
        self.assertIn(
            "grain_coherence_loss(decoded, target)",
            inspect.getsource(train_round2.image_structure_losses),
        )

    def test_structure_total_is_noise_plus_four_terms(self):
        decoded = torch.rand(2, 4, 32, 32, requires_grad=True)
        target = torch.rand(2, 4, 32, 32)
        parts = train_round2.image_structure_losses(decoded, target)
        parts["descriptor"] = decoded.sum() * 0.0
        self.assertEqual(set(parts), set(train_round2.STRUCTURE_TERM_KEYS))
        noise = torch.tensor(0.3)
        total = train_round2.diffusion_structure_total(noise, parts)
        expected = (
            1.0 * noise
            + 1.0 * parts["image"]
            + 0.5 * parts["edge"]
            + 0.5 * parts["boundary"]
            + 0.25 * parts["haar"]
            + 0.0 * parts["ssim"]
            + 0.0 * parts["descriptor"]
            + 0.1 * parts["grain_coherence"]
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(float(decoded.grad.abs().sum()), 0.0)

    def test_grain_coherence_penalizes_interior_color_variance(self):
        interior = torch.zeros(1, 4, 32, 32)
        interior[:, 3] = 1.0
        flat = interior.clone()
        speckled = interior.clone()
        speckled[:, 0, :, ::2] = 1.0
        speckled[:, 0, :, 1::2] = -1.0
        boundary_target = torch.zeros(1, 4, 32, 32)
        boundary_target[:, 3] = -1.0
        flat_loss = train_round2.grain_coherence_loss(flat, interior)
        speckle_loss = train_round2.grain_coherence_loss(speckled, interior)
        masked_loss = train_round2.grain_coherence_loss(speckled, boundary_target)
        self.assertLess(float(flat_loss), 1e-5)
        self.assertGreater(float(speckle_loss), 0.05)
        self.assertLess(float(masked_loss), float(speckle_loss) * 0.05)
        prediction = speckled.detach().clone().requires_grad_(True)
        train_round2.grain_coherence_loss(prediction, interior).backward()
        self.assertGreater(float(prediction.grad[:, :3].abs().sum()), 0.0)
        self.assertEqual(float(prediction.grad[:, 3].abs().sum()), 0.0)
        sample = (ROOT / "sample_id03.py").read_text(encoding="utf-8")
        self.assertIn('"--guidance"', sample)
        self.assertIn("GUIDANCE_SCALE = 2.0", sample)

    def test_frozen_vae_decode_passes_latent_grad_not_parameter_grad(self):
        vae = BoundaryAwareVAE(base_channels=8, channel_multipliers=(1, 2), latent_channels=4).eval()
        vae.requires_grad_(False)
        latent = torch.randn(1, 4, 8, 8, requires_grad=True)
        decoded = vae.decode(latent)
        parts = train_round2.image_structure_losses(decoded, torch.zeros_like(decoded))
        parts["descriptor"] = decoded.sum() * 0.0
        total = train_round2.diffusion_structure_total(decoded.new_zeros(()), parts)
        total.backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))

    def test_epoch_cap_resumes_only_from_round1(self):
        self.assertEqual(train_round2.TARGET_OPTIMIZER_STEPS, 20000)
        self.assertEqual(train_round2.UNUSED_LONG_RUN_STEPS, 80000)
        self.assertEqual(train_round2.optimizer_steps_per_epoch(16), 21)
        self.assertEqual(train_round2.optimizer_steps_per_epoch(8), 42)
        self.assertEqual(train_round2.default_extra_epochs(16), 952)
        self.assertEqual(train_round2.default_extra_epochs(8), 476)
        self.assertEqual(952 * 21, 19992)
        self.assertEqual(476 * 42, 19992)
        self.assertLessEqual(952 * 21, 20000)
        self.assertLessEqual(476 * 42, 20000)
        self.assertGreater(953 * 21, 20000)
        self.assertGreater(477 * 42, 20000)
        self.assertEqual(train_round2.resolve_epoch_range(1152, 952, 16), (1152, 2104))
        self.assertEqual(train_round2.resolve_epoch_range(1152, 476, 8), (1152, 1628))
        self.assertEqual(train_round2.resolve_epoch_range(200, 952, 16), (200, 1152))
        self.assertEqual(train_round2.resolve_epoch_range(250, 1, 16), (250, 251))
        train_round2.assert_round1_checkpoint(Path("checkpoint_epoch1152.pt"), 1152)
        train_round2.assert_round1_checkpoint(Path("checkpoint_epoch200.pt"), 200)
        train_round2.assert_round1_checkpoint(Path("checkpoint_epoch250.pt"), 250)
        for start in (400, 600):
            with self.assertRaises(RuntimeError):
                train_round2.resolve_epoch_range(start, 1, 16)
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(200, 953, 16)
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(200, 477, 8)
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(200, 80000, 16)
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(200, 0, 16)
        with self.assertRaises(RuntimeError):
            train_round2.assert_round1_checkpoint(Path("checkpoint_epoch400.pt"), 400)
        with self.assertRaises(RuntimeError):
            train_round2.assert_round1_checkpoint(Path("checkpoint_epoch600.pt"), 600)
        self.assertTrue(train_round2.should_rewrite_loss_log(1152, True))
        self.assertFalse(train_round2.should_rewrite_loss_log(1600, True))
        self.assertTrue(train_round2.should_rewrite_loss_log(1600, False))
        self.assertEqual(train_round2.LOSS_LOG_NAME, "train_round2_窗口条件_晶界0.5_晶粒均匀0.1.csv")
        self.assertEqual(train_round2.CONFIG_NAME, "第二轮窗口条件_晶界0.5_晶粒均匀0.1配置.json")
        self.assertNotEqual(train_round2.LOSS_LOG_NAME, "train_round2_晶界0.5_晶粒均匀0.1.csv")
        self.assertNotEqual(train_round2.LOSS_LOG_NAME, "train_round2_ssim描述符_权重0.1.csv")
        self.assertNotEqual(train_round2.LOSS_LOG_NAME, "train_round2_无ssim无描述符.csv")
        self.assertNotEqual(train_round2.LOSS_LOG_NAME, "train_round2_无ssim_描述符0.1.csv")
        self.assertNotEqual(train_round2.LOSS_LOG_NAME, "train_round2_晶界0.5_无ssim无描述符.csv")
        main_source = inspect.getsource(train_round2.main)
        self.assertIn("structure_weights=", main_source)
        self.assertIn("condition_dim=", main_source)
        self.assertIn("window_feature_statistics", main_source)
        self.assertIn("load_expanded_model", main_source)
        self.assertIn("batch_condition", inspect.getsource(train_round2._optimizer_step))
        self.assertIn("batch_condition", inspect.getsource(train_round2._val_noise_window))
        self.assertIn("ID03抽查_epoch1152", train_round2.RESERVED_SAMPLE_DIRS)
        train_round2.check_batch_size(16)
        train_round2.check_batch_size(8)
        with self.assertRaises(RuntimeError):
            train_round2.check_batch_size(32)
        parser_source = inspect.getsource(train_round2.parse_args)
        self.assertIn("default=None", parser_source)
        self.assertNotIn("MAX_EXTRA_EPOCHS", (ROOT / "train_round2.py").read_text(encoding="utf-8"))

    def test_epoch_line_prints_every_term_flushed(self):
        rows = [
            {
                "loss_total": "1.5",
                "loss_noise": "0.2",
                "loss_image": "0.4",
                "loss_edge": "0.3",
                "loss_boundary": "0.2",
                "loss_haar": "0.1",
                "loss_ssim": "0.4",
                "loss_descriptor": "0.2",
                "loss_grain_coherence": "0.08",
            },
            {
                "loss_total": "0.5",
                "loss_noise": "0.0",
                "loss_image": "0.2",
                "loss_edge": "0.1",
                "loss_boundary": "0.0",
                "loss_haar": "0.1",
                "loss_ssim": "0.2",
                "loss_descriptor": "0.0",
                "loss_grain_coherence": "0.02",
            },
        ]
        line = train_round2.format_epoch_line(201, rows, "0.12345678")
        self.assertEqual(
            line,
            "epoch 201 train_total 1.0000 noise 0.1000 image 0.3000 "
            "edge 0.2000 boundary 0.1000 haar 0.1000 ssim 0.3000 "
            "descriptor 0.1000 grain_coherence 0.0500 val_noise 0.12345678",
        )
        main = inspect.getsource(train_round2.main)
        self.assertIn("enable_live_stdout()", main)
        self.assertIn("print(format_epoch_line(epoch, epoch_rows, val_noise), flush=True)", main)
        self.assertEqual(main.count("print(gpu_utilization_line(), flush=True)"), 2)
        self.assertLess(main.index("enable_live_stdout()"), main.index("for epoch in range"))
        csv_at = main.index("csv.DictWriter")
        print_at = main.index("format_epoch_line")
        self.assertLess(csv_at, print_at)
        self.assertLess(print_at, main.rindex("print(gpu_utilization_line(), flush=True)"))

    def test_gpu_utilization_sample_does_not_raise(self):
        self.assertEqual(
            train_round2._format_gpu_query("45, 1024, 24576\n"),
            "gpu0 util 45% memory 1024/24576 MiB",
        )
        self.assertEqual(train_round2._format_gpu_query("bad"), "gpu util unavailable")
        self.assertTrue(train_round2.gpu_utilization_line())
        with unittest.mock.patch.object(train_round2, "_gpu_utilization_pynvml", side_effect=RuntimeError("nvml")):
            with unittest.mock.patch.object(
                train_round2, "_gpu_utilization_nvidia_smi", side_effect=FileNotFoundError("smi"),
            ):
                self.assertEqual(train_round2.gpu_utilization_line(), "gpu util unavailable")
        with unittest.mock.patch.object(train_round2, "_gpu_utilization_pynvml", return_value=""):
            with unittest.mock.patch.object(
                train_round2, "_gpu_utilization_nvidia_smi", return_value="gpu0 util 3% memory 1/2 MiB",
            ):
                self.assertEqual(train_round2.gpu_utilization_line(), "gpu0 util 3% memory 1/2 MiB")
        with unittest.mock.patch.object(
            train_round2, "_gpu_utilization_pynvml", return_value="gpu0 util 80% memory 10/20 MiB",
        ):
            self.assertEqual(train_round2.gpu_utilization_line(), "gpu0 util 80% memory 10/20 MiB")

    def test_paths_ignore_sample_and_memorization_modes(self):
        script = r"""
import os, sys
os.environ.pop("EBSD_013_MODE", None)
sys.path.insert(0, sys.argv[1])
sys.argv = ["sample_id03.py", "--mode", sys.argv[2], "--checkpoint", "x"]
from ebsd_feedback.paths import IMAGE_MODE, _image_mode
print(IMAGE_MODE)
print(_image_mode())
"""
        for mode in ("sample", "memorization"):
            completed = subprocess.run(
                [sys.executable, "-c", script, str(PKG), mode],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip().splitlines(), ["IPF_GB", "IPF_GB"])

    def test_low_noise_gate_keeps_noise_loss(self):
        self.assertEqual(train_round2.LOW_NOISE_FRACTION, 0.3)
        diffusion = (PKG / "ebsd_feedback" / "training" / "diffusion.py").read_text(encoding="utf-8")
        self.assertIn('descriptor_low_noise_fraction", 0.3', diffusion)
        timesteps = torch.tensor([10, 299, 300, 900])
        mask = train_round2.low_noise_mask(timesteps, 1000, 0.3)
        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertEqual(int(1000 * 0.3), 300)
        decoded = torch.zeros(4, 4, 16, 16, requires_grad=True)
        target = torch.ones(4, 4, 16, 16)
        parts = train_round2.structure_losses_for_timesteps(decoded, target, timesteps, 1000)
        parts["descriptor"] = decoded.sum() * 0.0
        total = train_round2.diffusion_structure_total(torch.tensor(0.2), parts)
        total.backward()
        self.assertIsNotNone(decoded.grad)
        self.assertGreater(float(decoded.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(decoded.grad[1].abs().sum()), 0.0)
        self.assertEqual(float(decoded.grad[2].abs().sum()), 0.0)
        self.assertEqual(float(decoded.grad[3].abs().sum()), 0.0)
        high = torch.tensor([300, 900])
        quiet = torch.rand(2, 4, 16, 16, requires_grad=True)
        quiet_parts = train_round2.structure_losses_for_timesteps(quiet, torch.rand_like(quiet), high, 1000)
        quiet_parts["descriptor"] = quiet.sum() * 0.0
        noise = torch.tensor(0.4)
        quiet_total = train_round2.diffusion_structure_total(noise, quiet_parts)
        self.assertTrue(torch.allclose(quiet_total, noise))
        for name in ("image", "edge", "boundary", "haar", "ssim", "grain_coherence"):
            self.assertEqual(float(quiet_parts[name].detach()), 0.0)

    def test_pixel_target_is_original_window(self):
        source = inspect.getsource(train_round2._optimizer_step)
        self.assertIn("low_noise_mask", source)
        self.assertIn("load_original_batch", source)
        self.assertIn("descriptor_loss_for_batch", source)
        self.assertIn("descriptor_consistency", inspect.getsource(train_round2.descriptor_loss_for_batch))
        self.assertIn("vae.decode(predicted_clean[active] / LATENT_SCALE)", source)
        self.assertNotIn("vae.decode(latent", source)
        self.assertNotIn(".bmp", source)
        self.assertNotIn("ID03", source)
        loader = inspect.getsource(train_round2.load_original_window)
        self.assertIn("image_to_tensor", loader)
        self.assertIn("Image.open", loader)
        main = inspect.getsource(train_round2.main)
        self.assertIn("assert_round1_checkpoint", main)
        self.assertNotIn("train_round2_loss.csv", main)
        self.assertNotEqual(train_round2.LOSS_LOG_NAME, "train_round2_loss.csv")
        self.assertNotEqual(train_round2.LOSS_LOG_NAME, "train_round2_原图低噪声.csv")
        self.assertIn("should_rewrite_loss_log", main)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alloy_dir = root / "ID01"
            alloy_dir.mkdir()
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            rgb[:128] = 255
            rgb[128, 0] = 50
            rgb[128, 1] = 51
            Image.fromarray(rgb, mode="RGB").save(alloy_dir / "ID01_v0_rot0_y0_x64.png")
            row = {"alloy_id": "ID01", "view": "v0_rot0", "pixel_y": "0", "pixel_x": "64"}
            path = train_round2.window_png_path(root, row)
            self.assertEqual(path, alloy_dir / "ID01_v0_rot0_y0_x64.png")
            tensor = train_round2.load_original_window(path)
            self.assertEqual(tuple(tensor.shape), (4, 256, 256))
            self.assertTrue(torch.allclose(tensor[:3, 0, 0], torch.ones(3)))
            self.assertEqual(float(tensor[3, 0, 0]), 1.0)
            self.assertTrue(torch.allclose(tensor[:3, 200, 0], -torch.ones(3)))
            self.assertEqual(float(tensor[3, 200, 0]), -1.0)
            self.assertAlmostEqual(float(tensor[0, 128, 0]), 50 / 127.5 - 1.0, places=5)
            self.assertEqual(float(tensor[3, 128, 0]), -1.0)
            self.assertAlmostEqual(float(tensor[0, 128, 1]), 51 / 127.5 - 1.0, places=5)
            self.assertEqual(float(tensor[3, 128, 1]), 1.0)
            batch = train_round2.load_original_batch(root, [row])
            self.assertEqual(tuple(batch.shape), (1, 4, 256, 256))
            with self.assertRaises(RuntimeError):
                train_round2.window_png_path(root, {**row, "alloy_id": "ID03"})
            with self.assertRaises(RuntimeError):
                train_round2.load_original_window(root / "ID03" / "ID03_v0_rot0_y0_x0.png")

    def test_sample_dir_does_not_overwrite_existing_spot_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            fresh = train_round2.heldout_sample_dir(log_dir, 400)
            self.assertEqual(fresh.name, "ID03抽查_epoch400")
            blocked = train_round2.heldout_sample_dir(log_dir, 600)
            self.assertEqual(blocked.name, "ID03抽查_epoch600_1")
            kept = train_round2.heldout_sample_dir(log_dir, 1152)
            self.assertEqual(kept.name, "ID03抽查_epoch1152_1")
            finished = train_round2.heldout_sample_dir(log_dir, 2104)
            self.assertEqual(finished.name, "ID03抽查_epoch2104")
            self.assertFalse(blocked.exists())
            for name in train_round2.RESERVED_SAMPLE_DIRS:
                self.assertIn(name, ("ID03抽查", "ID03抽查_第二轮", "ID03抽查_epoch600", "ID03抽查_epoch1152"))
                self.assertNotEqual(fresh.name, name)
                self.assertNotEqual(blocked.name, name)
            (log_dir / "ID03抽查").mkdir()
            (log_dir / "ID03抽查_第二轮").mkdir()
            (log_dir / "ID03抽查_epoch600").mkdir()
            fresh.mkdir()
            again = train_round2.heldout_sample_dir(log_dir, 400)
            self.assertEqual(again.name, "ID03抽查_epoch400_1")
            self.assertFalse(again.exists())
            self.assertNotIn(again.name, train_round2.RESERVED_SAMPLE_DIRS)

    def test_smoke_writes_neither_checkpoints_nor_images(self):
        self.assertFalse(train_round2.writes_checkpoints(True))
        self.assertFalse(train_round2.writes_id03_images(True))
        self.assertTrue(train_round2.writes_checkpoints(False))
        self.assertTrue(train_round2.writes_id03_images(False))
        smoke = inspect.getsource(train_round2._smoke_step)
        self.assertNotIn("save_checkpoint", smoke)
        self.assertNotIn("write_heldout_samples", smoke)
        self.assertNotIn("sample_id03", smoke)
        main = inspect.getsource(train_round2.main)
        smoke_at = main.index("if args.smoke:")
        sample_at = main.index("write_heldout_samples")
        self.assertLess(smoke_at, sample_at)
        self.assertIn("return", main[smoke_at:sample_at])

    def test_checkpoint_argument_is_required(self):
        parser_source = inspect.getsource(train_round2.parse_args)
        self.assertIn("required=True", parser_source)
        self.assertNotIn("checkpoint_epoch200.pt", parser_source.split("help=")[0])
        self.assertNotIn("checkpoint_epoch400.pt", parser_source.split("help=")[0])

    def test_descriptor_uses_window_values_and_requires_matching_proxy(self):
        rows = [
            {
                "D50": "5.0",
                "log_spread": "0.3",
                "area_weighted_aspect": "1.8",
                "coarse_area_fraction": "0.95",
                "boundary_length_density": "0.02",
            },
            {
                "D50": "",
                "log_spread": "0.3",
                "area_weighted_aspect": "1.8",
                "coarse_area_fraction": "0.95",
                "boundary_length_density": "0.02",
            },
            {
                "D50": "nan",
                "log_spread": "0.3",
                "area_weighted_aspect": "1.8",
                "coarse_area_fraction": "0.95",
                "boundary_length_density": "0.02",
            },
        ]
        values, present = train_round2.window_descriptor_batch(rows)
        self.assertEqual(present.tolist(), [True, False, False])
        self.assertTrue(torch.allclose(
            values[0],
            torch.tensor([5.0, 0.3, 1.8, 0.95, 0.02]),
        ))
        self.assertTrue(torch.isfinite(values[1]).all())
        self.assertTrue(torch.isfinite(values[2]).all())
        with self.assertRaises(FileNotFoundError) as missing:
            train_round2.load_frozen_descriptor_proxy(
                Path("/tmp/no-such-016-proxy.pt"),
                Path("/tmp/no-such-016-vae.pt"),
                ["ID01"],
                torch.device("cpu"),
            )
        self.assertIn("013", str(missing.exception))
        self.assertIn("ID03", str(missing.exception))
        fitted = torch.tensor(
            [
                [5.0, 0.3, 1.8, 0.95, 0.02],
                [6.0, 0.4, 1.75, 0.96, 0.03],
                [4.5, 0.2, 1.7, 0.94, 0.01],
                [7.0, 0.5, 1.82, 0.97, 0.04],
            ],
            dtype=torch.float32,
        )
        transform = DescriptorTargetTransform.fit(fitted)
        head = LatentDescriptorHead(latent_channels=3, hidden_dim=16, output_dim=5).eval()
        head.requires_grad_(False)
        decoded = torch.rand(2, 4, 32, 32, requires_grad=True)
        loss = train_round2.descriptor_loss_for_batch(
            torch.zeros(2, 4, 4, 4),
            decoded,
            torch.tensor([True, True]),
            torch.tensor([10, 10]),
            torch.tensor([False, True]),
            fitted[:2],
            torch.tensor([True, True]),
            head,
            transform,
            1000,
        )
        loss.backward()
        self.assertGreater(float(decoded.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(decoded.grad[1].abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            fold = Path(tmp) / "ID03"
            proxy = fold / "02b_图像描述符代理" / "图像描述符_最终模型.pt"
            vae = fold / "01_边界感知VAE" / "VAE_最终模型.pt"
            proxy.parent.mkdir(parents=True)
            vae.parent.mkdir(parents=True)
            vae.write_bytes(b"vae-bytes-that-do-not-match")
            torch.save(
                {
                    "schema": "013_image_proxy_v1",
                    "holdout_id": "ID03",
                    "train_ids": ["ID01"],
                    "vae_sha256": "0" * 64,
                },
                proxy,
            )
            with self.assertRaises(RuntimeError) as mismatched:
                train_round2.load_frozen_descriptor_proxy(proxy, vae, ["ID01"], torch.device("cpu"))
            self.assertIn("VAE已变化", str(mismatched.exception))
            digest = sha256(vae)
            torch.save(
                {
                    "schema": "013_image_proxy_v1",
                    "holdout_id": "ID03",
                    "train_ids": ["ID01"],
                    "vae_sha256": digest,
                    "model_config": head.config(),
                    "model": head.state_dict(),
                    "transform": transform.state_dict(),
                },
                proxy,
            )
            loaded, loaded_transform = train_round2.load_frozen_descriptor_proxy(
                proxy, vae, ["ID01"], torch.device("cpu"),
            )
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
            self.assertEqual(tuple(loaded_transform.columns), tuple(train_round2.DESCRIPTOR_COLUMNS))

    def test_round2_module_does_not_rewrite_round1_ast(self):
        round1_tree = ast.parse((ROOT / "train_round1.py").read_text(encoding="utf-8"))
        assigns = [
            node.targets[0].id
            for node in round1_tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        ]
        self.assertIn("MECHANICS_LOSS_WEIGHT", assigns)
        self.assertIn("LOSS_IMAGE", assigns)
        self.assertNotIn("train_round2", assigns)

    def test_window_condition_appends_position_and_masks_empty_fields(self):
        self.assertEqual(window_condition.ALLOY_CONDITION_DIM, 28)
        self.assertEqual(window_condition.CONDITION_DIM, 34)
        self.assertEqual(train_round2.CONDITION_DIM, 34)
        self.assertEqual(window_condition.finite_field(None), (0.0, 0.0))
        self.assertEqual(window_condition.finite_field(""), (0.0, 0.0))
        self.assertEqual(window_condition.finite_field("  "), (0.0, 0.0))
        self.assertEqual(window_condition.finite_field("nan"), (0.0, 0.0))
        self.assertEqual(window_condition.finite_field("inf"), (0.0, 0.0))
        self.assertEqual(window_condition.finite_field("not-a-number"), (0.0, 0.0))
        self.assertEqual(window_condition.finite_field("5.5"), (5.5, 1.0))
        empty = {
            "latent_x": "64",
            "latent_y": "48",
            "D50": "",
            "log_spread": "nan",
        }
        features = window_condition.window_features(empty)
        self.assertTrue(torch.equal(features, torch.tensor([0.5, 1.0, 0.0, 0.0, 0.0, 0.0])))
        filled = {"latent_x": "0", "latent_y": "0", "D50": "5.5", "log_spread": "0.2"}
        alloy = torch.arange(28, dtype=torch.float32)
        batch = window_condition.batch_condition(
            [
                {**empty, "alloy_id": "ID01"},
                {**filled, "alloy_id": "ID01"},
            ],
            {"ID01": alloy},
        )
        self.assertEqual(tuple(batch.shape), (2, 34))
        self.assertTrue(torch.equal(batch[:, :28], alloy.repeat(2, 1)))
        self.assertEqual(float(batch[1, 30]), 5.5)
        self.assertEqual(float(batch[1, 31]), 1.0)
        mean, std = window_condition.window_feature_statistics([
            empty,
            filled,
            {"latent_x": "32", "latent_y": "16", "D50": "6.5", "log_spread": "0.4"},
            {"latent_x": "16", "latent_y": "0"},
        ])
        self.assertEqual(tuple(mean.shape), (6,))
        self.assertTrue(torch.allclose(mean[2], torch.tensor(6.0)))
        self.assertGreater(float(std[2]), 0.0)
        self.assertEqual(float(mean[3]), 0.5)
        alloy_row = {name: "0.1" for name in train_round1.COND_COLUMNS}
        alloy_row["alloy_id"] = "ID03"
        alloy_row["desc_cond_grain_size_median_um"] = "16.5"
        alloy_row["desc_cond_grain_size_log_spread"] = "0.8"
        holdout = window_condition.holdout_condition(alloy_row, 4)
        self.assertEqual(tuple(holdout.shape), (4, 34))
        self.assertEqual(window_condition.HOLDOUT_SAMPLE_ORIGINS, ((0, 0), (64, 0), (0, 48), (64, 48)))
        self.assertTrue(torch.allclose(holdout[:, 28:30], torch.tensor([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
        ])))
        self.assertTrue(torch.allclose(holdout[:, 30:34], torch.tensor([16.5, 1.0, 0.8, 1.0]).repeat(4, 1)))
        absent = window_condition.window_features({"latent_x": "0", "latent_y": "0"})
        self.assertTrue(torch.equal(absent, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        sample = (ROOT / "sample_id03.py").read_text(encoding="utf-8")
        self.assertIn("holdout_condition", sample)
        self.assertIn("batch_condition", sample)
        self.assertNotIn("ID03.png", sample)

    def test_expanded_condition_layer_keeps_old_columns_at_zero_init(self):
        saved = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        target = torch.ones(8, 6)
        grown = train_round2.expand_saved_tensor(saved, target)
        self.assertEqual(tuple(grown.shape), (8, 6))
        self.assertTrue(torch.equal(grown[:, :4], saved))
        self.assertTrue(torch.equal(grown[:, 4:], torch.zeros(8, 2)))
        same = train_round2.expand_saved_tensor(saved, saved.clone())
        self.assertTrue(torch.equal(same, saved))
        with self.assertRaises(RuntimeError):
            train_round2.expand_saved_tensor(torch.zeros(8, 7), torch.zeros(8, 6))

        class Tiny(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.condition_encoder = torch.nn.Module()
                self.condition_encoder.network = torch.nn.Sequential(
                    torch.nn.Linear(dim, 8, bias=False)
                )
                self.condition_encoder.register_buffer("input_mean", torch.zeros(dim))
                self.condition_encoder.register_buffer("input_std", torch.ones(dim))

        old = Tiny(4)
        with torch.no_grad():
            old.condition_encoder.network[0].weight.copy_(saved)
            old.condition_encoder.input_mean.copy_(torch.arange(4, dtype=torch.float32))
        new = Tiny(6)
        live_mean = torch.arange(6, dtype=torch.float32) + 10
        live_std = torch.arange(6, dtype=torch.float32) + 3
        with torch.no_grad():
            new.condition_encoder.input_mean.copy_(live_mean)
            new.condition_encoder.input_std.copy_(live_std)
        train_round2.load_expanded_model(new, old.state_dict())
        weight = new.condition_encoder.network[0].weight
        self.assertTrue(torch.equal(weight[:, :4], saved))
        self.assertTrue(torch.equal(weight[:, 4:], torch.zeros(8, 2)))
        self.assertTrue(torch.equal(new.condition_encoder.input_mean, live_mean))
        self.assertTrue(torch.equal(new.condition_encoder.input_std, live_std))
        probe = torch.randn(2, 4)
        with torch.no_grad():
            old_out = old.condition_encoder.network[0](probe)
            new_out = new.condition_encoder.network[0](torch.cat([probe, torch.randn(2, 2)], dim=1))
        self.assertTrue(torch.allclose(old_out, new_out))

        narrow = torch.nn.Linear(4, 3, bias=False)
        opt = torch.optim.AdamW(narrow.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.95))
        narrow(torch.randn(2, 4)).sum().backward()
        opt.step()
        saved_opt = opt.state_dict()
        wide = torch.nn.Linear(6, 3, bias=False)
        wide_opt = torch.optim.AdamW(wide.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.95))
        train_round2.load_expanded_optimizer(wide_opt, saved_opt)
        moment = wide_opt.state[wide.weight]["exp_avg"]
        self.assertEqual(tuple(moment.shape), (3, 6))
        self.assertTrue(torch.equal(moment[:, :4], saved_opt["state"][0]["exp_avg"]))
        self.assertTrue(torch.equal(moment[:, 4:], torch.zeros(3, 2)))
        self.assertEqual(tuple(wide_opt.state[wide.weight]["step"].shape), ())
        wide(torch.randn(2, 6)).sum().backward()
        wide_opt.step()


if __name__ == "__main__":
    unittest.main()
