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
from ebsd_feedback.losses import VaeLossWeights, boundary_loss, edge_loss, haar_loss
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
        self.assertEqual(train_round2.LOSS_BOUNDARY, float(source.boundary))
        self.assertEqual(train_round2.LOSS_HAAR, float(source.haar))
        self.assertEqual(train_round2.structure_loss_weights(), {
            "image": 1.0,
            "edge": 0.5,
            "boundary": 0.25,
            "haar": 0.25,
        })
        self.assertEqual(train_round2.LOSS_NOISE, 1.0)
        self.assertEqual(train_round2.MECHANICS_LOSS_WEIGHT, 0.0)
        self.assertEqual(train_round2.LOSS_DESCRIPTOR, 0.0)
        self.assertEqual(train_round2.LOSS_UNROLLED_DESCRIPTOR, 0.0)
        self.assertEqual(train_round2.LOSS_OVERLAY, 0.0)
        self.assertEqual(train_round2.LOSS_ORIENTATION, 0.0)
        self.assertEqual(float(source.ssim), 0.25)
        self.assertEqual(float(source.flatness), 0.05)
        self.assertEqual(float(source.kl), 1e-6)
        text = (ROOT / "train_round2.py").read_text(encoding="utf-8")
        self.assertNotIn("EbsdMechanicsSurrogate", text)
        self.assertNotIn("ssim_loss(", text)
        self.assertNotIn("intragranular_flatness_loss(", text)
        self.assertIs(train_round2.edge_loss, edge_loss)
        self.assertIs(train_round2.boundary_loss, boundary_loss)
        self.assertIs(train_round2.haar_loss, haar_loss)

    def test_structure_total_is_noise_plus_four_terms(self):
        decoded = torch.rand(2, 4, 32, 32, requires_grad=True)
        target = torch.rand(2, 4, 32, 32)
        parts = train_round2.image_structure_losses(decoded, target)
        self.assertEqual(set(parts), {"image", "edge", "boundary", "haar"})
        noise = torch.tensor(0.3)
        total = train_round2.diffusion_structure_total(noise, parts)
        expected = (
            1.0 * noise
            + 1.0 * parts["image"]
            + 0.5 * parts["edge"]
            + 0.25 * parts["boundary"]
            + 0.25 * parts["haar"]
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(float(decoded.grad.abs().sum()), 0.0)

    def test_frozen_vae_decode_passes_latent_grad_not_parameter_grad(self):
        vae = BoundaryAwareVAE(base_channels=8, channel_multipliers=(1, 2), latent_channels=4).eval()
        vae.requires_grad_(False)
        latent = torch.randn(1, 4, 8, 8, requires_grad=True)
        decoded = vae.decode(latent)
        parts = train_round2.image_structure_losses(decoded, torch.zeros_like(decoded))
        total = train_round2.diffusion_structure_total(decoded.new_zeros(()), parts)
        total.backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))

    def test_epoch_cap_resumes_only_from_round1(self):
        self.assertEqual(train_round2.resolve_epoch_range(200, 200), (200, 400))
        self.assertEqual(train_round2.resolve_epoch_range(200, 1), (200, 201))
        train_round2.assert_round1_checkpoint(Path("checkpoint_epoch200.pt"), 200)
        for start in (199, 250, 400, 600):
            with self.assertRaises(RuntimeError):
                train_round2.resolve_epoch_range(start, 1)
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(200, 201)
        with self.assertRaises(RuntimeError):
            train_round2.assert_round1_checkpoint(Path("checkpoint_epoch400.pt"), 400)
        with self.assertRaises(RuntimeError):
            train_round2.assert_round1_checkpoint(Path("checkpoint_epoch600.pt"), 600)
        with self.assertRaises(RuntimeError):
            train_round2.assert_round1_checkpoint(Path("checkpoint_epoch250.pt"), 250)
        train_round2.check_batch_size(16)
        train_round2.check_batch_size(8)
        with self.assertRaises(RuntimeError):
            train_round2.check_batch_size(32)

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
        noise = torch.tensor(0.4)
        quiet_total = train_round2.diffusion_structure_total(noise, quiet_parts)
        self.assertTrue(torch.allclose(quiet_total, noise))
        for name in ("image", "edge", "boundary", "haar"):
            self.assertEqual(float(quiet_parts[name].detach()), 0.0)

    def test_pixel_target_is_original_window(self):
        source = inspect.getsource(train_round2._optimizer_step)
        self.assertIn("low_noise_mask", source)
        self.assertIn("load_original_batch", source)
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
            self.assertFalse(blocked.exists())
            for name in train_round2.RESERVED_SAMPLE_DIRS:
                self.assertIn(name, ("ID03抽查", "ID03抽查_第二轮", "ID03抽查_epoch600"))
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


if __name__ == "__main__":
    unittest.main()
