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

import torch

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

    def test_round2_raises_only_pixel_l1_and_keeps_mechanics_off(self):
        source = VaeLossWeights()
        self.assertEqual(float(source.rgb), 1.0)
        self.assertEqual(train_round2.LOSS_IMAGE, 4.0)
        self.assertNotEqual(train_round2.LOSS_IMAGE, float(source.rgb))
        self.assertEqual(train_round2.LOSS_EDGE, float(source.edge))
        self.assertEqual(train_round2.LOSS_BOUNDARY, float(source.boundary))
        self.assertEqual(train_round2.LOSS_HAAR, float(source.haar))
        self.assertEqual(train_round2.structure_loss_weights(), {
            "image": 4.0,
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
            + 4.0 * parts["image"]
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

    def test_epoch_cap_allows_resume_from_400(self):
        self.assertEqual(train_round2.resolve_epoch_range(200, 200), (200, 400))
        self.assertEqual(train_round2.resolve_epoch_range(400, 200), (400, 600))
        self.assertEqual(train_round2.resolve_epoch_range(400, 1), (400, 401))
        self.assertEqual(train_round2.resolve_epoch_range(250, 150), (250, 400))
        self.assertEqual(train_round2.resolve_epoch_range(200, 1), (200, 201))
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(200, 201)
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(199, 1)
        with self.assertRaises(RuntimeError):
            train_round2.resolve_epoch_range(400, 201)
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

    def test_pixel_target_is_clean_latent_decode(self):
        source = inspect.getsource(train_round2._optimizer_step)
        self.assertIn("vae.decode(latent / LATENT_SCALE)", source)
        self.assertNotIn("Image.open", source)
        self.assertNotIn(".bmp", source)
        self.assertNotIn(".png", source)
        self.assertNotIn("ID03", source)

    def test_sample_dir_does_not_overwrite_existing_spot_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "ID03抽查").mkdir()
            (log_dir / "ID03抽查_第二轮").mkdir()
            first = train_round2.heldout_sample_dir(log_dir, 600)
            self.assertEqual(first.name, "ID03抽查_epoch600")
            self.assertFalse(first.exists())
            first.mkdir()
            second = train_round2.heldout_sample_dir(log_dir, 600)
            self.assertNotEqual(second, first)
            self.assertFalse(second.exists())
            self.assertNotIn(second.name, train_round2.RESERVED_SAMPLE_DIRS)

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
