from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ebsd_feedback.data import image_to_tensor, pack_image, resize_full_frame, AlloyRepository
from ebsd_feedback.models.vae import BoundaryAwareVAE
from ebsd_feedback.models.descriptor_head import LatentDescriptorHead, DescriptorTargetTransform
from ebsd_feedback.training.descriptor_guidance import descriptor_consistency, fit_fold_transform
from ebsd_feedback.losses import boundary_aware_vae_loss, VaeLossWeights, soft_boundary_map
from ebsd_feedback.constants import DESCRIPTOR_COLUMNS


class Tests013(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_rgb_not_alpha_and_mask_before_resize(self):
        a = np.zeros((16,16,3), dtype=np.uint8)
        a[:, :8] = [0,0,255]  # dark saturated BLUE must not be GB
        im = Image.fromarray(a)
        x = image_to_tensor(resize_full_frame(im, 32, 32))
        self.assertEqual(x.shape, (4,32,32))
        self.assertTrue(torch.all(x[3,:,:16] == 1))
        self.assertTrue(torch.all(x[3,:,16:] == -1))
        self.assertTrue(torch.allclose(soft_boundary_map(x[None])[0,0], (1-x[3])/2))
        packed = pack_image(im)
        self.assertEqual(packed.mode, "RGBA")
        np.testing.assert_array_equal(np.asarray(packed.convert("RGB")), a)

    def test_transform_roundtrip_physical_support_and_no_holdout_fit(self):
        import pandas as pd
        a = np.array([[10,.8,1.2,.5,.04], [20,.9,2.3,.95,.1]], np.float32)
        transform = DescriptorTargetTransform.fit(a)
        x = torch.tensor(a)
        torch.testing.assert_close(transform.inverse(transform.transform(x)), x)
        y = transform.inverse(torch.full((1,5), -5.))
        self.assertGreaterEqual(float(y[0,2]), 1.)
        frame = pd.DataFrame(np.vstack([a, a[0]*2]), columns=["desc_true_"+n for n in DESCRIPTOR_COLUMNS])
        frame["alloy_id"] = ["ID01","ID02","ID07"]
        first = fit_fold_transform(frame, ["ID01","ID02"])
        frame.loc[2, "desc_true_grain_size_median_um"] = 1e10
        second = fit_fold_transform(frame, ["ID01","ID02"])
        torch.testing.assert_close(first.mean, second.mean)

    def test_frozen_proxy_gradient_reaches_latent_not_parameters(self):
        vae = BoundaryAwareVAE(base_channels=8, channel_multipliers=(1,2), latent_channels=4).eval().requires_grad_(False)
        head = LatentDescriptorHead(3,32,5).eval().requires_grad_(False)
        transform = DescriptorTargetTransform.fit([[10,.8,1.2,.5,.04],[20,.9,2.3,.95,.1]])
        z = torch.randn(2,4,16,16,requires_grad=True)
        decoded = vae.decode(z)
        physical = torch.tensor([[12,.85,1.5,.7,.07]]).repeat(2,1)
        loss, info = descriptor_consistency(z, physical, torch.tensor([10,500]), torch.tensor([False,False]),
            head, transform, 1000,.3,.5,decoded)
        loss.backward()
        self.assertEqual(info["active"].tolist(), [True,False])
        self.assertGreater(float(z.grad[0].abs().sum()), 0)
        self.assertEqual(float(z.grad[1].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in vae.parameters()))

    def test_vae_loss_finite_and_all_four_channels_receive_gradients(self):
        vae = BoundaryAwareVAE(base_channels=8,channel_multipliers=(1,2),latent_channels=4)
        a = np.full((32,32,3), [120,230,40], dtype=np.uint8)
        a[:,16] = 0
        x = image_to_tensor(Image.fromarray(a))[None].repeat(2,1,1,1)
        output = vae(x)
        output.reconstruction.retain_grad()
        loss = boundary_aware_vae_loss(output,x,VaeLossWeights())["loss_total"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.all(output.reconstruction.grad.abs().sum((0,2,3)) > 0))

    def test_real_data_all_22(self):
        repo = AlloyRepository()
        self.assertEqual(len(repo.alloy_ids),22)
        train, held = repo.split_ids("ID17")
        self.assertEqual(held,["ID17"])
        self.assertNotIn("ID17",train)
        self.assertEqual(len(train),21)


if __name__ == "__main__":
    unittest.main()
