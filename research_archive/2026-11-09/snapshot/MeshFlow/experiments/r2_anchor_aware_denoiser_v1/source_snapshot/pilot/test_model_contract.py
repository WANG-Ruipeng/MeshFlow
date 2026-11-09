"""Small CPU arithmetic fixtures only; zero official loads/model forwards.

Run with python -B -m r2.pilots.anchor_aware_denoiser_v1.test_model_contract.
The real checkpoint/precision/gradient checks remain separately budgeted.
"""
import json
import unittest

import torch
from torch import nn

from .model import AnchorAwareDenoiser


class TinyDecoder(nn.Module):
    def forward(self, h, cond, mask=None):
        assert mask is None
        # Global dependence stands in for reading C from free positions.
        return (h[..., :3] + h[..., :3].mean(dim=1, keepdim=True)).reshape(1, -1, 9)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe_nerf = nn.Identity()
        self.post_quant_linear = nn.Linear(3, 6)
        with torch.no_grad():
            self.post_quant_linear.weight.copy_(torch.arange(18).reshape(6, 3).float() / 20)
            self.post_quant_linear.bias.fill_(0.2)
        self.decoder = TinyDecoder()

    def decode(self, z, cond, mask):
        return self.decoder(self.post_quant_linear(z), cond, mask)


class ContractTests(unittest.TestCase):
    def inputs(self):
        x = torch.arange(27, dtype=torch.float32).reshape(1, 3, 9) / 50
        known = torch.tensor([[True, False, False]])
        return x, known

    def test_zero_roles_equal_reference_in_both_activation_dtypes(self):
        x, known = self.inputs()
        for anchor in (False, True):
            wrapper = AnchorAwareDenoiser(TinyBase(), anchor=anchor)
            for enabled in (False, True):
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
                    expected = wrapper.official_residual(x)
                    got = wrapper(x, known, capture_role_stats=True)
                self.assertTrue(torch.equal(got, expected))
                self.assertEqual(got.dtype, expected.dtype)
                self.assertEqual(wrapper.last_role_stats["effective_add_l2"], 0)

    def test_no_role_does_not_use_mask_values(self):
        x, known = self.inputs()
        wrapper = AnchorAwareDenoiser(TinyBase(), anchor=False)
        self.assertTrue(torch.equal(wrapper(x, known), wrapper(x, ~known)))
        self.assertTrue(torch.equal(wrapper(x, known), wrapper(x, None)))
        self.assertNotIn("e_known", dict(wrapper.named_parameters()))
        wrapper.validate_parameters()

    def test_free_loss_reaches_role_known_input_and_backbone(self):
        x, known = self.inputs()
        x.requires_grad_(True)
        wrapper = AnchorAwareDenoiser(TinyBase(), anchor=True)
        residual = wrapper(x, known)
        proposal = x.float() + residual.float()
        loss = proposal[:, ~known[0]].square().mean()
        loss.backward()
        self.assertGreater(float(wrapper.e_known.grad.abs().sum()), 0)
        self.assertGreater(float(x.grad[:, known[0]].abs().sum()), 0)
        self.assertGreater(float(wrapper.base.post_quant_linear.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(wrapper.e_known.grad).all())

    def test_joint_face_corner_permutation(self):
        x, known = self.inputs()
        wrapper = AnchorAwareDenoiser(TinyBase(), anchor=True)
        with torch.no_grad():
            wrapper.e_known.fill_(0.05)
        expected = wrapper(x, known).reshape(1, 3, 3, 3)
        faces, corners = torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0])
        permuted_x = x.reshape(1, 3, 3, 3)[:, faces][:, :, corners].reshape(1, 3, 9)
        got = wrapper(permuted_x, known[:, faces]).reshape(1, 3, 3, 3)
        self.assertTrue(torch.allclose(got, expected[:, faces][:, :, corners], atol=1e-7, rtol=1e-6))

    def test_validation_rejects_wrong_N_dtype_shape(self):
        x, known = self.inputs()
        wrapper = AnchorAwareDenoiser(TinyBase(), anchor=True)
        invalid = [(x, known, torch.tensor([2])), (x.bfloat16(), known, None),
                   (x, known.float(), None), (x, None, None),
                   (x.repeat(2, 1, 1), known.repeat(2, 1), None)]
        for args in invalid:
            with self.assertRaises(ValueError):
                wrapper(*args)

    def test_zero_initialization_consumes_no_RNG(self):
        for anchor in (False, True):
            base = TinyBase()
            before = torch.get_rng_state().clone()
            wrapper = AnchorAwareDenoiser(base, anchor=anchor)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(int(torch.count_nonzero(wrapper.e_known)), 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests": result.testsRun, "status": "PASS" if result.wasSuccessful() else "FAIL",
                      "official_model_loads": 0, "official_model_forwards": 0, "GPU_calls": 0}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
