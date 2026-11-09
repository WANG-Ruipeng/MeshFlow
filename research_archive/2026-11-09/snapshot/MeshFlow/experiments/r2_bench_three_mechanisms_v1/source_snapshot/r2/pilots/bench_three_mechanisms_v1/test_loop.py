"""CPU operator tests with real DiT/native arithmetic at a reduced hidden width.

The fixture bypasses only the production width constructor and CUDA runtime
guard. Production loop.py retains both guards; no checkpoint/GPU is loaded.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import unittest
from unittest import mock

import torch
from torch import nn

from . import loop as L
from meshflow_control.models.backbone.equidit import DiT


def fixture(rounds=2):
    torch.manual_seed(18419)
    model = L.LoopNative.__new__(L.LoopNative)
    nn.Module.__init__(model)
    model.backbone = DiT(hidden_dim=12, num_heads=3, num_layers=12,
        max_length=800, gradient_checkpointing=False, class_dropout_prob=0.,
        version=3, pe_freq=2, mixed_precision="fp32", use_dit_like_pe=False,
        face_cond=True, face_bin=20, use_rmsnorm=True, use_qknorm=True)
    model.role_embedding = nn.Parameter(torch.zeros(2, 12))
    model.context_encoder = None
    model.context_observer = None
    model.readout_mode = "none"
    model.readouts = nn.ModuleDict()
    model.readout_observer = None
    model.experiment_trainable = True
    # Nonzero gates/readout are necessary: default DiT zero exits hide errors.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and parameter.ndim == 1:
                parameter.fill_(1.)
            else:
                parameter.normal_(0., .065)
    model.set_loop_rounds(rounds)
    model.eval()
    return model


def inputs():
    generator = torch.Generator().manual_seed(9043)
    x = torch.randn(1, 128, 9, generator=generator, requires_grad=True)
    known = torch.zeros(1, 128, dtype=torch.bool)
    known[:, :4] = True
    return x, torch.tensor([.63]), torch.tensor([128]), torch.ones_like(known), known


class LoopOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU tests must not inherit initialized CUDA")

    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(L, "assert_native_runtime", lambda *a, **k: None))
        self.stack.enter_context(mock.patch.object(L.native, "assert_native_runtime", lambda *a, **k: None))
        self.addCleanup(self.stack.close)

    def test_r1_exact_output_and_input_vjp_bypass(self):
        model = fixture(1)
        baseline = copy.deepcopy(model)
        baseline.__class__ = L.NativeModel
        x, t, y, valid, known = inputs()
        original = baseline(x, t, y, valid, known)
        actual = model(x, t, y, valid, known)
        self.assertTrue(torch.equal(original, actual))
        cotangent = torch.randn_like(actual)
        a = torch.autograd.grad((original * cotangent).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((actual * cotangent).sum(), x)[0]
        self.assertTrue(torch.equal(a, b))

    def test_first_loop_exactly_equals_original(self):
        model = fixture(2)
        baseline = copy.deepcopy(model)
        baseline.__class__ = L.NativeModel
        x, t, y, valid, known = inputs()
        original = baseline(x, t, y, valid, known)
        v1, v2 = model.forward_train(x, t, y, valid, known)
        self.assertTrue(torch.equal(original, v1))
        self.assertGreater(float((v2-v1).abs().max()), 1e-6)

    def test_calls_c_refresh_and_no_input_mutation(self):
        model = fixture()
        x, t, y, valid, known = inputs()
        originals = [value.detach().clone() for value in (x, t, y, valid, known)]
        records = []
        model.forward_train(x, t, y, valid, known, observer=records.append)
        blocks = [r for r in records if r["site"] == "block"]
        self.assertEqual([r["layer"] for r in blocks], list(range(1, 13)) + [9, 10, 11, 12])
        self.assertEqual([r["loop_round"] for r in blocks], [1]*12+[2]*4)
        self.assertEqual(sum(r["site"] == "final" for r in records), 2)
        entry = next(r for r in records if r["site"] == "loop_core_entry")
        mask = entry["known_corner_mask"]
        self.assertTrue(torch.equal(entry["hidden"][mask], entry["reference"][mask]))
        self.assertTrue(torch.equal(entry["hidden"][~mask], entry["previous"][~mask]))
        for record in blocks:
            self.assertTrue(torch.equal(record["known_mask"], known))
            self.assertTrue(torch.equal(record["face_time"][known], torch.ones(4)))
            self.assertTrue(torch.equal(record["face_time"][~known], t.expand(124)))
        for before, after in zip(originals, (x, t, y, valid, known)):
            self.assertTrue(torch.equal(before, after))

    def test_reference_remains_connected_and_v2_backpropagates(self):
        model = fixture()
        x, t, y, valid, known = inputs()
        records = []
        _, v2 = model.forward_train(x, t, y, valid, known, observer=records.append)
        entry = next(r for r in records if r["site"] == "loop_core_entry")
        reference = entry["reference"]
        mask = entry["known_corner_mask"]
        direct = torch.autograd.grad(entry["hidden"][mask].sum(), reference, retain_graph=True)[0]
        self.assertTrue(torch.equal(direct[mask], torch.ones_like(direct[mask])))
        self.assertTrue(torch.equal(direct[~mask], torch.zeros_like(direct[~mask])))
        loss = v2[:, 4:].square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[:, :4].abs().sum()), 0.)
        for index in (0, 7, 8, 11):
            grads = [p.grad for p in model.backbone.layers[index].parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_inference_uses_one_readout_and_matches_v2(self):
        model = fixture()
        x, t, y, valid, known = inputs()
        _, expected = model.forward_train(x, t, y, valid, known)
        records = []
        actual = model(x, t, y, valid, known, observer=records.append)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(sum(r["site"] == "block" for r in records), 16)
        self.assertEqual(sum(r["site"] == "final" for r in records), 1)

    def test_state_dict_and_parameter_order_remain_native(self):
        model = fixture()
        baseline = copy.deepcopy(model)
        baseline.__class__ = L.NativeModel
        self.assertEqual(list(model.state_dict()), list(baseline.state_dict()))
        self.assertEqual([n for n, _ in model.named_parameters()], [n for n, _ in baseline.named_parameters()])
        model.load_state_dict(baseline.state_dict(), strict=True)
        self.assertFalse(any("loop" in key for key in model.state_dict()))

    def test_deep_supervision_preserves_mask_denominator_and_both_gradients(self):
        v1 = torch.full((1, 3, 9), 2., requires_grad=True)
        v2 = torch.full((1, 3, 9), 4., requires_grad=True)
        target = torch.zeros_like(v1)
        valid = torch.tensor([[True, True, False]])
        fixed = torch.tensor([[[True]*3, [False]*3, [False]*3]])
        loss, first, second = L.deep_supervision_loss(v1, v2, target, valid, fixed, denominator=18)
        self.assertEqual(float(first), 2.)
        self.assertEqual(float(second), 8.)
        self.assertEqual(float(loss), 5.)
        loss.backward()
        for value in (v1, v2):
            self.assertTrue(torch.equal(value.grad[:, (0, 2)], torch.zeros_like(value.grad[:, (0, 2)])))
            self.assertGreater(float(value.grad[:, 1].abs().sum()), 0.)

    def test_r1_train_returns_one_computation_not_two(self):
        model = fixture(1)
        x, t, y, valid, known = inputs()
        records = []
        v1, v2 = model.forward_train(x, t, y, valid, known, observer=records.append)
        self.assertIs(v1, v2)
        self.assertEqual(sum(r["site"] == "block" for r in records), 12)
        self.assertEqual(sum(r["site"] == "final" for r in records), 1)

    def test_invalid_rounds_and_native_masks_rejected(self):
        model = fixture()
        for value in (0, 3, True, 2.):
            with self.assertRaises(ValueError):
                model.set_loop_rounds(value)
        x, t, y, valid, known = inputs()
        with self.assertRaises(ValueError):
            model.forward_train(x, t, y-1, valid, known)
        with self.assertRaises(ValueError):
            model.forward_train(x, t, y, valid, torch.zeros_like(known))


if __name__ == "__main__":
    unittest.main(verbosity=2)
