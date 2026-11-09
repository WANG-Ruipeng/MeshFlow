"""CPU-only algebra, mock-forward, autograd and clock tests; no model assets."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch
from torch import nn

from . import async_core as A


def sample(t=.35, seam=True):
    n = 4
    fixed = np.zeros((n, 3), bool)
    fixed[0] = True
    if seam:
        fixed[1, :2] = True
    valid = np.array([1, 1, 1, 0], bool)
    ids = np.where(fixed, np.arange(12).reshape(n, 3), -1)
    x0 = np.arange(36, dtype=np.float32).reshape(n, 3, 3)/10
    x1 = x0+np.float32(2)
    x1[fixed] = x0[fixed]
    u = x1-x0
    xt = (np.float32(1)-np.float32(t))*x0+np.float32(t)*x1
    xt[fixed] = x0[fixed]
    return dict(N=np.asarray(n), K=np.asarray(1), valid_mask=valid,
                known_mask=np.array([1, 0, 0, 0], bool), anchor_ids=ids,
                x0=x0, x1=x1, xt=xt, u=u, t=np.asarray(t, np.float32),
                free_coordinate_count=np.asarray(int((valid[:, None]&~fixed).sum())*3),
                source_face_ids=np.arange(n), ot_face_map=np.array([1, 0]))


def tensors(s):
    n = int(s["N"])
    return dict(x=torch.from_numpy(s["xt"].reshape(1, n, 9)),
                target=torch.from_numpy(s["u"].reshape(1, n, 9)),
                t=torch.tensor([float(s["t"])], dtype=torch.float32),
                y=torch.tensor([n]),
                valid=torch.from_numpy(s["valid_mask"][None]),
                known=torch.from_numpy(s["known_mask"][None]),
                fixed=torch.from_numpy((s["anchor_ids"]>=0)[None]))


class AsyncAlgebraTests(unittest.TestCase):
    def test_clock_bounds_and_shared_boundary(self):
        for t in [0, .35, .699, .7, .9, 1]:
            np.testing.assert_array_equal(A.seam_clock(np.float32(t)),
                                          A.seam_clock(torch.tensor(t, dtype=torch.float32)).numpy())
        self.assertEqual(float(A.seam_clock(.35)), .5)
        self.assertEqual(float(A.seam_clock(.7)), 1.)
        for bad in [-1, float("nan"), 1.1]:
            with self.assertRaises(ValueError):
                A.seam_clock(bad)

    def test_sample_only_changes_tip_state(self):
        src = sample()
        before = copy.deepcopy(src)
        out = A.transform_sample(src)
        tips = out["seam_tip_mask"]
        np.testing.assert_allclose(out["xt"][tips], .5*src["x0"][tips]+.5*src["x1"][tips])
        np.testing.assert_array_equal(out["xt"][~tips], src["xt"][~tips])
        for key, value in before.items():
            np.testing.assert_array_equal(src[key], value)
        for key in ["x0", "x1", "u", "ot_face_map", "source_face_ids", "free_coordinate_count"]:
            np.testing.assert_array_equal(out[key], before[key])

    def test_late_training_tip_is_teacher_forced(self):
        out = A.transform_sample(sample(.8))
        tips = out["seam_tip_mask"]
        np.testing.assert_array_equal(out["xt"][tips], out["x1"][tips])
        self.assertFalse(out["active_free_corner_mask"][tips].any())
        self.assertTrue(bool(out["async_late_tip_teacher_forced"]))
        self.assertEqual(int(out["original_free_scalar_count"]), 12)

    def test_tau1_and_no_seam_keep_original_xt(self):
        for src, tau in [(sample(.8), 1.), (sample(.8, seam=False), .7)]:
            out = A.transform_sample(src, tau)
            self.assertIs(out["xt"], src["xt"])
            np.testing.assert_array_equal(out["active_free_corner_mask"],
                                          src["valid_mask"][:, None] & (src["anchor_ids"]<0))

    def test_reordered_faces_and_tip_corner(self):
        src = sample()
        order = [2, 0, 3, 1]
        for key in ["x0", "x1", "xt", "u", "anchor_ids", "known_mask", "valid_mask"]:
            src[key] = src[key][order]
        for key in ["x0", "x1", "xt", "u", "anchor_ids"]:
            src[key][3] = src[key][3][[2, 0, 1]]
        out = A.transform_sample(src)
        self.assertTrue(out["seam_tip_mask"][3, 0])
        self.assertEqual(int(out["seam_tip_mask"].sum()), 1)
        self.assertAlmostEqual(float(out["xt"][3, 0, 0]-src["x0"][3, 0, 0]), 1., places=6)

    def test_active_loss_and_fixed_original_denominator(self):
        b = tensors(A.transform_sample(sample(.35)))
        pred = (b["target"]+1).requires_grad_()
        loss, st = A.loss_and_stats(pred, b)
        self.assertEqual(float(loss), 1.)
        self.assertEqual(st["original_free_scalar_count"], 12)
        self.assertEqual(st["active_free_scalar_count"], 12)
        self.assertEqual(st["seam_tip_scalar_count"], 3)
        self.assertAlmostEqual(st["equivalent_active_global_velocity_sse"], 9+3/.49, places=5)
        loss.backward()
        active = (b["valid"][..., None]&~b["fixed"])[..., None].expand(1, 4, 3, 3).reshape_as(pred)
        self.assertTrue(torch.equal(pred.grad[~active], torch.zeros_like(pred.grad[~active])))
        self.assertTrue(bool((pred.grad[active] != 0).all()))

    def test_late_tip_zero_direct_gradient_no_bulk_renormalization(self):
        b = tensors(A.transform_sample(sample(.8)))
        pred = (b["target"]+1).requires_grad_()
        loss, st = A.loss_and_stats(pred, b)
        self.assertEqual(float(loss), .75)  # 9 bulk errors / original 12
        self.assertEqual(st["inactive_seam_tip_scalar_count"], 3)
        self.assertEqual(st["active_free_scalar_count"], 9)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad.reshape(1, 4, 3, 3)[0, 1, 2], torch.zeros(3)))
        self.assertTrue(bool((pred.grad.reshape(1, 4, 3, 3)[0, 2] != 0).all()))
        half, _ = A.loss_and_stats(pred, b, denominator=24)
        self.assertEqual(float(half), .375)

    def test_tau1_loss_exact_legacy_call(self):
        b = tensors(sample(.35))
        pred = b["target"]+1
        with patch.object(A, "corner_masked_fm_loss", wraps=A.corner_masked_fm_loss) as old:
            loss, _ = A.loss_and_stats(pred, b, tau=1.)
        self.assertEqual(old.call_count, 1)
        self.assertEqual(float(loss), 1.)

    def test_endpoint_probe_uses_local_remaining_time(self):
        for t in [.35, .8]:
            s = A.transform_sample(sample(t))
            b = tensors(s)
            got = A.endpoint_prediction(b["target"], b).reshape(4, 3, 3).numpy()
            np.testing.assert_allclose(got[s["valid_mask"]], s["x1"][s["valid_mask"]], atol=1e-6)

    def test_chain_rule_finite_difference(self):
        dt = 1e-4
        for t in [.2, .5, .8]:
            lo, hi = A.transform_sample(sample(t-dt)), A.transform_sample(sample(t+dt))
            derivative = (hi["xt"]-lo["xt"])/(2*dt)
            expected = lo["u"].copy()
            expected[lo["seam_tip_mask"]] *= (1/.7 if t < .7 else 0)
            np.testing.assert_allclose(derivative[lo["valid_mask"]], expected[lo["valid_mask"]],
                                       atol=.005, rtol=.005)

    def test_fifty_steps_integrate_tip_and_bulk_and_freeze_generated_tips(self):
        src = sample()
        b = tensors(src)
        state = torch.from_numpy(src["x0"][None].copy())
        anchors = state.clone()
        velocity = b["target"]*.75  # deliberately wrong: never replace by source Y
        saved = None
        for step in range(50):
            state = A.integration_step(state, velocity, anchors, b["valid"], b["known"], b["fixed"], step)
            if step == 34:
                saved = state[0, 1, 2].clone()
            elif step >= 35:
                self.assertTrue(torch.equal(state[0, 1, 2], saved))
            self.assertTrue(torch.equal(state[b["fixed"]], anchors[b["fixed"]]))
        expected = torch.from_numpy(src["x0"]+.75*src["u"])
        self.assertTrue(torch.allclose(state[0, :3], expected[:3], atol=5e-6, rtol=0))
        self.assertFalse(torch.allclose(state[0, 1, 2], torch.from_numpy(src["x1"][1, 2])))

    def test_exact_tau1_and_no_seam_old_euler_update(self):
        for src, tau in [(sample(), 1.), (sample(seam=False), .7)]:
            b = tensors(src)
            state = torch.from_numpy(src["x0"][None].copy())
            original = state.clone()
            velocity = b["target"]
            free = b["valid"][..., None]&~b["fixed"]
            expected = state.clone()
            expected[free] = state[free]+velocity.reshape_as(state)[free]*(1/50)
            expected[b["fixed"]] = original[b["fixed"]]
            got = A.integration_step(state, velocity, original, b["valid"], b["known"], b["fixed"], 0, tau=tau)
            self.assertTrue(torch.equal(got, expected))

    def test_mask_selection_keeps_roles_separate_from_clock(self):
        b = tensors(sample())
        _, seam, tips, _ = A.mask_partition(b["valid"], b["known"], b["fixed"])
        bulk, known, lead = [torch.tensor([[x]]) for x in [.35, 1., .5]]
        face = A._select_face(bulk, known, lead, b["known"], seam)
        corner = A._select_corner(bulk, known, lead, b["fixed"], tips).reshape(1, 4, 3)
        self.assertTrue(torch.equal(face[0, :3, 0], torch.tensor([1., .5, .35])))
        self.assertTrue(torch.equal(corner[0, 1], torch.tensor([1., 1., .5])))
        self.assertFalse(bool(b["fixed"][0, 1, 2]))  # tip identity remains generated

    def test_forward_delegates_tau1_and_no_seam_without_runtime(self):
        for src, tau in [(sample(), 1.), (sample(seam=False), .7)]:
            b = tensors(src)
            token = object()
            with patch.object(A, "seam_forward", return_value=token) as old:
                got = A.forward(object(), b["x"], b["t"], b["y"], b["valid"], b["known"], b["fixed"], tau=tau)
            self.assertIs(got, token)
            self.assertEqual(old.call_count, 1)

    def test_mock_forward_only_original_C_enters_Geo(self):
        # Tiny mock width=3; this is not a NativeModel or production generator.
        width, n = 3, 128
        x = torch.zeros(1, n, 9)
        valid = torch.ones(1, n, dtype=torch.bool)
        known = torch.zeros_like(valid); known[:, 0] = True
        fixed = torch.zeros(1, n, 3, dtype=torch.bool); fixed[:, 0] = True; fixed[:, 1, :2] = True
        clock_calls = []
        class Time(nn.Module):
            def forward(self, t):
                clock_calls.append(t.detach().clone())
                return t[:, None].expand(-1, width)
        class Embed(nn.Module):
            def forward(self, value):
                return None, torch.zeros(1, n, 3, width)
        class Mod(nn.Module):
            def __init__(self, chunks):
                super().__init__(); self.chunks = chunks
            def forward(self, c):
                return c.repeat(1, self.chunks)
        class Zero(nn.Module):
            def forward(self, h, mask=None):
                return torch.zeros_like(h)
        final = SimpleNamespace(adaLN_modulation=Mod(2), norm_final=nn.Identity(), linear=nn.Identity())
        block = SimpleNamespace(adaLN_modulation=Mod(6), norm1=nn.Identity(), norm2=nn.Identity(), attn=Zero(), mlp=Zero())
        backbone = SimpleNamespace(x_embedder=Embed(), t_embedder=Time(), y_embedder=lambda y, train: torch.zeros(1, width),
                                   face_bin=20, layers=[block], final_layer=final)
        geo_calls = []
        def geo(value, C, mask, observer=None):
            geo_calls.append(C.clone())
            return torch.zeros(1, n, width)
        role = nn.Parameter(torch.zeros(2, width))
        model = SimpleNamespace(backbone=backbone, role_embedding=role, context_encoder=geo,
                                context_observer=None, readout_mode="none", readouts={})
        with patch.object(A, "assert_native_runtime"):
            result = A.forward(model, x, torch.tensor([.35]), torch.tensor([n]), valid, known, fixed)
        self.assertEqual(result.shape, (1, n, 9))
        self.assertEqual([float(t.item()) for t in clock_calls], [float(torch.tensor(.35)), 1., .5])
        self.assertEqual(len(geo_calls), 1)
        self.assertTrue(torch.equal(geo_calls[0], known))
        result.reshape(1, n, 3, 3).sum().backward()
        self.assertIsNotNone(role.grad)
        self.assertTrue(torch.equal(fixed[0, 1], torch.tensor([True, True, False])))

    def test_bad_single_anchor_nonC_rejected(self):
        b = tensors(sample())
        b["fixed"][0, 1, 1] = False
        with self.assertRaises(ValueError):
            A.active_masks(b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
