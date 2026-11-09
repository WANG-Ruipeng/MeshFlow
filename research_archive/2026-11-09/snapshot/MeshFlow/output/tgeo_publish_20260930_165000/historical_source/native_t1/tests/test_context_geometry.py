"""Fixed CPU geometry/encoder gates; never run a complete MeshFlow backbone."""
import copy
import itertools
import unittest
import numpy as np
import torch
from torch import nn
from native_t1.context_geometry import (
    ContextGeometryEncoder, ContextGeometryObserver, face_geometry,
    inject_context, make_context_encoder, FORMULA_ATOL, FORMULA_RTOL)
from native_t1.model import NativeInpaintingModel

COUNTS = dict(geometry_calls=0, encoder_calls=0, output_space_autograd_calls=0,
              complete_backbone_forwards=0, optimizer_updates=0)


def geometry(c):
    COUNTS["geometry_calls"] += 1
    return face_geometry(c)


def encode(model, x, known, valid, observer=None):
    COUNTS["encoder_calls"] += 1
    return model(x, known, valid, observer=observer)


def fixture():
    return torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                         [[1., 0., 0.], [0., 0., 0.], [0., -1., 0.]],
                         [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.]],
                         [[10., 10., 10.], [11., 10., 10.], [10., 11., 10.]]])


def inputs(c=None, slots=None):
    c = fixture() if c is None else c
    n = 9
    slots = list(range(len(c))) if slots is None else slots
    x = torch.linspace(-2., 2., n*9).reshape(1, n, 9)
    known = torch.zeros(1, n, dtype=torch.bool)
    x[0, slots] = c.reshape(-1, 9)
    known[0, slots] = True
    return x, known, torch.ones_like(known)


def activate_exit(model):
    # Fixed test-only nonzero outlet; not an optimizer update or experiment init.
    with torch.no_grad():
        model.W_out.weight.copy_(torch.linspace(-.03, .03, 128*768).reshape(768, 128))


class ContextGeometryTests(unittest.TestCase):
    def close(self, a, b):
        torch.testing.assert_close(a, b, atol=FORMULA_ATOL, rtol=FORMULA_RTOL)

    def test_geometry_shared_edge_vertex_disconnected_and_nonmanifold(self):
        _, a, facts = geometry(fixture())
        expected = torch.zeros(4, 4, dtype=torch.bool)
        expected[0, 1] = expected[1, 0] = True
        self.assertTrue(torch.equal(a, expected))
        self.assertEqual(int(facts["nonmanifold_edge_count"]), 0)
        three = torch.cat((fixture()[:2], torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,0.,1.]]])))
        _, a, facts = geometry(three)
        self.assertTrue(torch.equal(a, ~torch.eye(3, dtype=torch.bool)))
        self.assertEqual(int(facts["nonmanifold_edge_count"]), 1)

    def test_reject_ambiguous_nonfinite_and_degenerate_known(self):
        c = fixture()[:1]
        bad = [torch.cat((c, c[:, [2, 1, 0]])),
               torch.tensor([[[0.,0.,0.],[0.,0.,0.],[1.,0.,0.]]]),
               torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]]),
               torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1e-12,0.]]]),
               c + float("nan"), c + float("inf")]
        for item in bad:
            with self.subTest(shape=item.shape), self.assertRaises(ValueError):
                geometry(item)
        geometry(torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,4e-12,0.]]]))

    def test_feature_S3_and_face_permutation(self):
        c = fixture()
        phi, a, _ = geometry(c)
        for order in itertools.permutations(range(3)):
            p, q, _ = geometry(c[:, list(order)])
            self.assertTrue(torch.equal(p, phi))
            self.assertTrue(torch.equal(q, a))
        order = torch.tensor([2, 0, 3, 1])
        p, q, _ = geometry(c[order])
        self.assertTrue(torch.equal(p, phi[order]))
        self.assertTrue(torch.equal(q, a[order][:, order]))

    def test_fp64_independent_features_and_encoder_formula(self):
        c = fixture().clone()
        c[2, 1] = torch.tensor([.15, .2, .9])
        p32, adjacency, _ = geometry(c)
        p64, _, _ = geometry(c.double())
        expected = []
        for tri in c.double().numpy():
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            twice_area = np.linalg.norm(normal)
            normal /= twice_area
            x, y, z = normal
            expected.append(np.r_[tri.mean(0), sorted(np.linalg.norm(tri[i]-tri[j])
                            for i,j in ((0,1),(0,2),(1,2))), twice_area*.5,
                            [x*x,y*y,z*z,x*y,x*z,y*z]])
        np.testing.assert_allclose(p64.numpy(), expected, atol=1e-12, rtol=1e-12)
        self.close(p32.double(), p64)
        model = ContextGeometryEncoder("graph")
        activate_exit(model)
        x, known, valid = inputs(c)
        observed = []
        result = encode(model, x, known, valid, observed.append)
        record = next(r for r in observed if r["site"] == "geometry")
        double = copy.deepcopy(model).double()
        p = adjacency.double()+torch.eye(len(c), dtype=torch.float64)
        p /= p.sum(-1, keepdim=True)
        h0 = double.mlp(p64)
        h1 = h0+torch.nn.functional.silu(double.W_self(h0)+double.W_neighbor(p@h0))
        delta = double.W_out(double.norm(h1))
        self.close(record["h1"].double(), h1)
        self.close(result[0, :len(c)].double(), delta)

    def test_matching_initialization_rng_and_zero_exit(self):
        before = torch.get_rng_state().clone()
        geo = make_context_encoder("geo")
        graph = make_context_encoder("graph")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in geo.parameters()), 149888)
        self.assertEqual(set(geo.state_dict()), set(graph.state_dict()))
        for key, value in geo.state_dict().items():
            self.assertTrue(torch.equal(value, graph.state_dict()[key]))
        x, known, valid = inputs()
        for model in (geo, graph):
            delta = encode(model, x, known, valid)
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.version, self.hidden_size = 3, 768
                self.use_dit_like_pe, self.face_cond = False, True
                self.weight = nn.Parameter(torch.ones(1))
                self.layers = nn.ModuleList()
        native = NativeInpaintingModel(Stub(), trainable=False)
        self.assertEqual(set(native.state_dict()), {"role_embedding", "backbone.weight"})
        self.assertIsNone(native.context_encoder)
        self.assertIsNone(native.context_observer)

    def test_C_only_free_padding_no_metadata_and_scattered_slots(self):
        model = ContextGeometryEncoder("graph")
        activate_exit(model)
        x, known, valid = inputs()
        records = []
        expected = encode(model, x, known, valid, records.append)
        noisy = x.clone()
        noisy[~known] = float("nan")
        valid2 = valid.clone()
        valid2[~known] = False
        records2 = []
        result = encode(model, noisy, known, valid2, records2.append)
        self.assertTrue(torch.equal(expected, result))
        r1 = next(r for r in records if r["site"] == "geometry")
        r2 = next(r for r in records2 if r["site"] == "geometry")
        for name in ("phi", "P", "h1"):
            self.assertTrue(torch.equal(r1[name], r2[name]))
        with self.assertRaises(TypeError):
            model(x, known, valid, full_target=torch.zeros_like(x))
        xs, ks, vs = inputs(slots=[1, 3, 6, 8])
        actual = encode(model, xs, ks, vs)
        self.assertTrue(torch.equal(actual[0, ks[0]], expected[0, known[0]]))
        self.assertTrue(torch.equal(actual[~ks], torch.zeros_like(actual[~ks])))
        invalid = vs.clone()
        invalid[ks] = False
        with self.assertRaises(ValueError):
            encode(model, xs, ks, invalid)

    def test_face_and_corner_permutations_and_repeat(self):
        x, known, valid = inputs()
        order = torch.tensor([8, 3, 1, 6, 0, 7, 4, 2, 5])
        for mode in ("geo", "graph"):
            model = ContextGeometryEncoder(mode)
            activate_exit(model)
            expected = encode(model, x, known, valid)
            self.assertTrue(torch.equal(expected, encode(model, x, known, valid)))
            permuted = x[:, order].reshape(1, 9, 3, 3)[:, :, [2, 0, 1]].reshape_as(x)
            actual = encode(model, permuted, known[:, order], valid[:, order])
            self.close(actual, expected[:, order])

    def test_observation_autocast_and_effective_injection(self):
        model = ContextGeometryEncoder("graph")
        activate_exit(model)
        x, known, valid = inputs()
        observer = ContextGeometryObserver(retain_geometry=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = encode(model, x, known, valid)
            actual = encode(model, x, known, valid, observer)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(expected, actual))
        hidden = torch.ones(1, 27, 768, dtype=torch.bfloat16)
        tiny = actual*1e-6
        after = inject_context(hidden, tiny, known, observer=observer)
        self.assertTrue(torch.equal(after, inject_context(hidden, tiny, known)))
        stats = observer.injections[-1]
        self.assertTrue(stats["free_direct_delta_exact_zero"])
        self.assertTrue(stats["free_hidden_unchanged"])
        self.assertGreater(stats["nonzero_delta_elements"], 0)
        self.assertEqual(stats["swallowed_nonzero_fraction"], 1.0)
        self.assertEqual(stats["effective_RMS_known"], 0.0)
        self.assertEqual(stats["hidden_dtype"], "torch.bfloat16")
        timing = []
        class Timing:
            requires_geometry_details = False
            requires_injection_details = False
            def __call__(self, record): timing.append(record["site"])
        encode(model, x, known, valid, Timing())
        self.assertEqual(timing, ["geometry_start", "geometry_end"])

    def test_zero_exit_then_nonzero_upstream_gradients(self):
        model = ContextGeometryEncoder("graph")
        x, known, valid = inputs()
        weights = torch.linspace(-1., 1., 9*768).reshape(1, 9, 768)
        names, params = zip(*model.named_parameters())
        result = encode(model, x, known, valid)
        COUNTS["output_space_autograd_calls"] += 1
        gradients = torch.autograd.grad((result*weights).sum(), params)
        for name, grad in zip(names, gradients):
            self.assertTrue(bool(torch.isfinite(grad).all()))
            if name == "W_out.weight": self.assertGreater(float(grad.norm()), 0.0)
            else: self.assertEqual(float(grad.norm()), 0.0)
        activate_exit(model)
        result = encode(model, x, known, valid)
        COUNTS["output_space_autograd_calls"] += 1
        gradients = torch.autograd.grad((result*weights).sum(), params)
        for name, grad in zip(names, gradients):
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(float(grad.norm()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
