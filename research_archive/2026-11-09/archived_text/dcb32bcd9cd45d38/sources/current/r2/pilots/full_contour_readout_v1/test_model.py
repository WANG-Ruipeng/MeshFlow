"""Bounded CPU-only synthetic tests; no full generator/model is constructed."""
from __future__ import annotations

import itertools
import json
import unittest
import torch
from torch import nn
from meshflow_control.models.native import NativeModel
from .model import (
    ContourReadout, EXTRA_PARAMETERS, apply_contour_readout, upgrade_model,
)
from .relations import (
    LENGTH_SCALE, boundary_mask_from_vertex_ids, closest_point_triangle,
    relation_features,
)


def rectangle():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                             [1., 1., 0.], [0., 1., 0.]])
    ids = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    return vertices[ids], ids


class GeometryTests(unittest.TestCase):
    def test_rectangle_excludes_internal_diagonal_and_preserves_face_ownership(self):
        tri, ids = rectangle()
        mask = boundary_mask_from_vertex_ids(ids)
        self.assertEqual(mask.tolist(), [[True, True, False], [False, True, True]])
        point = torch.tensor([[.5, .5, 0.], [.9, .1, .2]])
        f = relation_features(point, tri, mask, torch.tensor(.25))
        self.assertEqual(tuple(f.shape), (2, 2, 17))
        self.assertTrue(torch.equal(f[0, :, :4], torch.zeros(2, 4)))
        self.assertTrue(torch.allclose(f[0, :, 7], torch.full((2,), .5 / LENGTH_SCALE)))
        torch.testing.assert_close(f[0, 0, 4:7] * LENGTH_SCALE, torch.tensor([.25, -.25, 0.]))
        torch.testing.assert_close(f[0, 0, 8:14], torch.tensor([.5, .5, 0., 0., 0., 0.]))
        self.assertGreater(float(f[1, 1, 7]), float(f[1, 0, 7]))
        torch.testing.assert_close(f[:, :, 14], torch.ones(2, 2))
        torch.testing.assert_close(f[:, :, 15], torch.full((2, 2), 2 / 3))

    def test_all_corner_permutations_face_reorder_and_id_renaming(self):
        tri, ids = rectangle()
        points = torch.tensor([[.5, .5, .1], [0., 0., .2], [.15, .7, -.5], [2., -2., 0.]])
        original = relation_features(points, tri, boundary_mask_from_vertex_ids(ids), torch.tensor(.7))
        rename = torch.tensor([71, -3, 999, 18], dtype=torch.int64)
        for permutation in itertools.permutations(range(3)):
            p = list(permutation)
            permuted_ids = rename[ids[:, p].flip(0)]
            actual = relation_features(points, tri[:, p].flip(0),
                                       boundary_mask_from_vertex_ids(permuted_ids), torch.tensor(.7))
            torch.testing.assert_close(actual.flip(1), original, rtol=0, atol=0)

    def test_empty_boundary_finite_zero_preserves_surface(self):
        tri, _ = rectangle()
        f = relation_features(torch.tensor([[.2, .1, .7]]), tri,
                              torch.zeros(2, 3, dtype=torch.bool), torch.tensor(.9))
        self.assertTrue(bool(torch.isfinite(f).all()))
        self.assertTrue(torch.equal(f[:, :, 4:16], torch.zeros(1, 2, 12)))
        self.assertGreater(float(f[0, 0, 3]), 0.)
        self.assertEqual(float(f[0, 0, 16]), float(torch.tensor(.9)))

    def test_closest_point_is_finite_triangle_not_infinite_plane(self):
        tri = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        points = torch.tensor([[.2, .2, 1.], [2., 2., 0.], [-1., -1., 0.]])
        actual = closest_point_triangle(points, tri)[:, 0]
        torch.testing.assert_close(actual, torch.tensor([[.2, .2, 0.], [.5, .5, 0.], [0., 0., 0.]]))

    def test_degenerate_point_and_line_are_finite(self):
        tri = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                            [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
        f = relation_features(torch.tensor([[.5, 1., 1.]]), tri,
                              torch.ones(2, 3, dtype=torch.bool), torch.tensor(.5))
        self.assertTrue(bool(torch.isfinite(f).all()))
        torch.testing.assert_close(f[0, 0, 8:14], torch.zeros(6))

    def test_incidence_is_id_topology_not_coordinate_welding(self):
        ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
        self.assertTrue(bool(boundary_mask_from_vertex_ids(ids).all()))
        nonmanifold = torch.tensor([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=torch.int64)
        self.assertEqual(boundary_mask_from_vertex_ids(nonmanifold)[:, 0].tolist(), [False] * 3)


class ModuleTests(unittest.TestCase):
    def setUp(self):
        torch.random.default_generator.manual_seed(312)

    def inputs(self, hidden_dim=8):
        tri, ids = rectangle()
        return (torch.randn(3, hidden_dim), torch.randn(2, hidden_dim),
                torch.tensor([[.2, .1, .3], [.7, .3, -.2], [1.3, .9, .7]]),
                tri, boundary_mask_from_vertex_ids(ids), torch.tensor(.4))

    def test_capacity_and_initialization(self):
        module = ContourReadout()
        self.assertEqual(sum(p.numel() for p in module.parameters()), EXTRA_PARAMETERS)
        self.assertEqual(EXTRA_PARAMETERS, 412032)
        self.assertTrue(bool((module.W_O.weight == 0).all()))
        self.assertIsNone(module.W_Q.bias)
        self.assertIsNone(module.W_O.bias)
        self.assertIsNotNone(module.phi[0].bias)
        self.assertFalse(module.norm.elementwise_affine)

    def test_zero_output_first_gradient_then_qkv_phi_and_memory_gradients(self):
        module = ContourReadout(hidden_dim=8, readout_dim=8, heads=2)
        query, memory, *geometry = self.inputs()
        query.requires_grad_(); memory.requires_grad_()
        output = module(query, memory, *geometry)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        (output * torch.randn_like(output)).sum().backward()
        self.assertGreater(float(module.W_O.weight.grad.abs().sum()), 0.)
        for name, parameter in module.named_parameters():
            if name != "W_O.weight":
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(float(parameter.grad.abs().sum()), 0., name)
        with torch.no_grad():
            module.W_O.weight.normal_(0., .1)
        module.zero_grad(set_to_none=True); query.grad = None; memory.grad = None
        module(query, memory, *geometry).square().sum().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        self.assertGreater(float(query.grad.abs().sum()), 0.)
        self.assertGreater(float(memory.grad.abs().sum()), 0.)

    def test_geometry_value_depends_on_each_query(self):
        module = ContourReadout(hidden_dim=8, readout_dim=8, heads=2)
        with torch.no_grad():
            for p in module.parameters(): p.zero_()
            module.phi[0].weight[0, 2] = 1.
            module.phi[2].weight[0, 0] = 1.
            module.W_O.weight[0, 0] = 1.
        query = torch.zeros(2, 8); memory = torch.zeros(1, 8)
        points = torch.tensor([[.2, .1, .3], [.2, .1, .8]])
        tri = rectangle()[0][:1]
        result = module(query, memory, points, tri, torch.ones(1, 3, dtype=torch.bool), torch.tensor(.4))
        self.assertNotEqual(float(result[0, 0]), float(result[1, 0]))
        expected = nn.functional.silu(-points[:, 2] / LENGTH_SCALE)
        torch.testing.assert_close(result[:, 0], expected)

    def test_outer_bf16_autocast_does_not_reduce_branch_precision(self):
        module = ContourReadout(hidden_dim=8, readout_dim=8, heads=2)
        with torch.no_grad(): module.W_O.weight.normal_(0., .1)
        inputs = self.inputs()
        expected = module(*inputs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = module(*inputs)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(expected, actual))

    def test_only_valid_free_updated_fp32_add_before_cast_and_h0_grad(self):
        module = ContourReadout(hidden_dim=8, readout_dim=8, heads=2)
        with torch.no_grad(): module.W_O.weight.normal_(0., .1)
        hidden = torch.randn(1, 12, 8).bfloat16().requires_grad_()
        h0 = torch.randn(1, 4, 3, 8, requires_grad=True)
        known = torch.tensor([[True, False, False, False]])
        valid = torch.tensor([[True, True, True, False]])
        x = torch.randn(1, 4, 9)
        x[0, 0] = rectangle()[0][0].flatten()
        mask = torch.zeros(1, 4, 3, dtype=torch.bool); mask[0, 0] = True
        memory = (h0.float().mean(2)[0, known[0]],)
        receipts = []
        result = apply_contour_readout(hidden, memory, x, torch.tensor([.5]), known,
                                      valid, mask, module, observer=receipts.append)
        self.assertTrue(torch.equal(result[0, :3], hidden[0, :3]))
        self.assertTrue(torch.equal(result[0, 9:], hidden[0, 9:]))
        self.assertFalse(torch.equal(result[0, 3:9], hidden[0, 3:9]))
        addition = next(r for r in receipts if r["site"] == "contour_addition")
        self.assertTrue(torch.equal(addition["after"],
                                   (addition["before"].float() + addition["residual"]).bfloat16()))
        result.float().square().sum().backward()
        self.assertGreater(float(h0.grad[0, 0].abs().sum()), 0.)
        self.assertGreater(float(hidden.grad.abs().sum()), 0.)

    def test_c_face_reorder_equivariance(self):
        module = ContourReadout(hidden_dim=8, readout_dim=8, heads=2)
        with torch.no_grad(): module.W_O.weight.normal_(0., .1)
        q, m, p, tri, mask, t = self.inputs()
        torch.testing.assert_close(module(q, m, p, tri, mask, t),
                                   module(q, m.flip(0), p, tri.flip(0), mask.flip(0), t),
                                   rtol=1e-6, atol=1e-7)

    def test_upgrade_preserves_old_parameter_objects_names_order_and_rng(self):
        # Tiny Native-shaped fixture: never a generator construction/forward.
        base = NativeModel.__new__(NativeModel)
        nn.Module.__init__(base)
        base.role_embedding = nn.Parameter(torch.randn(2, 8))
        base.backbone = nn.Linear(8, 8)
        base.context_encoder = nn.Linear(8, 8)
        base.readouts = nn.ModuleDict(); base.readout_mode = "none"
        base.experiment_trainable = True
        original = [(n, id(p)) for n, p in base.named_parameters()]
        state = torch.get_rng_state().clone()
        candidate = upgrade_model(base, seed=20261014)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual([(n, id(p)) for n, p in candidate.named_parameters()
                          if not n.startswith("contour_readout.")], original)
        self.assertEqual(candidate.contour_metadata()["extra_parameters"], EXTRA_PARAMETERS)
        json.dumps(candidate.contour_metadata(), allow_nan=False)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    unittest.main(verbosity=2)
