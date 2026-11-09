"""CPU contracts only: no learned-model forward, training, or CUDA dispatch."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "r2/src"))
from .model import (area_center, to_local, world_trajectory, as_c_reference_model,
    CReferenceModel, _build_native, make_noise)
import numpy as np
import torch
from meshflow_control.models.geometry import face_geometry

def condition():
    return np.asarray([[[0, 0, 0], [2, 0, 0], [0, 2, 0]],
                       [[0, 0, 3], [1, 0, 3], [0, 1, 3]]], np.float32)

class GeometryContract(unittest.TestCase):
    def test_surface_center_is_area_weighted_not_corner_count(self):
        center = area_center(condition())
        self.assertEqual(center.dtype, np.float32)
        self.assertEqual(center.shape, (3,))
        np.testing.assert_allclose(center, [.6, .6, .6], rtol=0, atol=1e-7)
        self.assertGreater(np.max(np.abs(center - condition().mean((0, 1)))), .5)

    def test_augmented_C_only_and_translation(self):
        C = condition()
        scale = np.asarray([1.25, 1, 1], np.float32)
        a = area_center(C * scale)
        np.testing.assert_allclose(a, area_center(C) * scale, rtol=0, atol=1e-7)
        shift = np.asarray([.25, -.5, 1], np.float32)
        np.testing.assert_allclose(area_center(C + shift), area_center(C) + shift, atol=2e-7, rtol=0)
        self.assertEqual(to_local(C, area_center(C)).dtype, np.float32)

    def test_reject_degenerate_and_wrong_precision(self):
        for C in (condition().astype(np.float64), np.zeros((1, 3, 3), np.float32),
                  np.empty((0, 3, 3), np.float32), condition() * np.float32(np.nan)):
            with self.assertRaises(ValueError):
                area_center(C)

    def test_fixed_gaussian_and_C_byte_copy_for_all_states(self):
        C = condition()
        C[0, 0, 0] = np.float32(-0.0)
        center = area_center(C)
        z = make_noise(4821, 128, len(C))
        before = z.tobytes()
        local = np.repeat(z, 51, axis=0)
        local[:, :len(C)] = to_local(C, center).reshape(len(C), 9)
        world = world_trajectory(local, C, center)
        self.assertEqual(z.tobytes(), before)
        self.assertEqual(local[0, len(C):].tobytes(), z[0, len(C):].tobytes())
        for state in world:
            self.assertEqual(state[:len(C)].tobytes(), C.tobytes())
        expected = local[:, len(C):].reshape(51, -1, 3) + center
        self.assertEqual(world[:, len(C):].tobytes(), expected.tobytes())
        self.assertNotEqual(world[0, len(C):].tobytes(), z[0, len(C):].tobytes())

    def test_Geo_world_centroid_is_preserved_as_separate_domain(self):
        C = condition()
        local = to_local(C, area_center(C))
        world_phi, _, _ = face_geometry(torch.from_numpy(C))
        local_phi, _, _ = face_geometry(torch.from_numpy(local))
        np.testing.assert_allclose(world_phi[:, :3].numpy() - local_phi[:, :3].numpy(),
            np.repeat(area_center(C)[None], len(C), axis=0), atol=2e-7, rtol=0)
        np.testing.assert_allclose(world_phi[:, 3:].numpy(), local_phi[:, 3:].numpy(), atol=5e-7, rtol=0)

    def test_pre_OT_target_translation_is_not_cached_xt_translation(self):
        pre = condition() * np.float32(.5)
        a = area_center(condition())
        expected = to_local(condition(), a)
        actual = (pre - a / np.float32(2)) * np.float32(2)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        z = np.ones_like(condition())
        t = np.float32(.25)
        new_xt = (np.float32(1) - t) * z + t * expected
        wrong_xt = (np.float32(1) - t) * z + t * condition() - a
        self.assertFalse(np.array_equal(new_xt, wrong_xt))

    def test_promotion_preserves_actual_native_parameter_objects_and_keys(self):
        config = dict(version=3, hidden_dim=768, num_layers=12, num_heads=12,
            max_length=800, face_bin=20, pe_freq=20, use_qknorm=True,
            use_rmsnorm=True, use_coord_encoding=True, use_dit_like_pe=False,
            face_cond=True, gradient_checkpointing=False, mixed_precision="bf16")
        with torch.device("meta"):
            native = _build_native(config, readout_mode="none", trainable=True)
        keys = tuple(native.state_dict())
        ids = tuple((name, id(p)) for name, p in native.named_parameters())
        counts = sum(p.numel() for p in native.parameters())
        promoted = as_c_reference_model(native)
        self.assertIs(promoted, native)
        self.assertIs(type(promoted), CReferenceModel)
        self.assertEqual(tuple(promoted.state_dict()), keys)
        self.assertEqual(tuple((name, id(p)) for name, p in promoted.named_parameters()), ids)
        self.assertEqual(sum(p.numel() for p in promoted.parameters()), counts)
        self.assertIs(as_c_reference_model(promoted), promoted)
        self.assertFalse(torch.cuda.is_initialized())

if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryContract))
    print(json.dumps(dict(status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, learned_model_forwards=0, training_updates=0,
        cuda_initialized=torch.cuda.is_initialized())))
    raise SystemExit(0 if result.wasSuccessful() else 1)