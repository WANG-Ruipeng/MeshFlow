"""Portable synthetic CPU contracts; no historical assets, model or GPU."""
from __future__ import annotations

import inspect
import json
import unittest

import numpy as np

from native_t1.postprocessing import RADIUS, process


def mesh(*triangles):
    return np.asarray(triangles, dtype=np.float32)


def known():
    return [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]


def soup_key(triangles):
    return sorted(tuple(sorted(tuple(float(v) for v in p) for p in tri)) for tri in triangles)


class PostprocessingContracts(unittest.TestCase):
    def check_contracts(self, raw, result, K=1):
        self.assertEqual(raw[:K].tobytes(), result['output'][:K].tobytes())
        self.assertEqual(result['vertices'][result['faces']].tobytes(), result['output'].tobytes())
        self.assertTrue(np.all(result['corner_displacement'] <= RADIUS))
        self.assertEqual(result['output'].dtype, np.float32)
        reconstructed = raw.reshape(-1, 3)[result['corner_target_input_ids']]
        kept = result['source_face_ids']
        self.assertEqual(reconstructed[kept].tobytes(), result['output'].tobytes())
        json.dumps(result['stats']); json.dumps(result['face_records'])

    def test_01_noop_and_signed_zero_C(self):
        C = mesh(known(), [[-0., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
        raw = np.concatenate((C, mesh([[3., 0., 0.], [4., 0., 0.], [3., 1., 0.]])))
        original = raw.tobytes()
        raw.flags.writeable = False
        out = process(raw, 2)
        self.check_contracts(raw, out, 2)
        self.assertEqual(raw.tobytes(), original)
        self.assertEqual(out['output'].tobytes(), original)
        self.assertNotEqual(out['faces'][0, 0], out['faces'][1, 0])

    def test_02_unique_anchor_snap(self):
        raw = mesh(known(), [[RADIUS * .5, 0., 0.], [2., 0., 0.], [2., 1., 0.]])
        out = process(raw, 1)
        self.check_contracts(raw, out)
        self.assertEqual(out['faces'][1, 0], out['faces'][0, 0])
        self.assertEqual(out['corner_status'][1, 0], 'unique_C_anchor')
        self.assertEqual(out['corner_target_input_ids'][1, 0], 0)

    def test_03_multiple_anchor_conflict_stays_fixed(self):
        raw = mesh([[0., 0., 0.], [RADIUS, 0., 0.], [0., 1., 0.]],
                   [[RADIUS * .5, 0., 0.], [2., 0., 0.], [2., 1., 0.]])
        out = process(raw, 1)
        self.check_contracts(raw, out)
        self.assertEqual(out['corner_C_candidate_count'][1, 0], 2)
        self.assertEqual(out['corner_status'][1, 0], 'ambiguous_C_unchanged')
        self.assertEqual(out['output'][1, 0].tobytes(), raw[1, 0].tobytes())

    def test_04_free_clustering_is_nontransitive(self):
        a, b, c = 3., 3. + .75 * RADIUS, 3. + 1.5 * RADIUS
        raw = mesh(known(), [[a, 0., 0.], [b, 0., 0.], [c, 0., 0.]])
        out = process(raw, 1)
        self.check_contracts(raw, out)
        self.assertEqual(out['corner_target_input_ids'][1].tolist(), [3, 3, 5])
        self.assertEqual(out['corner_status'][1, 2], 'free_representative')
        self.assertEqual(out['stats']['deleted_free_faces'], 1)

    def test_05_duplicates_known_priority_and_winding(self):
        free = [[3., 0., 0.], [4., 0., 0.], [3., 1., 0.]]
        raw = mesh(known(), known()[::-1], free, free[::-1])
        out = process(raw, 1)
        self.check_contracts(raw, out)
        self.assertEqual(out['source_face_ids'].tolist(), [0, 2])
        self.assertEqual(out['face_records'][1]['reason'], 'duplicate_C')
        self.assertEqual(out['face_records'][3]['reason'], 'duplicate_free')
        self.assertEqual(out['output'][1].tobytes(), raw[2].tobytes())

    def test_06_exact_degenerate_free_only(self):
        raw = mesh([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
                   [[3., 0., 0.], [4., 0., 0.], [5., 0., 0.]],
                   [[3., 2., 0.], [3., 2., 0.], [4., 2., 0.]],
                   [[3., 4., 0.], [4., 4., 0.], [5., 4., 1e-8]])
        out = process(raw, 1)
        self.check_contracts(raw, out)
        self.assertEqual(out['source_face_ids'].tolist(), [0, 3])
        self.assertEqual(out['stats']['known_degenerate_faces_retained'], 1)
        self.assertEqual(out['face_records'][1]['reason'], 'exact_collinear')
        self.assertEqual(out['face_records'][2]['reason'], 'exact_repeated_vertex')

    def test_07_face_and_corner_permutation_geometry(self):
        raw = mesh(known(), [[.01, 0., 0.], [3., 0., 0.], [3., 1., 0.]],
                   [[3. + RADIUS * .7, 0., 0.], [4., 0., 0.], [4., 1., 0.]],
                   [[3. + RADIUS * 1.4, 0., 0.], [5., 0., 0.], [5., 1., 0.]])
        permuted = raw[[0, 3, 1, 2]][:, [2, 0, 1]].copy()
        a = process(raw, 1); b = process(permuted, 1)
        self.check_contracts(raw, a); self.check_contracts(permuted, b)
        self.assertEqual(soup_key(a['output']), soup_key(b['output']))
        self.assertEqual(a['stats']['moved_free_corners'], b['stats']['moved_free_corners'])

    def test_08_contract_rejection_no_GT_or_tuning(self):
        raw = mesh(known(), [[3., 0., 0.], [4., 0., 0.], [3., 1., 0.]])
        self.assertEqual(list(inspect.signature(process).parameters), ['raw', 'K', 'radius'])
        with self.assertRaises(TypeError): process(raw, 1, GT=raw)
        with self.assertRaises(TypeError): process(raw.astype(np.float64), 1)
        with self.assertRaises(ValueError): process(raw, 0)
        with self.assertRaises(ValueError): process(raw, 1, radius=RADIUS * 2)
        invalid = raw.copy(); invalid[1, 0, 0] = np.nan
        with self.assertRaises(ValueError): process(invalid, 1)
        a = process(raw, 1); b = process(raw, 1)
        self.assertEqual(a['output'].tobytes(), b['output'].tobytes())
        self.assertEqual(a['faces'].tobytes(), b['faces'].tobytes())
        self.check_contracts(raw, a)

    def test_09_duplicate_and_opposite_winding_known_faces_retained(self):
        raw = mesh(known(), known(), known()[::-1],
                   [[3., 0., 0.], [4., 0., 0.], [3., 1., 0.]])
        out = process(raw, 3)
        self.check_contracts(raw, out, 3)
        self.assertEqual(out['source_face_ids'].tolist(), [0, 1, 2, 3])
        self.assertEqual(out['output'][:3].tobytes(), raw[:3].tobytes())
        self.assertTrue(all(r['known'] and r['kept'] for r in out['face_records'][:3]))
        self.assertEqual(out['faces'][0].tolist(), out['faces'][1].tolist())
        self.assertEqual(out['faces'][0].tolist(), out['faces'][2][::-1].tolist())

    def test_10_all_free_faces_may_be_deleted(self):
        raw = mesh(known(), known()[::-1],
                   [[3., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
        out = process(raw, 1)
        self.check_contracts(raw, out)
        self.assertEqual(out['output'].shape, (1, 3, 3))
        self.assertEqual(out['faces'].shape, (1, 3))
        self.assertEqual(out['source_face_ids'].tolist(), [0])
        self.assertEqual(out['source_face_kept'].tolist(), [True, False, False])
        self.assertEqual(out['stats']['deleted_free_faces'], 2)
        self.assertTrue(np.all(out['corner_vertex_ids'][2] == -1))
        self.assertEqual(out['corner_target_input_ids'].shape, (3, 3))

    def test_11_noncontiguous_readonly_input_matches_contiguous(self):
        expected = mesh(known(), [[.01, 0., 0.], [3., 0., 0.], [3., 1., 0.]])
        storage = np.zeros((4, 3, 3), dtype=np.float32)
        storage[::2] = expected
        raw = storage[::2]
        self.assertFalse(raw.flags.c_contiguous)
        original_storage = storage.tobytes()
        raw.flags.writeable = False
        a, b = process(raw, 1), process(expected, 1)
        self.check_contracts(raw, a)
        for key, value in a.items():
            if isinstance(value, np.ndarray):
                self.assertEqual(value.dtype, b[key].dtype, key)
                self.assertEqual(value.shape, b[key].shape, key)
                self.assertEqual(value.tobytes(), b[key].tobytes(), key)
        self.assertEqual(a['face_records'], b['face_records'])
        self.assertEqual({k: v for k, v in a['stats'].items() if k != 'seconds'},
                         {k: v for k, v in b['stats'].items() if k != 'seconds'})
        self.assertEqual(storage.tobytes(), original_storage)

    def test_12_invalid_shape_known_count_and_radius_are_rejected(self):
        raw = mesh(known(), [[3., 0., 0.], [4., 0., 0.], [3., 1., 0.]])
        for bad_K in (-1, 0, 2, 3, True, 1.0):
            with self.subTest(K=bad_K), self.assertRaises(ValueError):
                process(raw, bad_K)
        for bad_raw in (raw.reshape(2, 9), raw[:0], raw[:, :2]):
            with self.subTest(shape=bad_raw.shape), self.assertRaises(ValueError):
                process(bad_raw, 1)
        for bad_radius in (0., -RADIUS, float('nan'), float('inf')):
            with self.subTest(radius=bad_radius), self.assertRaises(ValueError):
                process(raw, 1, radius=bad_radius)
        invalid = raw.copy(); invalid[1, 0, 0] = np.inf
        with self.assertRaises(ValueError): process(invalid, 1)


if __name__ == '__main__':
    unittest.main()
