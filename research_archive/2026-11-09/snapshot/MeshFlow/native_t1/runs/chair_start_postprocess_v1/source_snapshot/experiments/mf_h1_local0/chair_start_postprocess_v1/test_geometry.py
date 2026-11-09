"""Fixed synthetic CPU contracts. No formal RAW, GT, model or GPU is read."""
from __future__ import annotations

import hashlib
import inspect
import json
import time
import unittest

import numpy as np

from .geometry import FIXED_CONFIG, RADIUS, process


def mesh(*triangles):
    return np.asarray(triangles, dtype=np.float32)


def known():
    return [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]


def soup_key(triangles):
    return sorted(tuple(sorted(tuple(float(v) for v in p) for p in tri)) for tri in triangles)


class GeometryContracts(unittest.TestCase):
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


def run():
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeometryContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    from pathlib import Path
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), Path(__file__).with_name('geometry.py'))}
    receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL',
                   synthetic_test_cases=result.testsRun, fixture_count=8,
                   scope='Synthetic CPU only; no formal RAW, GT, model, CUDA or files written',
                   source_sha256=hashes, fixed_config=FIXED_CONFIG,
                   seconds=time.perf_counter() - start, model_calls=0)
    print(json.dumps(receipt, sort_keys=True))
    return result.wasSuccessful()


if __name__ == '__main__':
    raise SystemExit(0 if run() else 1)
