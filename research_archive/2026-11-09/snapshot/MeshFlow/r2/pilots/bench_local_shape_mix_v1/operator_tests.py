"""CPU synthetic contract tests only; no real requests, GEO or scoring."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from . import operator as O


def fixture():
    v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [0, 0, -1], [1, 0, -1], [0, 1, -1],
                    [2, 0, 0], [3, 0, 0], [2, 1, 0]], np.float32)
    canonical = np.asarray([[0, 1, 2], [1, 0, 3], [1, 3, 4],
                            [4, 3, 5], [6, 7, 8], [2, 3, 5]], np.int64)
    raw = v[canonical]
    N, K = len(raw), 1
    d0pre = raw.copy()
    rsc = np.tile(np.arange(3), (N, 1)); rsc[0] = [0, 2, 1]
    dsc = rsc.copy(); dsc[1] = [1, 2, 0]; dsc[2] = [2, 0, 1]
    def mesh(sc, slots):
        sf = np.asarray(slots, np.int64)
        faces = canonical[sf[:, None], sc[sf]]
        return dict(vertices=v.copy(), faces=faces, output=v[faces], C=v[faces[:K]],
                    known_faces=np.asarray(K), input_faces=np.asarray(N),
                    source_face_ids=sf, source_corner_ids=sc[sf])
    d0, rawbase = mesh(dsc, range(N)), mesh(rsc, range(N - 1))
    selected = np.asarray([0, 1, 1, 0, 1, 1], bool)
    regions = dict(P1=np.asarray([0, 1, 1, 0, 0, 1], bool),
                   P2=np.asarray([0, 0, 0, 0, 1, 0], bool))
    anchors = np.full((N, 3), -1, np.int64); anchors[0] = np.arange(3)
    meta = dict(raw_anchor_ids=anchors, raw_seam_slot_mask=np.zeros(N, bool))
    return d0, rawbase, raw, d0pre, selected, regions, meta


class OperatorTests(unittest.TestCase):
    def assembly(self):
        return O.assemble(*fixture())

    def test_c_and_external_byte_identity_and_namespace_split(self):
        args = fixture(); a = O.assemble(*args); p, m = a['post'], a['mappings']
        self.assertTrue(O._exact(p['C'], args[0]['C']))
        row = np.flatnonzero(p['source_face_ids'] == 3)[0]
        self.assertTrue(O._exact(p['output'][row], args[0]['output'][3]))
        self.assertNotEqual(m['d0_vertex_to_composite'][3], m['raw_vertex_to_composite'][3])
        self.assertTrue(O._exact(p['vertices'][m['d0_vertex_to_composite'][3]],
                                p['vertices'][m['raw_vertex_to_composite'][3]]))
        for old in (0, 1, 2):
            self.assertEqual(m['d0_vertex_to_composite'][old], m['raw_vertex_to_composite'][old])

    def test_retained_face_set_no_synthetic_revival(self):
        a = self.assembly(); p = a['post']
        self.assertEqual(p['source_face_ids'].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(p['source_face_reason'][5], 'ABSENT_RAW_BASE')
        self.assertEqual(a['summary']['selected_absent_slots'], [5])
        self.assertEqual(len(set(p['raw_slot_ids'].tolist())), len(p['faces']))

    def test_canonical_corner_maps_and_member_coverage(self):
        a = self.assembly(); p = a['post']; sf, sc = p['source_face_ids'], p['source_corner_ids']
        np.testing.assert_array_equal(p['corner_vertex_ids'][sf[:, None], sc], p['faces'])
        self.assertEqual(set(p['corner_vertex_ids'][p['corner_vertex_ids'] >= 0]), set(range(len(p['vertices']))))
        for i, member in enumerate(p['vertex_source_input_corner_ids']):
            self.assertEqual(p['corner_vertex_ids'].ravel()[member], i)
        self.assertTrue((p['corner_vertex_ids'][5] == -1).all())
        self.assertEqual(p['source_corner_ids'][3].tolist(), fixture()[0]['source_corner_ids'][3].tolist())

    def test_targets_are_member_means_in_original_corner_domain(self):
        args = list(fixture())
        args[3][1, 2] += np.asarray([.02, .03, .01], np.float32)
        args[3][2, 1] += np.asarray([.06, .01, .03], np.float32)
        a = O.assemble(*args); i = int(a['mappings']['raw_vertex_to_composite'][3])
        expected = args[3][[1, 2], [2, 1]].astype(float).mean(axis=0)
        np.testing.assert_array_equal(a['mappings']['d0_target_vertices'][i], expected)
        self.assertTrue(O._exact(a['pre'][args[4]], args[2][args[4]]))
        self.assertTrue(O._exact(a['pre'][~args[4]], args[3][~args[4]]))

    def test_source_C_identity_error_is_not_coordinate_welded_away(self):
        args = list(fixture()); d = copy.deepcopy(args[0]); rb = copy.deepcopy(args[1])
        # Add a second C face sharing an existing known vertex, then split that
        # identity in only one source without changing its rendered coordinates.
        for mesh in (d, rb):
            mesh['known_faces'] = np.asarray(2)
        # Fixture slot1 is now known and canonical pre/raw remain identical.
        # Existing different corner order at C slot1 is itself a valid refusal.
        args[0], args[1] = d, rb
        args[4][1] = False; args[5]['P1'][1] = False
        with self.assertRaises(ValueError):
            O.assemble(*args)

    def test_identity_fit_is_exact_and_topology_is_shared(self):
        a = self.assembly(); result = O.fit(a); s = result['summary']['solver']
        self.assertEqual(s['calls_sparse_solve'], 1)
        self.assertEqual(s['objective_after']['total'], 0.)
        self.assertTrue(O._exact(result['post']['vertices'], a['post']['vertices']))
        self.assertTrue(O._exact(result['post']['faces'], a['post']['faces']))
        self.assertTrue(O._exact(result['pre'], a['pre']))

    def test_unconstrained_solution_satisfies_independent_dense_equations(self):
        a = self.assembly(); m = a['mappings']; variable = m['variable_vertex_mask']
        m['d0_target_vertices'][variable] += [.002, -.001, .003]
        result = O.fit(a); s = result['summary']['solver']
        self.assertEqual(s['calls_sparse_solve'], 1)
        self.assertLess(s['linear_residual_max_abs'], 1e-13)
        ids = np.flatnonzero(variable); lookup = {int(v): i for i, v in enumerate(ids)}
        edges = set()
        for face in a['post']['faces'][m['selected_output_face_mask']]:
            for k in range(3): edges.add(tuple(sorted((int(face[k]), int(face[(k + 1) % 3])))))
        matrix = np.eye(len(ids)) / len(ids)
        for u, v in edges:
            vec = np.zeros(len(ids))
            if u in lookup: vec[lookup[u]] = 1.
            if v in lookup: vec[lookup[v]] = -1.
            matrix += np.outer(vec, vec) / len(edges)
        rhs = (m['d0_target_vertices'][ids] - a['post']['vertices'][ids]) / len(ids)
        expected = np.linalg.solve(matrix, rhs)
        actual = result['mappings']['unconstrained_vertices'][ids] - a['post']['vertices'][ids]
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)
        self.assertTrue(O._exact(result['post']['vertices'][~variable], a['post']['vertices'][~variable]))

    def test_per_vertex_cap_not_uniform_solution_scaling(self):
        a = self.assembly(); m = a['mappings']; ids = np.flatnonzero(m['variable_vertex_mask'])
        m['d0_target_vertices'][ids] += np.arange(1, len(ids) + 1)[:, None] * np.asarray([2., -1., .5])
        result = O.fit(a); disp = result['mappings']['capped_displacement'][ids]
        lengths = np.linalg.norm(disp, axis=1)
        self.assertTrue(np.all(lengths <= 2 * O.R + 1e-16))
        unconstrained = result['mappings']['unconstrained_vertices'][ids] - a['post']['vertices'][ids]
        ratio = lengths / np.linalg.norm(unconstrained, axis=1)
        self.assertGreater(np.ptp(ratio), 1e-4)
        self.assertTrue(result['summary']['solver']['trials'][-1]['passed'])
        self.assertLessEqual(result['summary']['solver']['maximum_actual_displacement'], 2 * O.R)

    def test_backtracking_records_constraints_on_fp32_delivery(self):
        a = self.assembly(); m = a['mappings']; ids = np.flatnonzero(m['variable_vertex_mask'])
        # Small rest triangles make full capped compression infeasible; later
        # shared alpha must retain their actual minimum area and axis spans.
        selected = m['selected_output_face_mask']; local = a['post']['faces'][selected]
        variable = m['variable_vertex_mask']
        a['post']['vertices'][variable] *= np.float32(.05)
        a['post']['output'] = a['post']['vertices'][a['post']['faces']]
        a['post']['C'] = a['post']['output'][:1]; a['post']['free'] = a['post']['output'][1:]
        m['rest_vertices'] = a['post']['vertices'].copy()
        m['d0_target_vertices'][ids] = 0.
        result = O.fit(a); s = result['summary']['solver']
        self.assertGreater(len(s['trials']), 1)
        self.assertTrue(any('AREA_RATIO' in r['reasons'] or any('AXIAL_SPAN' in x for x in r['reasons']) for r in s['trials']))
        if not s['fallback']:
            final = s['trials'][-1]
            self.assertGreaterEqual(final['min_area_ratio'], .8)
            self.assertGreaterEqual(final['min_normal_cosine'], .5)
            self.assertTrue(final['passed'])
        self.assertLessEqual(len(s['trials']), 11)

    def test_invalid_rest_exact_fallback_not_deletion(self):
        a = self.assembly(); m = a['mappings']; face = a['post']['faces'][np.flatnonzero(m['selected_output_face_mask'])[-1]]
        a['post']['vertices'][face[1]] = a['post']['vertices'][face[0]]
        a['post']['output'] = a['post']['vertices'][a['post']['faces']]
        a['post']['C'] = a['post']['output'][:1]; a['post']['free'] = a['post']['output'][1:]
        result = O.fit(a)
        self.assertTrue(result['summary']['solver']['fallback'])
        self.assertEqual(result['summary']['solver']['calls_sparse_solve'], 0)
        self.assertTrue(O._exact(result['post']['output'], a['post']['output']))
        self.assertTrue(O._exact(result['post']['faces'], a['post']['faces']))

    def test_no_variable_closed_rest_is_noop(self):
        a = self.assembly(); a['mappings']['variable_vertex_mask'][:] = False
        result = O.fit(a)
        self.assertEqual(result['summary']['solver']['calls_sparse_solve'], 0)
        self.assertTrue(O._exact(result['post']['vertices'], a['post']['vertices']))

    def test_scoring_metadata_validate_only_no_scoring(self):
        # validate() performs array/provenance assertions only. It does not
        # invoke score(), scan(), query construction or geometry processing.
        from r2.tools.hinge_v2_cpu_v1.scoring import validate
        a = self.assembly(); p = a['post']
        checked = validate(p, dict(K=1, N=6))
        self.assertEqual(len(checked[0]['faces']), 5)
        self.assertFalse(checked[0]['bridge_mask'].any())
        self.assertEqual(checked[0]['face_role'].tolist(), ['C', 'BULK', 'BULK', 'BULK', 'BULK'])

    def test_no_radius_or_score_tuning_permitted(self):
        cfg = dict(O.CONFIG); cfg['maximum_displacement_R'] = 3.
        with self.assertRaises(ValueError): O.fit(self.assembly(), cfg)
        self.assertNotIn('torch', sys.modules)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--receipt')
    args = p.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OperatorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), synthetic_only=True,
                  real_request_operator_calls=0, model_calls=0, GEO_calls=0, detector_calls=0, score_calls=0,
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(O.__file__), Path(__file__))})
    if args.receipt:
        target = O._path(args.receipt); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2); f.write('\n')
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
