"""Small regression tests for portable bookkeeping; no reconstruction/training."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from common import dump, mesh_audit, save_mesh, sha
from exact_audit import exact_mesh
from finalize_audit import (finalize, historical_thin_u_comparison,
                            protocol_slots, summarize_rows, validate_slots,
                            vertex_link_witnesses)


def protocol(objects, states, groups=None):
    config = dict(version='TEST', run_states=states)
    if groups is not None:
        config['state_groups'] = groups
    return dict(config=config, objects={name: {} for name in objects})


def pinched_tetrahedra():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    first = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    second = np.array([0, 4, 5, 6])[first][:, ::-1]
    return v, np.vstack([first, second])


def write_run_skeleton(root, frozen, rows):
    for name in ['vendor', 'inputs', 'runs']:
        (root/name).mkdir()
    # A test fixture with self-consistent provenance, never called an extractor.
    source = b'# unit-test provenance fixture\n'
    (root/'vendor/PoNQ_to_mesh.py').write_bytes(source)
    blob = hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
    dump(root/'SOURCE_PROVENANCE.json', dict(official_extractor_blob=blob))
    dump(root/'FROZEN_PROTOCOL.json', frozen)
    dump(root/'FROZEN_PROTOCOL_SHA256.json', dict(sha256=sha(root/'FROZEN_PROTOCOL.json')))
    dump(root/'LEDGER_RAW.json', rows)


class FinalizeAuditTests(unittest.TestCase):
    def test_dynamic_slots_cover_partial_real_and_arbitrary_states(self):
        for object_count, states, expected in [
                (8, ['clean128', 'clean256', 'position_light', 'position_strong',
                     'normal_light', 'normal_strong', 'qem_light', 'qem_strong', 'joint_light'], 72),
                (4, ['s'+str(i) for i in range(9)], 36),
                (2, ['identity', 'metric_only'], 4)]:
            frozen = protocol(['object'+str(i) for i in range(object_count)], states)
            rows = [dict(object=o, state=s, status='NOT_RUN_MISSING_RAW_ASSET')
                    for o, s in sorted(protocol_slots(frozen))]
            self.assertEqual(len(validate_slots(rows, frozen)), expected)
            with self.assertRaises(ValueError):
                validate_slots(rows[:-1], frozen)
            with self.assertRaises(ValueError):
                validate_slots(rows + [rows[0]], frozen)

    def test_failure_invalid_and_unrun_counts_are_distinct(self):
        frozen = protocol(['shape'], ['identity', 'light', 'failed', 'invalid', 'missing'],
                          dict(clean=['identity'], mild=['light']))
        rows = [
            dict(object='shape', state='identity', status='EXTRACTED', extraction_seconds=1.,
                 geometry_audited=True, betti_match=True, closed_oriented_2manifold=True),
            dict(object='shape', state='light', status='EXTRACTED', extraction_seconds=2.,
                 geometry_audited=True, betti_match=False, nonmanifold_vertices=1),
            dict(object='shape', state='failed', status='EXTRACTION_OR_EVALUATION_FAILED'),
            dict(object='shape', state='invalid', status='EXTRACTED_INVALID_CUT_WEIGHTS', extraction_seconds=3.),
            dict(object='shape', state='missing', status='NOT_RUN_MISSING_RAW_ASSET'),
        ]
        summary = summarize_rows(rows, frozen)
        for key, expected in dict(requested_slots=5, actual_extractions=3, successful_extractions=2,
                                  execution_failures=1, invalid_extractions=1, unrun_missing_assets=1,
                                  clean_extractions=1, perturbed_extractions=1, mild_output_count=1,
                                  mild_topology_matching_outputs=0, outputs_with_nonmanifold_vertices=1).items():
            self.assertEqual(summary[key], expected, key)

    def test_vertex_witness_survives_reindexing(self):
        v, f = pinched_tetrahedra()
        witnesses = vertex_link_witnesses(v, f)
        self.assertEqual(len(witnesses), 1)
        self.assertTrue(witnesses[0]['two_disjoint_cycles'])
        permutation = np.array([4, 2, 6, 1, 5, 0, 3])
        inverse = np.argsort(permutation)
        moved = vertex_link_witnesses(v[permutation], inverse[f])
        self.assertEqual(moved[0]['vertex_id'], int(inverse[0]))
        self.assertEqual(moved[0]['link_component_count'], 2)
        self.assertTrue(moved[0]['two_disjoint_cycles'])
        self.assertNotEqual(moved[0]['vertex_id'], witnesses[0]['vertex_id'])

    def test_changed_historical_geometry_is_nonblocking(self):
        rows = [dict(object='thin_u', state='qem_strong', status='EXTRACTED',
                     beta0=1, beta1=0, beta2=1)]
        result = historical_thin_u_comparison(rows, {'thin_u__qem_strong': []})
        self.assertFalse(result['matches_original_signature'])
        self.assertFalse(result['blocking'])

    def test_all_failed_and_missing_finalize_without_nan(self):
        frozen = protocol(['shape'], ['attempt', 'absent'])
        rows = [dict(object='shape', state='attempt', status='EXTRACTION_OR_EVALUATION_FAILED'),
                dict(object='shape', state='absent', status='NOT_RUN_MISSING_RAW_ASSET')]
        with tempfile.TemporaryDirectory(prefix='ponq-finalize-test-') as temp:
            root = Path(temp)
            write_run_skeleton(root, frozen, rows)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = finalize(root)
            self.assertEqual(summary['actual_extractions'], 0)
            self.assertEqual(summary['execution_failures'], 1)
            self.assertIsNone(summary['extraction_seconds_median'])
            saved = json.loads((root/'SUMMARY.json').read_text())
            self.assertEqual(saved['mild_output_count'], 0)
            self.assertTrue(json.loads((root/'audit/POSTRUN_CONSISTENCY_CHECKS.json').read_text())['passed'])

    def test_nonmanifold_output_is_recorded_not_rejected_and_hash_retained(self):
        v, f = pinched_tetrahedra()
        audit = mesh_audit(v, f)
        row = dict(object='thin_u', state='qem_strong', status='EXTRACTED',
                   reference_betti=[1, 0, 1], extraction_seconds=.01, **audit)
        frozen = protocol(['thin_u'], ['qem_strong'])
        with tempfile.TemporaryDirectory(prefix='ponq-finalize-test-') as temp:
            root = Path(temp)
            write_run_skeleton(root, frozen, [row])
            rd = root/'runs/thin_u__qem_strong'
            rd.mkdir()
            save_mesh(rd/'mesh.obj', v, f)
            np.savez(rd/'mesh.npz', vertices=v, faces=f)
            row['mesh_sha256'] = sha(rd/'mesh.obj')
            dump(rd/'exact_audit.json', dict(source_sha256=row['mesh_sha256'], **exact_mesh(v, f)))
            dump(root/'LEDGER_RAW.json', [row])
            # A dynamically named reference checks the generic input scan too.
            reference = root/'inputs/new_name'
            reference.mkdir()
            save_mesh(reference/'reference.obj', v[:4], f[:4])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = finalize(root)
            self.assertEqual(summary['outputs_with_nonmanifold_vertices'], 1)
            self.assertEqual(summary['execution_failures'], 0)
            self.assertEqual(summary['topology_matching_outputs'], 0)
            self.assertFalse(summary['historical_thin_u_comparison']['matches_original_signature'])
            checks = json.loads((root/'audit/POSTRUN_CONSISTENCY_CHECKS.json').read_text())
            self.assertTrue(checks['passed'])
            self.assertEqual(checks['audited_reference_count'], 1)
            merged = json.loads((root/'LEDGER.json').read_text())[0]
            self.assertEqual(merged['mesh_sha256'], row['mesh_sha256'])
            self.assertEqual(merged['exact_intersection_pairs'], 0)
            # Corrupted saved output remains an implementation/provenance error.
            with (rd/'mesh.obj').open('a') as handle:
                handle.write('# tampered bytes\n')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
                finalize(root)


if __name__ == '__main__':
    unittest.main()
