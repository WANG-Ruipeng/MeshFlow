"""Bounded tests for the safety adapter, using saved Gate 1 parameters."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import torch
from safe_extract import (COS_TOLERANCE, MeshFromPoNQ, SafeExtractionError,
                          SafeMeshFromPoNQ, extract, guarded_arccos)

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3] / 'output'
OUTPUT = Path(os.environ['PONQ_SAFE_VALIDATION_DIR']) if os.environ.get('PONQ_SAFE_VALIDATION_DIR') else None
RECORDS = []


def load_case(study, name, state):
    rd = ROOT/f'ponq_gate1_{study}_20260926_01/runs/{name}__{state}'
    with np.load(rd/'parameters.npz', allow_pickle=False) as archive:
        params = {k: archive[k].copy() for k in archive.files}
    return rd, params


def record_case(name, vertices, faces, ids, trace, audit, comparison):
    record = dict(case=name, audit=audit, comparison=comparison)
    RECORDS.append(record)
    if OUTPUT is not None:
        dest = OUTPUT/name
        dest.mkdir()
        np.savez_compressed(dest/'mesh.npz', vertices=vertices, faces=faces, input_vertex_ids=ids)
        np.savez_compressed(dest/'trace.npz', **trace)
        (dest/'AUDIT.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


class SafeExtractTests(unittest.TestCase):
    def test_arccos_clips_only_roundoff_boundary(self):
        dots = np.array([-1.-np.finfo(float).eps, -.2, 1.+np.finfo(float).eps])
        angle, audit = guarded_arccos(dots)
        self.assertTrue(np.isfinite(angle).all())
        self.assertEqual(audit['clipped_count'], 2)
        self.assertEqual(angle[0], np.pi)
        self.assertEqual(angle[-1], 0.)
        self.assertEqual(angle[1], np.arccos(-.2))
        for invalid in [np.array([1.+2*COS_TOLERANCE]), np.array([np.nan]), np.array([np.inf])]:
            with self.assertRaises(SafeExtractionError):
                guarded_arccos(invalid)

    def test_zero_or_nonfinite_input_normal_is_rejected(self):
        v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        params = dict(v=v, n=np.ones((4,3)), Q=np.tile(np.eye(4),(4,1,1)), eigs=np.ones((4,3)))
        for replacement in [np.zeros(3), np.array([np.nan,0.,1.])]:
            params['n'][0] = replacement
            with self.assertRaises(SafeExtractionError):
                extract(params)

    def test_actual_bad_graph_is_rejected_before_original_solver(self):
        for bad in [-1., np.nan]:
            stub = SimpleNamespace(
                final_scores=np.array([bad, 1.]), tet_colors=np.array([4,0,1]),
                triangle_faces_neighbors=np.array([[0,2],[2,1]]),
                triangle_faces=np.array([[0,1,2],[0,2,3]]),
                trace_arrays={}, audit=dict(original_mincut_called=False))
            with self.assertRaises(SafeExtractionError):
                SafeMeshFromPoNQ.min_cut(stub)
            self.assertFalse(stub.audit['original_mincut_called'])
            self.assertIn('graph_capacities', stub.trace_arrays)

    def test_real_speaker2_nan_controls_become_valid_without_q_or_capacity_clips(self):
        for state, old_nan in [('clean128',92),('clean256',166)]:
            rd, params = load_case('real','loudspeaker_local_2',state)
            saved = json.loads((rd/'result.json').read_text())
            self.assertEqual(saved['nonfinite_cut_weights'], old_nan)
            original_arrays = {k: params[k].copy() for k in ('v','n','Q','eigs')}
            vertices, faces, ids, trace, audit = extract(params)
            self.assertGreater(audit['arccos_clipped_count'], 0)
            self.assertEqual(audit['arccos_calls'][0]['faces_with_clipped_dot'], old_nan)
            self.assertEqual(audit['candidate_score_stats']['nonfinite_count'],0)
            self.assertEqual(audit['graph_capacity_stats']['nonfinite_count'],0)
            self.assertEqual(audit['graph_capacity_stats']['negative_count'],0)
            self.assertTrue(audit['original_mincut_called'])
            self.assertFalse(audit['quadric_repair_performed'])
            self.assertFalse(audit['capacity_repair_performed'])
            for key, old in original_arrays.items():
                self.assertTrue(np.array_equal(params[key],old),key)
            self.assertTrue(np.array_equal(vertices,trace['delaunay_points'][ids]))
            self.assertTrue(np.array_equal(trace['surface_faces_input_ids'],ids[faces]))
            with np.load(rd/'mesh.npz',allow_pickle=False) as old:
                comparison=dict(saved_status=saved['status'],saved_candidate_nan=old_nan,
                    vertices_equal=np.array_equal(vertices,old['vertices']),
                    faces_equal=np.array_equal(faces,old['faces']),
                    old_faces=len(old['faces']),new_faces=len(faces))
            record_case('speaker2__'+state,vertices,faces,ids,trace,audit,comparison)

    def test_sphere_and_torus_controls_match_original_arrays(self):
        for name in ['sphere','torus']:
            rd, params = load_case('synthetic',name,'clean256')
            vertices, faces, ids, trace, audit = extract(params)
            with warnings.catch_warnings(record=True):
                official = MeshFromPoNQ(torch.from_numpy(params['v']),torch.from_numpy(params['eigs']),
                    torch.from_numpy(params['Q']),torch.from_numpy(params['n']),grid_scale=32,
                    correct_tet_color=True,compute_mincut=True)
                ov,of,oi=official.get_surface(return_indices=True)
            self.assertEqual(audit['arccos_clipped_count'],0)
            self.assertTrue(np.array_equal(vertices,ov))
            self.assertTrue(np.array_equal(faces,of))
            self.assertTrue(np.array_equal(ids,oi))
            self.assertTrue(np.array_equal(trace['candidate_scores'],official.final_scores))
            with np.load(rd/'mesh.npz',allow_pickle=False) as old:
                self.assertTrue(np.array_equal(vertices,old['vertices']))
                self.assertTrue(np.array_equal(faces,old['faces']))
            self.assertTrue(audit['inside_seeds_preserved'])
            self.assertTrue(audit['outside_seeds_preserved'])
            record_case(name+'__clean256',vertices,faces,ids,trace,audit,
                        dict(saved_arrays_equal=True,official_replay_arrays_equal=True,
                             official_input_ids_equal=True,official_candidate_scores_equal=True))


if __name__ == '__main__':
    if OUTPUT is not None:
        OUTPUT.mkdir(parents=True, exist_ok=False)
    result = unittest.main(verbosity=2,exit=False).result
    if OUTPUT is not None:
        vendor = Path(__file__).resolve().parents[1]/'vendor/PoNQ_to_mesh.py'
        document=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            passed=result.wasSuccessful(),cases=RECORDS,source_files=[
                dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [
                    Path(__file__),Path(__file__).with_name('safe_extract.py'),vendor]])
        (OUTPUT/'VALIDATION.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
        files=[p for p in sorted(OUTPUT.rglob('*')) if p.is_file()]
        (OUTPUT/'MANIFEST.json').write_text(json.dumps(dict(files=[
            dict(path=str(p.relative_to(OUTPUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in files]),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
