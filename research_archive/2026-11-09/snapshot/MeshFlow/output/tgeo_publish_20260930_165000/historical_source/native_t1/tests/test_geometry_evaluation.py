"""CPU registration/selection/score tests; no model forward or training."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from native_t1.geometry_evaluation import (select_new_patches, prepare_evaluation,
                                          load_evaluation, score_output, binding_pair)


def grid_source():
    vertices = np.asarray([[x, 0., z] for z in range(8) for x in range(9)], dtype=np.float32)
    faces = []
    for z in range(7):
        for x in range(8):
            a = z*9+x
            faces.extend([[a, a+1, a+9], [a+1, a+10, a+9]])
    return vertices, np.asarray(faces, dtype=np.int64)


def write_inputs(path):
    vertices, ids = grid_source(); arrays = {}
    raw_ids = np.arange(111, -1, -1, dtype=np.int64)
    known, free = np.arange(12, dtype=np.int64), np.arange(12, 112, dtype=np.int64)
    for p in (0, 1):
        v = vertices + np.asarray([0., .4*p, 0.], dtype=np.float32)
        target = v[ids].reshape(112, 9); pre = (target/2).reshape(112, 3, 3)
        arrays.update({f'parent{p}_target':target, f'parent{p}_pre_ot':pre,
                       f'parent{p}_source_vertices':v, f'parent{p}_source_face_vertex_ids':ids,
                       f'parent{p}_canonical_to_raw_face':raw_ids})
        prefix = f'parent{p}_K12_'
        arrays.update({prefix+'constraints':target[known].reshape(12,3,3), prefix+'free_target':target[free],
                       prefix+'source_face_ids':known, prefix+'free_source_ids':free,
                       prefix+'pre_ot_free':pre[free]})
    np.savez(path, **arrays)


class GeometryEvaluationTests(unittest.TestCase):
    def test_raw_id_bfs_order_novelty_and_exact_complement(self):
        vertices, ids = grid_source(); target = vertices[ids].reshape(112,9)
        mapping = np.arange(111, -1, -1, dtype=np.int64)
        patches, audit = select_new_patches(target, ids, mapping, np.arange(12))
        self.assertEqual(audit['status'], 'PASS')
        self.assertEqual(patches[0]['seed_raw_face_id'], 0)
        self.assertEqual(patches[0]['seed_canonical_face_id'], 111)
        self.assertNotEqual(set(patches[0]['canonical_face_ids']), set(patches[1]['canonical_face_ids']))
        for selected in patches:
            faces = selected['canonical_face_ids']
            self.assertEqual(len(faces), 12)
            self.assertGreaterEqual(len(set(faces)-set(range(12))), 6)
            self.assertEqual(selected['raw_face_ids'], mapping[faces].tolist())
            visited = {faces[0]}
            for face in faces[1:]:
                self.assertTrue(any(len(set(ids[face]) & set(ids[prior])) == 2 for prior in visited))
                visited.add(face)

    def test_vertex_only_connections_and_shortfall_are_not_replaced(self):
        vertices, ids = grid_source(); target = vertices[ids].reshape(112,9)
        disconnected = np.column_stack([np.zeros(112, dtype=np.int64),
                                       np.arange(1,225,2), np.arange(2,225,2)])
        selected, audit = select_new_patches(target, disconnected, np.arange(112), np.arange(12))
        self.assertEqual(selected, [])
        self.assertEqual(audit['status'], 'INSUFFICIENT_PATCHES')
        with self.assertRaises(ValueError):
            select_new_patches(target, ids, np.zeros(112, dtype=np.int64), np.arange(12))

    def test_registration_reload_and_patch_specific_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'inputs.npz'; write_inputs(source)
            bank, audit = prepare_evaluation(source, root/'registration')
            self.assertEqual(audit['status'], 'PASS')
            self.assertEqual(audit['formal_rollout_count'], 64)
            self.assertEqual(len(bank['cases']), 6)
            loaded, loaded_audit = load_evaluation(audit['registration']['path'], audit['registration']['sha256'])
            self.assertEqual(audit, loaded_audit)
            for key, value in bank['arrays'].items():
                self.assertEqual(value.tobytes(), loaded['arrays'][key].tobytes())
                self.assertFalse(loaded['arrays'][key].flags.writeable)
            for p in (0,1):
                old_ids = set(bank['cases'][p,0].source_face_ids.tolist())
                for j in (1,2):
                    case = bank['cases'][p,j]
                    self.assertNotEqual(set(case.source_face_ids.tolist()), old_ids)
                    expected = [i for i in range(112) if i not in case.source_face_ids]
                    self.assertEqual(case.free_source_ids.tolist(), expected)
                    np.testing.assert_array_equal(case.free_target, case.full_target[expected])
            results=[]
            for p in (0,1):
                case=loaded['cases'][p,1]
                raw=np.concatenate([case.constraints.reshape(12,9),case.free_target])
                result=score_output(raw,loaded,p,1,model_name='fixture',noise_seed=9701)
                self.assertLess(result['remainder']['own']['symmetric_rms_bbox_pct'],1e-10)
                self.assertEqual(result['independent_full_geometry_audit']['status'],'NOT_RUN')
                self.assertEqual(result['free_faces'],100)
                results.append(result)
            binding=binding_pair(*results,patch_index=1)
            self.assertTrue(binding['both_directions_positive'])
            self.assertEqual(binding['context_pair'],['P0_patch1_K12','P1_patch1_K12'])
            with self.assertRaises(ValueError): binding_pair(results[0],dict(results[1],patch_index=2))
            with self.assertRaises(ValueError): binding_pair(results[0],dict(results[1],noise_seed=9702))
            with self.assertRaises(FileExistsError): prepare_evaluation(source,root/'registration')
            with self.assertRaises(ValueError): load_evaluation(audit['registration']['path'],'0'*64)


if __name__ == '__main__':
    unittest.main()