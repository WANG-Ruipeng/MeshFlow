"""CPU-only scientific plumbing tests; no sampling, D0 or scoring calls."""
import pickle
import unittest
import numpy as np
from . import evaluation as E
from .post import shared_index_audit,projection_audit


class EvaluationContract(unittest.TestCase):
    def test_resource_formats_preserve_headroom_rule(self):
        flat=dict(gpu_name='CPU-contract',total_mib=16000,used_mib=4000,free_mib=12000,gpu_util_percent=20)
        E.dispatch_guard(flat,3500)
        E.dispatch_guard({'gpu':{'returncode':0,'stdout':'CPU-contract,16000,4000,12000,20'}},3500)
        flat['free_mib']=5000
        with self.assertRaisesRegex(RuntimeError,'RESOURCE_WAIT'):E.dispatch_guard(flat,3500)
        with self.assertRaises(ValueError):E.dispatch_guard({'gpu_name':'missing'},3500)
    def test_private_namespace_and_worker_pickle(self):
        from r2.pilots.bench_four_routes_v1.exposure import evaluation as old
        before=old.OUT
        identity=E.cpu_dispatch_identity('PATCH200')
        self.assertTrue(identity['original_root_unchanged'] and identity['new_root_preserved'])
        self.assertEqual(before,old.OUT)
        self.assertIs(pickle.loads(pickle.dumps(E._cpu_worker)),E._cpu_worker)

    def test_original_12_request_subset_and_budgets(self):
        qs=E.requests()
        self.assertEqual(len(qs),12)
        self.assertEqual(len({q['parent_id'] for q in qs}),6)
        self.assertEqual(E.budget()['new_RAW'],72)
        self.assertEqual(E.budget()['total_score_rows'],252)
        self.assertEqual(E.label_spec('PLAIN267')['global_step'],25767)
        with self.assertRaises(ValueError):E.label_spec('EXPOSURE')

    def test_source_corner_permutation_controls_shared_audit(self):
        # Raw slot2 corner0 appears in output column2, slot3 corner1 in column0.
        post=dict(source_face_ids=np.array([2,3]),source_corner_ids=np.array([[1,2,0],[1,0,2]]),
            faces=np.array([[6,8,4],[4,9,7]]))
        r=shared_index_audit(post,np.array([[6,10]]))
        self.assertEqual(r['retained_shared'],1)
        post['faces'][1,0]=100
        with self.assertRaises(ValueError):shared_index_audit(post,np.array([[6,10]]))

    def test_deleted_face_not_claimed_shared(self):
        post=dict(source_face_ids=np.array([2]),source_corner_ids=np.array([[0,1,2]]),faces=np.array([[4,6,8]]))
        r=shared_index_audit(post,np.array([[6,10]]))
        self.assertEqual(r['retained_shared'],0)
        self.assertEqual(r['pairs_with_deleted_face'],1)

    def test_projection_damage_not_hidden(self):
        before=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,1],[1,0,1],[0,1,1]]],np.float32)
        after=before.copy();after[1,2]=after[1,0]
        r=projection_audit(before,after,1,np.array([[3,5]]))
        self.assertFalse(r['C_changed']);self.assertEqual(r['newly_degenerate_faces'],[1])
        self.assertGreater(r['max_free_corner_displacement_L'],0)


if __name__=='__main__':unittest.main()
