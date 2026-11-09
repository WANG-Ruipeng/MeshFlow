"""Synthetic mathematical validation for incremental full-quadric transport."""
import hashlib
import json
import os
from pathlib import Path
import unittest
import numpy as np
from motion_jacobian import prepare_neighborhood
from incremental_transport import refresh_neighborhood, step_transport

DETAILS=[]


def affine_matrix(A,b):
    H=np.eye(4);H[:3,:3]=A;H[:3,3]=b
    return H


def apply(H,x):
    return x@H[:3,:3].T+H[:3,3]


def congruence(Q,H):
    inv=np.linalg.inv(H)
    return inv.T@Q@inv


class IncrementalTransportTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(519)
        self.x=rng.normal(size=(80,3))
        factors=rng.normal(size=(80,4,4))
        self.Q=factors@factors.transpose(0,2,1)
        self.nb=prepare_neighborhood(self.x)

    def test_translation_complete_quadric_and_no_mutation(self):
        x=np.random.default_rng(7).integers(-32,33,size=(80,3))/32.
        nb=prepare_neighborhood(x)
        delta=np.array([.25,-.5,1.])
        old=self.Q.copy();old_x=x.copy()
        result,H,estimate,current_nb=step_transport(self.Q,x,x+delta,nb)
        expected_H=affine_matrix(np.eye(3),delta)
        self.assertTrue(np.array_equal(H,np.broadcast_to(expected_H,H.shape)))
        self.assertTrue(np.array_equal(result,congruence(self.Q,expected_H)))
        self.assertTrue(np.array_equal(self.Q,old))
        self.assertTrue(np.array_equal(x,old_x))
        self.assertTrue(np.array_equal(estimate['arrays']['used_sigma_min'],np.ones(len(x))))
        self.assertEqual(estimate['stats']['fallback_count'],0)

    def test_rigid_rotation_and_quadric_energy_invariance(self):
        a=.31
        A=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
        true_H=affine_matrix(A,np.array([.13,-.07,.04]))
        result,H,estimate,_=step_transport(self.Q,self.x,apply(true_H,self.x),self.nb)
        self.assertLess(np.max(abs(H-true_H)),1e-13)
        self.assertLess(np.max(abs(result-congruence(self.Q,true_H))),1e-12)
        points=np.c_[self.x+.1,np.ones(len(self.x))]
        moved=points@true_H.T
        before=np.einsum('ni,nij,nj->n',points,self.Q,points)
        after=np.einsum('ni,nij,nj->n',moved,result,moved)
        self.assertLess(np.max(abs(before-after)),1e-11)
        self.assertLess(np.max(abs(estimate['arrays']['used_sigma_min']-1.)),1e-13)

    def test_full_rank_affine_inverse_congruence(self):
        A=np.array([[1.1,.12,-.03],[.02,.83,.09],[.04,-.06,1.2]])
        true_H=affine_matrix(A,np.array([.2,-.1,.07]))
        result,H,estimate,_=step_transport(self.Q,self.x,apply(true_H,self.x),self.nb)
        err=float(np.max(abs(result-congruence(self.Q,true_H))))
        self.assertLess(err,1e-11)
        self.assertLess(np.max(abs(H-true_H)),1e-13)
        self.assertEqual(estimate['stats']['fallback_count'],0)
        self.assertLess(np.max(abs(estimate['arrays']['used_sigma_min']-np.linalg.svd(A,compute_uv=False)[-1])),1e-13)
        DETAILS.append(dict(test='full_rank_affine',quadric_max_abs_error=err))

    def test_multistep_composition_and_inverse_undo(self):
        steps=[affine_matrix(np.array([[1.1,.04,0.],[0.,.9,.03],[0.,0.,1.03]]),np.array([.07,-.05,.02])),
               affine_matrix(np.array([[.95,0.,.02],[.03,1.04,0.],[0.,-.02,1.1]]),np.array([-.03,.04,-.02])),
               affine_matrix(np.array([[1.02,-.05,0.],[.05,1.02,0.],[0.,0.,.97]]),np.array([.02,.01,.03]))]
        x=self.x.copy();Q=self.Q.copy();combined=np.eye(4)
        for step in steps:
            target=apply(step,x)
            Q,H,estimate,_=step_transport(Q,x,target,self.nb)
            self.assertEqual(estimate['stats']['fallback_count'],0)
            combined=step@combined;x=target
        composed_error=float(np.max(abs(Q-congruence(self.Q,combined))))
        self.assertLess(composed_error,1e-11)
        for step in reversed(steps):
            target=apply(np.linalg.inv(step),x)
            Q,_,estimate,_=step_transport(Q,x,target,self.nb)
            self.assertEqual(estimate['stats']['fallback_count'],0)
            x=target
        undo_error=float(np.max(abs(Q-self.Q)))
        self.assertLess(undo_error,1e-11)
        self.assertLess(np.max(abs(x-self.x)),1e-13)
        DETAILS.append(dict(test='compose_and_undo',composed_Q_max_abs_error=composed_error,
                            undo_Q_max_abs_error=undo_error))

    def test_frozen_neighbors_weights_scale_and_dynamic_regression_cache(self):
        current=self.x@np.diag([2.,.5,1.1])
        refreshed=refresh_neighborhood(current,self.nb)
        old=self.nb['arrays'];new=refreshed['arrays']
        for key in ['neighbor_indices','weights','h','neighbor_distances']:
            self.assertTrue(np.array_equal(new[key],old[key]),key)
        expected=(current[old['neighbor_indices']]-current[:,None,:])/old['h'][:,None,None]
        self.assertTrue(np.array_equal(new['X'],expected))
        self.assertTrue(np.array_equal(new['x0'],current))
        self.assertFalse(np.allclose(new['weighted_singular_values'],old['weighted_singular_values']))
        self.assertFalse(np.allclose(new['G'],old['G']))
        self.assertFalse(np.array_equal(prepare_neighborhood(current)['arrays']['neighbor_indices'],old['neighbor_indices']))
        self.assertTrue(np.array_equal(old['x0'],self.x))

    def test_rank_deficient_fallback_translates_instead_of_freezing_world_Q(self):
        x=self.x.copy();x[:,2]=0.
        nb=prepare_neighborhood(x)
        target=x@np.array([[1.2,.1,0.],[0.,.8,0.],[.2,.1,1.]]).T+np.array([.1,.2,-.1])
        Q,H,estimate,_=step_transport(self.Q,x,target,nb)
        self.assertTrue(np.all(estimate['arrays']['used_fallback']))
        self.assertTrue(np.all(estimate['arrays']['reason_code']==1))
        self.assertTrue(np.isnan(estimate['arrays']['raw_sigma_min']).all())
        self.assertTrue(np.array_equal(estimate['arrays']['used_sigma_min'],np.ones(len(x))))
        self.assertTrue(np.array_equal(H[:,:3,:3],np.broadcast_to(np.eye(3),(len(x),3,3))))
        self.assertTrue(np.array_equal(H[:,:3,3],target-x))
        expected=np.array([congruence(q,h) for q,h in zip(self.Q,H)])
        self.assertTrue(np.array_equal(Q,expected))
        self.assertFalse(np.array_equal(Q,self.Q))
        json.dumps(estimate['stats'],allow_nan=False)


if __name__=='__main__':
    result=unittest.main(verbosity=2,exit=False).result
    destination=os.environ.get('PONQ_INCREMENTAL_VALIDATION_DIR')
    if destination:
        root=Path(destination);root.mkdir(parents=True,exist_ok=False)
        files=[Path(__file__),Path(__file__).with_name('incremental_transport.py'),
               Path(__file__).with_name('motion_jacobian.py')]
        record=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
                    failures=len(result.failures),errors=len(result.errors),details=DETAILS,
                    scope='Synthetic mathematical tests only; no real extraction or hyperparameter selection.',
                    source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
                    failure_details=[dict(test=str(t),traceback=tb) for t,tb in result.failures+result.errors])
        (root/'VALIDATION.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
