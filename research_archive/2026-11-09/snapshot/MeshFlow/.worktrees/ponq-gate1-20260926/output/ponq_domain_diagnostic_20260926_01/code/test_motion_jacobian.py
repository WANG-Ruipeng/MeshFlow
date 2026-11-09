"""Mathematical synthetic tests; no object/reference data or analytic-map import."""
import hashlib
import json
import os
from pathlib import Path
import unittest
import numpy as np
from motion_jacobian import (prepare_neighborhood, estimate_motion, K, RANK_RTOL,
                             RIDGE_TRACE_FACTOR, DET_FLOOR, COND_LIMIT)

DETAILS=[]


class MotionJacobianTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(4821).normal(size=(80,3))
        self.neighborhood=prepare_neighborhood(self.x)

    def test_known_3d_affine_lsq_and_ridge_normal_equations(self):
        J=np.array([[1.2,.13,-.09],[.04,.8,.12],[-.03,.06,1.1]])
        moved=self.x@J.T+np.array([.3,-.2,.1])
        lsq=estimate_motion(self.x,moved,self.neighborhood,'lsq16')
        err=float(np.max(abs(lsq['J_used']-J)))
        self.assertLess(err,1e-13)
        self.assertEqual(lsq['stats']['fallback_count'],0)
        ridge=estimate_motion(self.x,moved,self.neighborhood,'ridge16')
        a=self.neighborhood['arrays']
        expected=np.array([np.eye(3)+np.linalg.solve(G+lam*np.eye(3),G@(J-np.eye(3)).T).T
                           for G,lam in zip(a['G'],a['ridge_lambda'])])
        self.assertLess(np.max(abs(ridge['J_used']-expected)),1e-13)
        DETAILS.append(dict(test='affine',lsq_max_error=err,
                            ridge_closed_form_max_error=float(np.max(abs(ridge['J_used']-expected)))))

    def test_t0_and_exactly_representable_translation_are_exact_identity(self):
        x=np.random.default_rng(34).integers(-32,33,size=(80,3))/32.
        neighborhood=prepare_neighborhood(x)
        identity=np.broadcast_to(np.eye(3),(len(x),3,3))
        for mode in ['lsq16','ridge16']:
            for moved in [x.copy(),x+np.array([.25,-.5,1.])]:
                result=estimate_motion(x,moved,neighborhood,mode)
                self.assertTrue(np.array_equal(result['J_used'],identity))
                self.assertEqual(result['stats']['fallback_count'],0)
                self.assertEqual(result['arrays']['used_weighted_residual_normalized'].max(),0.)

    def test_planar_lsq_rejects_and_ridge_keeps_unobserved_axis_identity(self):
        x=self.x.copy();x[:,2]=0
        J=np.array([[1.2,.1,0.],[.05,.8,0.],[.2,.3,1.]])
        neighborhood=prepare_neighborhood(x)
        moved=x@J.T
        lsq=estimate_motion(x,moved,neighborhood,'lsq16')
        ridge=estimate_motion(x,moved,neighborhood,'ridge16')
        self.assertTrue(np.all(lsq['arrays']['rank']==2))
        self.assertTrue(np.all(lsq['arrays']['reason_code']==1))
        self.assertTrue(np.isnan(lsq['J_raw']).all())
        self.assertTrue(np.array_equal(lsq['J_used'],np.broadcast_to(np.eye(3),(len(x),3,3))))
        self.assertEqual(ridge['stats']['fallback_count'],0)
        self.assertTrue(np.array_equal(ridge['J_used'][:,:,2],np.tile([0.,0.,1.],(len(x),1))))
        json.dumps(lsq['stats'],allow_nan=False)
        json.dumps(ridge['stats'],allow_nan=False)

    def test_known_rotation(self):
        angle=.37
        J=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        result=estimate_motion(self.x,self.x@J.T,self.neighborhood,'lsq16')
        self.assertLess(np.max(abs(result['J_used']-J)),1e-13)
        self.assertEqual(result['stats']['fallback_count'],0)

    def test_nonpositive_determinant_and_extreme_condition_fallback_without_clipping(self):
        reflection=np.diag([-2.,1.,1.])
        for mode in ['lsq16','ridge16']:
            result=estimate_motion(self.x,self.x@reflection.T,self.neighborhood,mode)
            self.assertTrue(np.all(result['arrays']['raw_determinant']<0))
            self.assertTrue(np.all(result['arrays']['reason_code']==3))
            self.assertTrue(np.array_equal(result['J_used'],np.broadcast_to(np.eye(3),(len(self.x),3,3))))
        stretch=np.diag([2e4,1.,1.])
        result=estimate_motion(self.x,self.x@stretch.T,self.neighborhood,'lsq16')
        self.assertTrue(np.all(result['arrays']['reason_code']==4))
        self.assertGreater(result['arrays']['raw_condition'].min(),1e4)

    def test_permutation_equivariance_without_distance_ties(self):
        self.assertEqual(self.neighborhood['stats']['exact_adjacent_distance_ties'],0)
        moved=self.x+np.column_stack([.03*self.x[:,0]**2,.02*self.x[:,1]*self.x[:,2],-.01*self.x[:,2]**2])
        order=np.random.default_rng(28).permutation(len(self.x))
        reordered=prepare_neighborhood(self.x[order])
        for mode in ['lsq16','ridge16']:
            a=estimate_motion(self.x,moved,self.neighborhood,mode)
            b=estimate_motion(self.x[order],moved[order],reordered,mode)
            self.assertLess(np.max(abs(a['J_used'][order]-b['J_used'])),1e-13)
            self.assertTrue(np.array_equal(a['arrays']['reason_code'][order],b['arrays']['reason_code']))
            self.assertTrue(np.array_equal(order[reordered['arrays']['neighbor_indices']],self.neighborhood['arrays']['neighbor_indices'][order]))

    def test_frozen_binding_and_nonfinite_motion_accounting(self):
        with self.assertRaises(ValueError):
            estimate_motion(self.x[::-1],self.x[::-1],self.neighborhood)
        with self.assertRaises(ValueError):
            prepare_neighborhood(self.x,k=8)
        moved=self.x.copy();moved[0,0]=np.nan
        result=estimate_motion(self.x,moved,self.neighborhood)
        self.assertGreater(result['stats']['nonfinite_motion_elements'],0)
        self.assertTrue(np.isfinite(result['J_used']).all())
        json.dumps(result['stats'],allow_nan=False)


if __name__=='__main__':
    result=unittest.main(verbosity=2,exit=False).result
    destination=os.environ.get('PONQ_MOTION_VALIDATION_DIR')
    if destination:
        root=Path(destination);root.mkdir(parents=True,exist_ok=False)
        paths=[Path(__file__),Path(__file__).with_name('motion_jacobian.py')]
        report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
                    failures=len(result.failures),errors=len(result.errors),details=DETAILS,
                    scope='Synthetic mathematical tests only; no real objects or hyperparameter selection.',
                    constants=dict(k=K,rank_rtol=RANK_RTOL,ridge_trace_factor=RIDGE_TRACE_FACTOR,
                                   determinant_floor=DET_FLOOR,condition_limit=COND_LIMIT),
                    source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                    failure_details=[dict(test=str(test),traceback=tb) for test,tb in result.failures+result.errors])
        (root/'VALIDATION.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
