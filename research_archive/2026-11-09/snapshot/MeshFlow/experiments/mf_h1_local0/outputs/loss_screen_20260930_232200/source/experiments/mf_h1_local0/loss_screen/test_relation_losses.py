"""Bounded CPU fixtures for registered relation losses; no model or OT calls."""
import itertools
import json
import time
import unittest

import numpy as np
import torch

from experiments.mf_h1_local0.loss_screen.relation_losses import (
    FORMULA_SPEC, prepare_relations, relation_loss, point_triangle_distance_sq)

LEDGER = dict(prepare_calls=0, relation_forward_calls=0,
              independent_distance_forward_calls=0, autograd_calls=0,
              finite_difference_evaluations=0, model_forwards=0, OT_calls=0,
              optimizer_updates=0)


def prep(*args):
    LEDGER['prepare_calls'] += 1
    return prepare_relations(*args)


def loss(name, value, target):
    LEDGER['relation_forward_calls'] += 1
    return relation_loss(name, value, target)


def grad(value, inputs, **kwargs):
    LEDGER['autograd_calls'] += 1
    return torch.autograd.grad(value, inputs, **kwargs)


def distance(points, triangles):
    LEDGER['independent_distance_forward_calls'] += 1
    return point_triangle_distance_sq(points, triangles)


def fixture(dtype=torch.float64, two=True):
    vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                           [.25,0.,1.],[1.,1.,.5]],dtype=dtype)
    fi=torch.tensor([[1,0,3],[1,3,4]] if two else [[1,0,3]])
    ki=torch.tensor([[0,1,2]])
    return vertices[fi],vertices[ki],fi,ki


class RelationLossTests(unittest.TestCase):
    def test_oracle_zero_and_contact_gradients(self):
        gt,C,fi,ki=fixture();target=prep(gt,C,fi,ki,1.)
        # Endpoint oracle: Xt+(1-t)*(Y-Z)=Y, tested in FP64.
        z=torch.arange(gt.numel(),dtype=gt.dtype).reshape_as(gt)/37
        oracle=(.25*z+.75*gt)+.25*(gt-z)
        self.assertTrue(torch.allclose(oracle,gt,atol=1e-15,rtol=0))
        for name in ('L1','L2','L3','L6'):
            pred=oracle.clone().requires_grad_();out=loss(name,pred,target)
            self.assertTrue(out['applicable']);self.assertLess(abs(float(out['loss'])),1e-25)
            gradient=grad(out['loss'],pred)[0]
            self.assertTrue(torch.isfinite(gradient).all());self.assertLess(float(gradient.abs().max()),1e-12)
        # Exactly on the known plane has no positive artificial contact margin.
        pred=gt.clone();pred[...,2]=0;pred.requires_grad_()
        value=loss('L1',pred,target)['loss'];gradient=grad(value,pred)[0]
        self.assertEqual(float(value),0.);self.assertEqual(int(torch.count_nonzero(gradient)),0)

    def test_all_four_finite_differences(self):
        gt,C,fi,ki=fixture(two=False);target=prep(gt,C,fi,ki,1.)
        perturb=torch.tensor([[[.031,.052,-.44],[-.023,.027,-.39],[.044,.31,-.21]]],dtype=gt.dtype)
        for name in ('L1','L2','L3','L6'):
            x=(gt+perturb).requires_grad_();value=loss(name,x,target)['loss']
            analytic=grad(value,x)[0].detach().numpy();numeric=np.empty_like(analytic)
            h=1e-6
            for index in np.ndindex(numeric.shape):
                plus=x.detach().clone();minus=x.detach().clone();plus[index]+=h;minus[index]-=h
                numeric[index]=(float(loss(name,plus,target)['loss'])-float(loss(name,minus,target)['loss']))/(2*h)
                LEDGER['finite_difference_evaluations']+=2
            np.testing.assert_allclose(analytic,numeric,atol=3e-7,rtol=2e-6,err_msg=name)
            self.assertGreater(float(np.linalg.norm(analytic)),1e-8)

    def test_source_labels_follow_faces_and_all_S3_corners(self):
        gt,C,fi,ki=fixture();pred=gt+torch.tensor([[[.01,.04,-.22],[.05,.08,-.17],[.03,.12,.15]],
                                                [[.07,-.03,.06],[-.04,.07,-.08],[.02,.09,-.05]]],dtype=gt.dtype)
        target=prep(gt,C,fi,ki,1.7)
        expected={name:float(loss(name,pred,target)['loss']) for name in ('L1','L2','L3','L6')}
        for permutation in itertools.permutations(range(3)):
            p=list(permutation);face=torch.tensor([1,0]);cp=list(reversed(p))
            revised=prep(gt[face][:,p],C[:,cp],fi[face][:,p],ki[:,cp],1.7)
            for name,reference in expected.items():
                actual=float(loss(name,pred[face][:,p],revised)['loss'])
                self.assertAlmostEqual(actual,reference,delta=2e-12,msg=name)

    def test_L1_coplanar_is_inapplicable_no_forced_side(self):
        gt,C,fi,ki=fixture(two=False);gt=gt.clone();gt[0,2]=torch.tensor([.25,-1.,0.])
        target=prep(gt,C,fi,ki,1.)
        self.assertEqual(target['counts']['L1_inapplicable_coplanar_or_ambiguous'],1)
        pred=(gt+1).requires_grad_();out=loss('L1',pred,target)
        self.assertFalse(out['applicable']);self.assertEqual(float(out['loss']),0.)
        self.assertEqual(int(torch.count_nonzero(grad(out['loss'],pred)[0])),0)
        # Threshold is <=1e-6 L, with no post hoc tolerance or sign assignment.
        for height,applicable in [(1e-6,False),(1.01e-6,True)]:
            clean=gt.clone();clean[0,2,2]=height
            t=prep(clean,C,fi,ki,1.)
            self.assertEqual(bool(t['counts']['L1_applicable_relations']),applicable)

    def test_L2_right_angle_and_translation_blind_spot(self):
        gt,C,fi,ki=fixture(two=False);target=prep(gt,C,fi,ki,1.)
        self.assertAlmostEqual(float(target['pair_gt_cos2'][0]),0.,delta=1e-15)
        flattened=gt.clone();flattened[0,2]=torch.tensor([.25,-1.,0.])
        self.assertAlmostEqual(float(loss('L2',flattened,target)['loss']),1.,delta=1e-14)
        self.assertLess(float(loss('L2',gt+torch.tensor([5.,2.,-3.]),target)['loss']),1e-25)

    def test_L3_tangential_to_transverse_ratio(self):
        gt,C,fi,ki=fixture(two=False);target=prep(gt,C,fi,ki,1.)
        parallel=(gt+torch.tensor([.2,0.,0.])).requires_grad_()
        transverse=(gt+torch.tensor([0.,.2,0.])).requires_grad_()
        a=loss('L3',parallel,target)['loss'];b=loss('L3',transverse,target)['loss']
        self.assertAlmostEqual(float(a/b),.05,delta=1e-12)
        ga=grad(a,parallel)[0];gb=grad(b,transverse)[0]
        self.assertEqual(int(torch.count_nonzero(ga[...,1:])),0)
        self.assertEqual(int(torch.count_nonzero(gb[...,[0,2]])),0)
        self.assertAlmostEqual(float(ga.norm()/gb.norm()),.05,delta=1e-12)

    def test_L6_finite_surface_missing_midpoint_and_band(self):
        gt,C,fi,ki=fixture(two=False)
        extra=torch.tensor([[[3.,0.,0.],[4.,0.,0.],[3.,1.,0.]]],dtype=gt.dtype)
        clean=torch.cat((gt,extra));ids=torch.cat((fi,torch.tensor([[5,6,7]])))
        target=prep(clean,C,ids,ki,1.)
        # Both GT interface endpoints exist in the prediction; its middle does not.
        pred=torch.tensor([[[0.,0.,0.],[.05,0.,0.],[0.,.05,0.]],
                           [[1.,0.,0.],[.95,0.,0.],[1.,.05,0.]]],dtype=gt.dtype)
        q=torch.tensor([[0.,0.,0.],[.5,0.,0.],[1.,0.,0.]],dtype=gt.dtype)
        d=distance(q,pred).min(-1).values
        self.assertEqual(float(d[0]),0.);self.assertEqual(float(d[2]),0.)
        self.assertGreater(float(d[1]),.19)
        self.assertGreater(float(loss('L6',pred,target)['loss']),.01)
        one=prep(gt,C,fi,ki,1.)
        flat=gt.clone();flat[0,2]=torch.tensor([.25,-1.,0.])
        self.assertLess(float(distance(one['gamma'].reshape(-1,3),flat).min(-1).values.max()),1e-25)
        self.assertGreater(float(distance(one['band'].reshape(-1,3),flat).min(-1).values.mean()),.01)
        self.assertGreater(float(loss('L6',flat,one)['loss']),.005)
        # Point is on the infinite supporting plane but outside the finite triangle.
        query=torch.tensor([[4.,4.,0.]],dtype=gt.dtype)
        self.assertGreater(float(distance(query,C)[0,0]),20.)

    def test_nonmanifold_edges_keep_all_pairs_and_no_proximity_labels(self):
        gt,C,fi,ki=fixture(two=False)
        second=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[.5,0.,-1.]]],dtype=gt.dtype)
        clean=torch.cat((gt,second));ids=torch.cat((fi,torch.tensor([[0,1,4]])))
        target=prep(clean,C,ids,ki,1.);counts=target['counts']
        self.assertEqual(counts['nonmanifold_source_edges'],1)
        self.assertEqual(counts['adjacent_pairs_at_least_one_free'],3)
        self.assertEqual(counts['known_free_interface_relations'],2)
        self.assertEqual(counts['free_free_pairs'],1)
        for name in ('L1','L2','L3','L6'):
            self.assertLess(float(loss(name,clean,target)['loss']),1e-25)
        # Same geometric positions with separate true IDs do not invent adjacency.
        separate=prep(gt,C,fi,ki+100,1.)
        for name in ('L1','L2','L3','L6'):
            self.assertFalse(loss(name,gt,separate)['applicable'])
        inconsistent=C.clone();inconsistent[0,0,0]=.1
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            prep(gt,inconsistent,fi,ki,1.)

    def test_degenerate_prediction_is_finite_not_deleted(self):
        gt,C,fi,ki=fixture(two=False);target=prep(gt,C,fi,ki,1.)
        for name,key in [('L2','prediction_normal_epsilon_faces'),('L6','prediction_surface_edge_fallback_faces')]:
            pred=torch.zeros_like(gt,requires_grad=True);out=loss(name,pred,target)
            self.assertTrue(out['applicable']);self.assertTrue(torch.isfinite(out['loss']))
            self.assertEqual(int(out['counts'][key]),1)
            self.assertTrue(torch.isfinite(grad(out['loss'],pred)[0]).all())

    def test_fixed_GT_C_detach_precision_and_scale(self):
        gt,C,fi,ki=fixture();gt.requires_grad_();C.requires_grad_()
        target=prep(gt,C,fi,ki,2.3)
        self.assertTrue(all(not v.requires_grad for v in target.values() if isinstance(v,torch.Tensor)))
        for name in ('L1','L2','L3','L6'):
            pred=(gt.detach()+torch.tensor([.12,.07,-.3],dtype=gt.dtype)).requires_grad_()
            value=loss(name,pred,target)['loss'];g=grad(value,(pred,gt,C),allow_unused=True)
            self.assertIsNone(g[1]);self.assertIsNone(g[2]);self.assertTrue(torch.isfinite(g[0]).all())
            scaled=prep(gt.detach()*7,C.detach()*7,fi,ki,2.3*7)
            self.assertAlmostEqual(float(loss(name,pred.detach()*7,scaled)['loss']),float(value),delta=2e-12)
            fp32=prep(gt.detach().float(),C.detach().float(),fi,ki,2.3)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                out=loss(name,pred.detach().float(),fp32)
            self.assertEqual(out['loss'].dtype,torch.float32)
            self.assertAlmostEqual(float(out['loss']),float(value),delta=2e-6)


def run_cpu_tests():
    for key in LEDGER:LEDGER[key]=0
    start=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RelationLossTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),seconds=time.perf_counter()-start,
                counts=dict(LEDGER),formula_spec=FORMULA_SPEC,
                numerical_thresholds=dict(FP64_oracle_loss=1e-25,FP64_oracle_gradient=1e-12,
                    finite_difference_step=1e-6,finite_difference_atol=3e-7,finite_difference_rtol=2e-6,
                    FP64_permutation_atol=2e-12,FP32_FP64_atol=2e-6),
                CUDA_initialized=torch.cuda.is_initialized())
    return report


if __name__=='__main__':
    torch.set_num_threads(1)
    result=run_cpu_tests()
    print(json.dumps(result,sort_keys=True))
    raise SystemExit(0 if result['status']=='PASS' else 1)
