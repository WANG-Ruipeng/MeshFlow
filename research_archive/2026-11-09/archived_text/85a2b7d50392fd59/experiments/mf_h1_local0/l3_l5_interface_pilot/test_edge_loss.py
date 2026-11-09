"""Bounded CPU witnesses for D_edge. Never loads a generator or advances OT."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from .edge_loss import prepare_edge_target,edge_loss,centered_segment_distance_sq,FORMULA_SPEC

COUNTS=dict(prepare_attempts=0,prepare_returns=0,edge_loss_attempts=0,edge_loss_returns=0,
            segment_distance_calls=0,CPU_coordinate_autograd_calls=0,finite_difference_evaluations=0,
            model_calls=0,OT_calls=0,optimizer_updates=0)
MEASUREMENTS={}


def prepare(*args):
    COUNTS['prepare_attempts']+=1
    result=prepare_edge_target(*args);COUNTS['prepare_returns']+=1
    return result


def evaluate(x,target):
    COUNTS['edge_loss_attempts']+=1
    result=edge_loss(x,target);COUNTS['edge_loss_returns']+=1
    if result['applicable']:COUNTS['segment_distance_calls']+=2
    return result


def gradient(loss,inputs,**kwargs):
    COUNTS['CPU_coordinate_autograd_calls']+=1
    return torch.autograd.grad(loss,inputs,**kwargs)


def distance(points,segments):
    COUNTS['segment_distance_calls']+=1
    return centered_segment_distance_sq(points,segments)


def fixture(two=False,dtype=torch.float64):
    vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                           [.3,-1.,.2],[1.,1.,1.]],dtype=dtype)
    free_ids=torch.tensor([[1,0,3],[2,1,4]] if two else [[1,0,3]])
    known_ids=torch.tensor([[0,1,2]])
    return dict(free_gt=vertices[free_ids],known=vertices[known_ids],
                free_ids=free_ids,known_ids=known_ids,bbox_L=1.)


class EdgeTests(unittest.TestCase):
    def test_oracle_interface_zero(self):
        label=fixture(two=True);target=prepare(label)
        pred=label['free_gt'].clone().requires_grad_();result=evaluate(pred,target)
        self.assertTrue(result['applicable']);self.assertEqual(target['counts']['eligible_interface_edges'],2)
        self.assertLess(float(result['loss']),1e-25)
        g=gradient(result['loss'],pred)[0]
        self.assertTrue(torch.isfinite(g).all());self.assertLess(float(g.abs().max()),1e-12)
        MEASUREMENTS['oracle_loss']=float(result['loss'])

    def test_collapsed_and_very_short_prediction_retained(self):
        label=fixture();target=prepare(label)
        pred=label['free_gt'].clone();pred[0,:2]=torch.tensor([.5,.2,0.],dtype=pred.dtype);pred.requires_grad_()
        result=evaluate(pred,target);g=gradient(result['loss'],pred)[0]
        self.assertTrue(torch.isfinite(g).all());self.assertEqual(int(result['counts']['zero_predicted_edges']),1)
        self.assertEqual(int(result['counts']['short_predicted_edges']),1)
        self.assertAlmostEqual(float(result['diagnostics']['GT_to_pred_mean']),.09375+.04,delta=1e-14)
        self.assertAlmostEqual(float(result['diagnostics']['pred_to_GT_mean']),.04,delta=1e-14)
        self.assertTrue(torch.equal(g[0,0],g[0,1]));self.assertGreater(float(g.norm()),0.)
        centered=pred.detach().clone();centered[0,:2]=torch.tensor([.5,0.,0.],dtype=pred.dtype);centered.requires_grad_()
        value=evaluate(centered,target)['loss'];gc=gradient(value,centered)[0]
        self.assertAlmostEqual(float(value),.046875,delta=1e-14)
        self.assertEqual(int(torch.count_nonzero(gc)),0)
        tiny=pred.detach().clone();tiny[0,0,0]-=1e-8;tiny[0,1,0]+=1e-8;tiny.requires_grad_()
        short=evaluate(tiny,target);gs=gradient(short['loss'],tiny)[0]
        self.assertEqual(int(short['counts']['short_predicted_edges']),1)
        self.assertTrue(torch.isfinite(gs).all());self.assertGreater(float(short['loss']),.04)
        MEASUREMENTS['centered_collapsed_loss']=float(value)
        MEASUREMENTS['centered_collapsed_gradient_L2']=float(gc.norm())
        MEASUREMENTS['off_axis_collapsed_gradient_L2']=float(g.norm())

    def test_collinear_overshoot_is_seen_by_reverse_direction(self):
        label=fixture();target=prepare(label);pred=label['free_gt'].clone()
        # Source corner order is ID1,ID0: maintain correspondence while extending.
        pred[0,0]=torch.tensor([1.5,0.,0.]);pred[0,1]=torch.tensor([-.5,0.,0.])
        result=evaluate(pred,target)
        self.assertLess(float(result['diagnostics']['GT_to_pred_mean']),1e-25)
        self.assertAlmostEqual(float(result['diagnostics']['pred_to_GT_mean']),.0625,delta=1e-14)
        self.assertAlmostEqual(float(result['loss']),.03125,delta=1e-14)
        MEASUREMENTS['overshoot_loss']=float(result['loss'])

    def test_translation_and_fixed_GT_length_weights(self):
        label=fixture(two=True);target=prepare(label);pred=label['free_gt'].clone()
        pred[0]+=torch.tensor([0.,0.,.1],dtype=pred.dtype)
        pred[1]+=torch.tensor([0.,0.,.3],dtype=pred.dtype)
        result=evaluate(pred,target)
        expected=(.01+.09*np.sqrt(2))/(1+np.sqrt(2))
        self.assertAlmostEqual(float(result['loss']),expected,delta=2e-14)
        label=fixture();target=prepare(label)
        value=evaluate(label['free_gt']+torch.tensor([0.,0.,.2],dtype=torch.float64),target)['loss']
        self.assertAlmostEqual(float(value),.04,delta=1e-14)
        MEASUREMENTS['GT_length_weighted_loss']=float(result['loss'])

    def test_endpoint_and_face_S3_permutations(self):
        label=fixture(two=True);target=prepare(label)
        pred=label['free_gt']+torch.tensor([[[.12,.13,-.11],[-.07,.04,.16],[0.,0.,0.]],
                                            [[-.05,.02,.07],[.12,-.1,.13],[0.,0.,0.]]],dtype=torch.float64)
        expected=float(evaluate(pred,target)['loss'])
        for permutation in itertools.permutations(range(3)):
            corner=list(permutation);face=torch.tensor([1,0]);known_corner=list(reversed(corner))
            moved=dict(free_gt=label['free_gt'][face][:,corner],known=label['known'][:,known_corner],
                       free_ids=label['free_ids'][face][:,corner],known_ids=label['known_ids'][:,known_corner],bbox_L=1.)
            got=evaluate(pred[face][:,corner],prepare(moved))['loss']
            self.assertAlmostEqual(float(got),expected,delta=2e-12)
        reversed_target=copy.copy(target)
        reversed_target['GT_endpoints']=target['GT_endpoints'].flip(1)
        reversed_target['free_corners']=target['free_corners'].flip(1)
        self.assertAlmostEqual(float(evaluate(pred,reversed_target)['loss']),expected,delta=2e-12)
        segment=torch.tensor([[[.1,.2,.3],[1.1,.5,.7]]],dtype=torch.float64,requires_grad=True)
        query=torch.tensor([[[.2,-.2,.8],[.6,.8,.1],[1.8,.4,.9]]],dtype=torch.float64)
        a=distance(query,segment).sum();ga=gradient(a,segment)[0]
        swap=segment.detach().flip(1).requires_grad_();b=distance(query,swap).sum();gb=gradient(b,swap)[0]
        self.assertAlmostEqual(float(a),float(b),delta=1e-14)
        self.assertTrue(torch.allclose(ga,gb.flip(1),atol=1e-14,rtol=1e-14))

    def test_fixed_C_GT_detached_and_free_corner_gradient(self):
        label=fixture();label['free_gt'].requires_grad_();label['known'].requires_grad_()
        target=prepare(label)
        self.assertTrue(all(not v.requires_grad for v in target.values() if isinstance(v,torch.Tensor)))
        pred=(label['free_gt'].detach()+torch.tensor([0.,0.,.2],dtype=torch.float64)).requires_grad_()
        value=evaluate(pred,target)['loss'];g,gt,c=gradient(value,(pred,label['free_gt'],label['known']),allow_unused=True)
        self.assertIsNone(gt);self.assertIsNone(c);self.assertGreater(float(g[:,:2].norm()),0.)
        self.assertEqual(int(torch.count_nonzero(g[:,2])),0)

    def test_FP64_finite_difference_and_FP32_scale(self):
        label=fixture();target=prepare(label)
        pred=(label['free_gt']+torch.tensor([[[.11,.17,-.12],[-.07,.04,.16],[.02,.09,.05]]],dtype=torch.float64)).requires_grad_()
        value=evaluate(pred,target)['loss'];analytic=gradient(value,pred)[0].numpy();numeric=np.empty_like(analytic);h=1e-6
        for idx in np.ndindex(numeric.shape):
            plus=pred.detach().clone();minus=pred.detach().clone();plus[idx]+=h;minus[idx]-=h
            numeric[idx]=(float(evaluate(plus,target)['loss'])-float(evaluate(minus,target)['loss']))/(2*h)
            COUNTS['finite_difference_evaluations']+=2
        np.testing.assert_allclose(analytic,numeric,atol=3e-7,rtol=2e-6)
        MEASUREMENTS['finite_difference_max_abs_error']=float(np.max(np.abs(analytic-numeric)))
        fp32={k:(v.float() if isinstance(v,torch.Tensor) and v.is_floating_point() else v) for k,v in label.items()}
        with torch.autocast('cpu',dtype=torch.bfloat16):out=evaluate(pred.detach().float(),prepare(fp32))
        self.assertEqual(out['loss'].dtype,torch.float32);self.assertAlmostEqual(float(out['loss']),float(value),delta=2e-6)
        scaled=dict(label,free_gt=label['free_gt']*7,known=label['known']*7,bbox_L=7.)
        self.assertAlmostEqual(float(evaluate(pred.detach()*7,prepare(scaled))['loss']),float(value),delta=2e-12)

    def test_incidence_exclusion_and_GT_short_edge_rejection(self):
        label=fixture();second=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[.2,1.,1.]]],dtype=torch.float64)
        nonmanifold=dict(label,free_gt=torch.cat((label['free_gt'],second)),free_ids=torch.cat((label['free_ids'],torch.tensor([[0,1,9]]))))
        target=prepare(nonmanifold);counts=target['counts']
        self.assertEqual(counts['eligible_interface_edges'],0)
        self.assertEqual(counts['excluded_nonmanifold_mixed_edges'],1)
        self.assertEqual(counts['excluded_nonmanifold_known_free_pairs'],2)
        pred=nonmanifold['free_gt'].clone().requires_grad_();out=evaluate(pred,target)
        self.assertFalse(out['applicable']);self.assertEqual(float(out['loss']),0.)
        self.assertEqual(int(torch.count_nonzero(gradient(out['loss'],pred)[0])),0)
        tiny=fixture();tiny['free_gt'][0,0,0]=1e-7;tiny['known'][0,1,0]=1e-7
        with self.assertRaisesRegex(ValueError,'Abnormal GT interface'):
            prepare(tiny)
        # Exactly two free or two known faces do not become eligible interfaces.
        ff=dict(label,free_gt=torch.cat((label['free_gt'],second)),free_ids=torch.cat((label['free_ids'],torch.tensor([[0,1,9]]))),
                known=torch.empty((0,3,3),dtype=torch.float64),known_ids=torch.empty((0,3),dtype=torch.int64))
        self.assertEqual(prepare(ff)['counts']['excluded_two_free_edges'],1)


def run_cpu_tests():
    for key in COUNTS:COUNTS[key]=0
    MEASUREMENTS.clear();start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EdgeTests))
    folder=Path(__file__).resolve().parent
    return dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),seconds=time.perf_counter()-start,
        counts=dict(COUNTS),measurements=dict(MEASUREMENTS),formula=FORMULA_SPEC,
        thresholds=dict(FP64_oracle_loss=1e-25,FP64_oracle_gradient=1e-12,finite_difference_step=1e-6,
          finite_difference_atol=3e-7,finite_difference_rtol=2e-6,FP64_permutation_atol=2e-12,FP32_FP64_atol=2e-6),
        source_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('edge_loss.py','test_edge_loss.py')},
        CUDA_initialized=torch.cuda.is_initialized(),old_L3_L5_implementation_modified=False)


if __name__=='__main__':
    torch.set_num_threads(1)
    report=run_cpu_tests();print(json.dumps(report,sort_keys=True,allow_nan=False))
    raise SystemExit(0 if report['status']=='PASS' else 1)
