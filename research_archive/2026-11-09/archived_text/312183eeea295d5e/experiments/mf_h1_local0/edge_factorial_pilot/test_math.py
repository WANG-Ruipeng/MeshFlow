"""Bounded independent CPU fixtures for the added edge metric and units.

The fixed 19-case production/dense/analytic comparison runs separately once.
"""
import hashlib
import json
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from .edge_metric import gamma_to_free_edges,check_surface_lower_bound,COVERAGE_FRACTIONS,METRIC_SPEC
from .quadrature import new_counts,fixed_target,loss_at,analytic,POLICY,CASES

COUNTS=new_counts()
MEASUREMENTS={}


def metric(*args,**kwargs):
    return gamma_to_free_edges(*args,ledger=COUNTS,**kwargs)


class MathTests(unittest.TestCase):
    def test_finite_triangle_surface_lower_bound_and_interior_counterexample(self):
        triangles=np.asarray([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        points=np.asarray([[.25,.25,0.],[.25,.25,.1],[2.,0.,.3],[.3,0.,0.]])
        weights=np.asarray([1.,2.,3.,4.])
        record=metric(triangles,points,weights,1.)
        # Independently known closest points on this closed right triangle.
        surface=np.asarray([0.,.1,np.sqrt(1.09),0.])
        expected_edge=np.asarray([.25,np.sqrt(.25**2+.1**2),np.sqrt(1.09),0.])
        np.testing.assert_allclose(record['per_point_distances'],expected_edge,atol=1e-14,rtol=1e-14)
        bound=check_surface_lower_bound(record['per_point_distances'],surface,1.)
        self.assertEqual(bound['status'],'PASS')
        self.assertGreater(record['per_point_distances'][0],surface[0])
        self.assertEqual(record['edge_count'],3);self.assertEqual(record['zero_length_edge_count'],0)
        MEASUREMENTS['surface_interior_distance']=float(surface[0])
        MEASUREMENTS['edge_distance_same_interior_query']=record['per_point_distances'][0]

    def test_weighted_p95_strict_thresholds_zero_edges_and_duplicates(self):
        triangles=np.zeros((2,3,3));points=np.asarray([[0.,0.,0.],[.001,0.,0.],[.01,0.,0.]])
        result=metric(triangles,points,np.asarray([94.,4.,2.]),1.)
        self.assertEqual(result['edge_count'],6);self.assertEqual(result['zero_length_edge_count'],6)
        self.assertEqual(result['p95'],.001)
        self.assertEqual(result['coverage_by_bbox_fraction'],{'0.0005':.94,'0.001':.94,'0.002':.98,'0.005':.98,'0.01':.98})
        self.assertAlmostEqual(result['RMS'],np.sqrt((4*.001**2+2*.01**2)/100),delta=1e-15)
        self.assertEqual(check_surface_lower_bound(result['per_point_distances'],[0.,.001,.01],1.)['status'],'PASS')

    def test_edge_vertex_face_query_permutation_and_units(self):
        triangles=np.asarray([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]])
        points=np.asarray([[.2,.3,.4],[.6,-.1,.8],[1.1,.7,.6]])
        weights=np.asarray([.2,.3,.5]);base=metric(triangles,points,weights,2.)
        moved=metric(triangles[::-1][:,[2,0,1]],points[::-1],weights[::-1],2.)
        np.testing.assert_allclose(base['per_point_distances'],moved['per_point_distances'][::-1],atol=1e-14,rtol=1e-14)
        self.assertAlmostEqual(base['RMS'],moved['RMS'],delta=1e-14)
        scaled=metric(triangles*7,points*7,weights,14.)
        self.assertAlmostEqual(base['RMS_bbox_pct'],scaled['RMS_bbox_pct'],delta=1e-12)
        self.assertAlmostEqual(base['p95_bbox_pct'],scaled['p95_bbox_pct'],delta=1e-12)
        self.assertEqual(base['coverage_by_bbox_fraction'],scaled['coverage_by_bbox_fraction'])

    def test_finite_endpoint_not_infinite_line_and_degenerate_line(self):
        triangles=np.asarray([[[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]]])
        result=metric(triangles,np.asarray([[2.,0.,0.],[.5,.2,0.]]),np.ones(2),1.)
        np.testing.assert_allclose(result['per_point_distances'],[1.,.2],atol=1e-14,rtol=1e-14)
        self.assertEqual(result['zero_length_edge_count'],1)
        self.assertEqual(check_surface_lower_bound(result['per_point_distances'],[1.,.2],1.)['status'],'PASS')
        self.assertEqual(check_surface_lower_bound([0.],[.1],1.)['status'],'FAIL')
        self.assertEqual(check_surface_lower_bound([.1-5e-11],[.1],1.)['status'],'PASS')

    def test_invalid_input_rejected_and_metric_only_five_registered_thresholds(self):
        with self.assertRaises(ValueError):metric(np.zeros((1,3,3)),[[float('nan'),0.,0.]],[1.],1.)
        self.assertEqual(COVERAGE_FRACTIONS,(.0005,.001,.002,.005,.01))
        self.assertEqual([x['parameter'] for x in CASES if x['kind']=='collinear'],[.001,.01,.05,.1,.2])
        self.assertEqual(len(CASES),19)

    def test_production_quadrature_normalizes_once_and_preserves_parameter_gradient(self):
        case=dict(name='units_fixture',kind='collinear',parameter_name='delta',parameter=.1)
        target=fixed_target(ell=2.,bbox_L=3.,ledger=COUNTS)
        parameter=torch.tensor(.1,dtype=torch.float64,requires_grad=True)
        value=loss_at(case,parameter,5,target,COUNTS,ell=2.)
        COUNTS['parameter_autograd_attempts']+=1
        gradient=torch.autograd.grad(value,parameter)[0];COUNTS['parameter_autograd_returns']+=1
        reference=analytic(case,ell=2.,bbox_L=3.);COUNTS['analytic_reference_evaluations']+=1
        self.assertAlmostEqual(float(value),reference['five_point_loss'],delta=1e-14)
        self.assertAlmostEqual(float(gradient),reference['five_point_parameter_derivative'],delta=1e-14)
        scaled=fixed_target(ell=14.,bbox_L=21.,ledger=COUNTS)
        scaled_case=dict(case,parameter=.7)
        scaled_value=loss_at(scaled_case,torch.tensor(.7,dtype=torch.float64),5,scaled,COUNTS,ell=14.)
        self.assertAlmostEqual(float(value),float(scaled_value),delta=1e-14)


def run_cpu_tests(out):
    path=Path(out)/'cpu_math_tests.json'
    if path.exists():raise FileExistsError('CPU fixture record already exists')
    if not path.parent.is_dir():raise FileNotFoundError('Root must create isolated output directory')
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only fixture process required')
    COUNTS.clear();COUNTS.update(new_counts());MEASUREMENTS.clear();start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MathTests))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),counts=dict(COUNTS),measurements=dict(MEASUREMENTS),
        metric_spec=METRIC_SPEC,quadrature_policy=POLICY,seconds=time.perf_counter()-start,CUDA_initialized=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('edge_metric.py'),Path(__file__).with_name('quadrature.py'))})
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    if not result.wasSuccessful():raise AssertionError('CPU fixture failure; record retained, no automatic retry')
    return report
