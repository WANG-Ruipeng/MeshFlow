"""Small CPU-only standalone loss and source-identity fixtures.

No historical imports/assets, model construction, OT or optimizer updates.
"""
import copy
import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from native_t1.data import load_archive
from native_t1.losses import TrainingGeometryTargets, prepare_targets, evaluate
from native_t1.losses.edge import prepare_edge_target, edge_loss
from native_t1.losses.surface import prepare_distribution_target, distribution_loss

COUNTS = dict(edge_loss_calls=0, L5_loss_calls=0, coordinate_autograd_calls=0,
              finite_difference_evaluations=0, model_forwards=0, OT_calls=0, optimizer_updates=0)
MEASUREMENTS = {}


def loss(name, prediction, target):
    COUNTS['edge_loss_calls' if name=='D_edge' else 'L5_loss_calls'] += 1
    return evaluate(name,prediction,target)


def grad(value, inputs, **kwargs):
    COUNTS['coordinate_autograd_calls'] += 1
    return torch.autograd.grad(value,inputs,**kwargs)


def fixture(dtype=torch.float64):
    V=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.3,-1.,.2],[1.,1.,1.]],dtype=dtype)
    f=torch.tensor([[1,0,3],[2,1,4]]);k=torch.tensor([[0,1,2]])
    return dict(free_gt=V[f],known=V[k],free_ids=f,known_ids=k,bbox_L=1.)


class Edge5LossTests(unittest.TestCase):
    def test_edge_oracle_S3_and_centered_collapse(self):
        label=fixture();target=prepare_edge_target(label)
        pred=label['free_gt'].clone().requires_grad_()
        oracle=loss('D_edge',pred,target)['loss'];g=grad(oracle,pred)[0]
        self.assertLess(abs(float(oracle)),1e-25);self.assertLess(float(g.abs().max()),1e-12)
        moved=pred.detach()+torch.tensor([.08,-.05,.13],dtype=torch.float64)
        expected=float(loss('D_edge',moved,target)['loss'])
        for permutation in itertools.permutations(range(3)):
            ix=list(permutation);fi=torch.tensor([1,0])
            lab=dict(label,free_gt=label['free_gt'][fi][:,ix],known=label['known'][:,ix],
                     free_ids=label['free_ids'][fi][:,ix],known_ids=label['known_ids'][:,ix])
            actual=loss('D_edge',moved[fi][:,ix],prepare_edge_target(lab))['loss']
            self.assertAlmostEqual(float(actual),expected,delta=2e-12)
        one=dict(label,free_gt=label['free_gt'][:1],free_ids=label['free_ids'][:1])
        collapsed=one['free_gt'].clone();collapsed[0,:2]=torch.tensor([.5,0.,0.]);collapsed.requires_grad_()
        value=loss('D_edge',collapsed,prepare_edge_target(one))['loss'];gradient=grad(value,collapsed)[0]
        self.assertAlmostEqual(float(value),.046875,delta=1e-14)
        self.assertEqual(int(torch.count_nonzero(gradient)),0)

    def test_edge_detachment_incidence_and_short_GT_rejection(self):
        label=fixture();label['free_gt'].requires_grad_();label['known'].requires_grad_()
        target=prepare_edge_target(label)
        self.assertTrue(all(not x.requires_grad for x in target.values() if torch.is_tensor(x)))
        pred=(label['free_gt'].detach()+.02).requires_grad_()
        value=loss('D_edge',pred,target)['loss']
        g,gt,c=grad(value,(pred,label['free_gt'],label['known']),allow_unused=True)
        self.assertGreater(float(g.norm()),0);self.assertIsNone(gt);self.assertIsNone(c)
        raw=fixture();extra=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[.2,1.,1.]]],dtype=torch.float64)
        non=dict(raw,free_gt=torch.cat((raw['free_gt'][:1],extra)),free_ids=torch.tensor([[1,0,3],[0,1,9]]))
        no=prepare_edge_target(non);self.assertEqual(no['counts']['eligible_interface_edges'],0)
        self.assertEqual(no['counts']['excluded_nonmanifold_mixed_edges'],1)
        self.assertEqual(no['counts']['excluded_nonmanifold_known_free_pairs'],2)
        tiny=fixture();tiny['free_gt'][0,0,0]=1e-7;tiny['free_gt'][1,1,0]=1e-7;tiny['known'][0,1,0]=1e-7
        with self.assertRaisesRegex(ValueError,'Abnormal GT interface'):prepare_edge_target(tiny)

    def test_L5_oracle_detach_and_unsigned_normal(self):
        label=fixture();gt=label['free_gt'].requires_grad_();target=prepare_distribution_target(gt,1.)
        pred=gt.detach().clone().requires_grad_();value=loss('L5',pred,target)['loss']
        g,reference=grad(value,(pred,gt),allow_unused=True)
        self.assertAlmostEqual(float(value),0.,delta=1e-14);self.assertLess(float(g.abs().max()),1e-12)
        self.assertIsNone(reference)
        permuted=pred.detach().flip(0)[:,[0,2,1]]
        self.assertAlmostEqual(float(loss('L5',permuted,target)['loss']),0.,delta=1e-14)
        shifted=(pred.detach()+torch.tensor([.035,-.017,.023],dtype=torch.float64)).requires_grad_()
        actual=loss('L5',shifted,target);g=grad(actual['loss'],shifted)[0]
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0.)
        self.assertFalse(actual['diagnostics']['predicted_mass_independently_normalized'])
        self.assertEqual(actual['counts']['sinkhorn_solves'],0)

    def test_L5_fixed_GT_area_mass_not_predicted_probability(self):
        gt=fixture()['free_gt'];target=prepare_distribution_target(gt,1.)
        result=loss('L5',gt*2,target)
        self.assertAlmostEqual(result['diagnostics']['predicted_measure_mass'],4.,delta=1e-13)
        self.assertAlmostEqual(result['diagnostics']['GT_measure_mass'],1.,delta=1e-13)
        self.assertFalse(result['diagnostics']['area_floor_applied'])
        self.assertEqual(result['counts']['kernel_matrix_evaluations'],9)
        again=loss('L5',gt*2,target)
        self.assertEqual(again['counts']['kernel_matrix_evaluations'],6)
        self.assertEqual(again['counts']['GTGT_kernel_cache_hits'],1)
        self.assertTrue(torch.equal(result['loss'],again['loss']))

    def test_L5_FP64_finite_difference_and_FP32_autocast(self):
        gt=fixture()['free_gt'];target=prepare_distribution_target(gt,1.)
        pred=(gt+torch.tensor([[[.012,.021,-.011],[.005,-.014,.009],[-.008,.016,.004]],
                              [[-.007,.019,.013],[.004,-.006,.011],[.017,.003,-.002]]],dtype=torch.float64)).requires_grad_()
        value=loss('L5',pred,target)['loss'];analytic=grad(value,pred)[0].numpy();numeric=np.empty_like(analytic);h=1e-6
        for index in np.ndindex(numeric.shape):
            plus=pred.detach().clone();minus=pred.detach().clone();plus[index]+=h;minus[index]-=h
            numeric[index]=(float(loss('L5',plus,target)['loss'])-float(loss('L5',minus,target)['loss']))/(2*h)
            COUNTS['finite_difference_evaluations']+=2
        np.testing.assert_allclose(analytic,numeric,atol=3e-7,rtol=2e-6)
        MEASUREMENTS['L5_FP64_finite_difference_max_abs_error']=float(np.max(np.abs(analytic-numeric)))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            value32=loss('L5',pred.detach().float(),prepare_distribution_target(gt.float(),1.))['loss']
        self.assertEqual(value32.dtype,torch.float32);self.assertAlmostEqual(float(value32),float(value),delta=2e-6)

    def test_only_two_losses_and_preparation_no_predicted_cache(self):
        label=fixture();targets=prepare_targets(label)
        self.assertEqual(set(targets),{'D_edge','L5'})
        with self.assertRaises(ValueError):evaluate('L4',label['free_gt'],targets['L5'])
        with self.assertRaises(ValueError):distribution_loss('L4',label['free_gt'],targets['L5'])
        self.assertFalse(targets['L5']['free_gt'].requires_grad)
        self.assertEqual(targets['L5']['_cache'],{})

    def test_explicit_source_archive_and_permuted_target_identity(self):
        V=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.3,-1.,.2]],np.float32)
        F=np.asarray([[0,1,2],[1,0,3]]*56,np.int64);Y=V[F].reshape(112,9)
        ids=np.arange(4);free_ids=np.arange(4,112)
        with tempfile.TemporaryDirectory(prefix='edge5_cpu_') as td:
            path=Path(td)/'inputs.npz'
            np.savez(path,parent0_source_vertices=V,parent0_source_face_vertex_ids=F,parent0_target=Y,
                parent0_K4_constraints=Y[ids].reshape(4,3,3),parent0_K4_free_target=Y[free_ids],
                parent0_K4_source_face_ids=ids,parent0_K4_free_source_ids=free_ids)
            archive=load_archive(path);bank=TrainingGeometryTargets(archive)
            face=np.arange(112)[::-1].copy();corners=np.asarray([[0,2,1],[1,2,0]]*56)
            coordinates=V[np.take_along_axis(F[face],corners,axis=1)]
            known=face<4;x1=np.zeros((112,9),np.float32);C=x1.copy()
            x1[~known]=coordinates[~known].reshape(-1,9);C[known]=coordinates[known].reshape(-1,9)
            sample=dict(parent_index=0,donor_index=0,K=4,source_face_ids=face,corner_permutations=corners,
                        known_mask=known,valid_mask=np.ones(112,bool),x1=x1,context=C)
            target=bank.sample(sample)
            np.testing.assert_array_equal(target['free_ids'],np.take_along_axis(F[face],corners,axis=1)[~known])
            self.assertEqual(target['free_gt'].tobytes(),coordinates[~known].tobytes())
            self.assertEqual(target['known'].tobytes(),coordinates[known].tobytes())
            with self.assertRaises(ValueError):bank.sample(dict(sample,donor_index=1))
            broken=x1.copy();broken[0,0]+=1
            with self.assertRaises(ValueError):bank.sample(dict(sample,x1=broken))


if __name__=='__main__':
    unittest.main()
