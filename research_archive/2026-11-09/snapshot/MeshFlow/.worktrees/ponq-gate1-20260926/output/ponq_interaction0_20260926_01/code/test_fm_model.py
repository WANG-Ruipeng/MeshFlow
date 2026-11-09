"""Synthetic math/interface tests only; no dataset training."""
import unittest
import torch
from fm_model import DeepSetsVelocity,local_velocity_gradient,compose_velocity,validate_chart


class FMPilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(260926)
        torch.set_num_threads(1)

    def test_zero_initialization_and_parameter_structure(self):
        a=DeepSetsVelocity(); b=DeepSetsVelocity()
        z=torch.randn(2,32,22); t=torch.rand(2)
        self.assertTrue(torch.equal(a(z,t),torch.zeros_like(z)))
        self.assertEqual([(k,v.shape) for k,v in a.state_dict().items()],
                         [(k,v.shape) for k,v in b.state_dict().items()])

    def test_permutation_equivariance_nonzero_network_and_transport(self):
        model=DeepSetsVelocity(width=24).double()
        torch.nn.init.normal_(model.decoder[-1].weight,std=.1)
        z=torch.randn(2,40,22,dtype=torch.float64);t=torch.rand(2,dtype=torch.float64)
        mean=torch.zeros(22,dtype=torch.float64);scale=torch.ones_like(mean)
        permutation=torch.randperm(40)
        raw=model(z,t); perm=model(z[:,permutation],t)
        torch.testing.assert_close(perm,raw[:,permutation],rtol=1e-11,atol=1e-11)
        total,_=compose_velocity(raw,z,mean,scale,'transport_residual')
        perm_total,_=compose_velocity(perm,z[:,permutation],mean,scale,'transport_residual')
        torch.testing.assert_close(perm_total,total[:,permutation],rtol=1e-10,atol=1e-10)

    def test_affine_gradient_matches_ridge_closed_form(self):
        x=torch.randn(2,45,3,dtype=torch.float64)
        truth=torch.tensor([[.2,.1,-.05],[.03,-.4,.07],[0.,.06,.15]],dtype=torch.float64)
        velocity=x@truth.T+torch.tensor([.3,-.2,.7],dtype=torch.float64)
        L,d=local_velocity_gradient(x,velocity)
        b=torch.arange(2)[:,None,None];ids=d['neighbor_indices']
        X=(x[b,ids]-x[:,:,None,:])/d['h'][...,None,None]
        G=X.transpose(-1,-2)@(d['weights'][...,None]*X)
        expected=torch.linalg.solve(G+d['ridge_lambda'][...,None,None]*torch.eye(3,dtype=x.dtype),G@truth.T).transpose(-1,-2)
        torch.testing.assert_close(L,expected,rtol=1e-11,atol=1e-11)
        self.assertLess(float((L-truth).abs().max()),.01)

    def test_standardized_transport_uses_physical_factor_and_velocity(self):
        z=torch.randn(1,40,22,dtype=torch.float64);raw=torch.randn_like(z)*.1
        mean=torch.linspace(-.3,.4,22,dtype=z.dtype)
        scale=torch.tensor([2.]*3+[.4]*3+[3.]*16,dtype=z.dtype)
        validate_chart(mean,scale)
        total,d=compose_velocity(raw,z,mean,scale,'transport_residual')
        F=(z*scale+mean)[...,6:].reshape(1,40,4,4)
        K=torch.zeros(1,40,4,4,dtype=z.dtype);K[...,:3,:3]=d['L']
        expected=(-K.transpose(-1,-2)@F).reshape(1,40,16)/3.
        torch.testing.assert_close(total[...,6:]-raw[...,6:],expected)
        torch.testing.assert_close(total[...,:6],raw[...,:6])
        direct,dd=compose_velocity(raw,z,mean,scale,'direct')
        self.assertTrue(torch.equal(direct,raw))
        torch.testing.assert_close(dd['L'],d['L'])

    def test_total_velocity_supervision_backpropagates_through_position_head(self):
        z=torch.randn(1,40,22,dtype=torch.float64);raw=torch.zeros_like(z,requires_grad=True)
        mean=torch.zeros(22,dtype=z.dtype);scale=torch.ones_like(mean)
        total,_=compose_velocity(raw,z,mean,scale,'transport_residual')
        target=torch.randn_like(z[...,6:])
        ((total[...,6:]-target)**2).mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad[...,:3].abs().sum()),0.)

    def test_coupling_reorders_complete_tokens_and_stream_is_paired(self):
        import numpy as np
        from train_fm_pilot import BalancedSchedule,coupled_batch
        states=np.random.default_rng(7).normal(size=(3,24,22))
        mean=np.zeros(22);scale=np.ones(22)
        left=BalancedSchedule(3,9);right=BalancedSchedule(3,9)
        ga=torch.Generator().manual_seed(11);gb=torch.Generator().manual_seed(11)
        a=coupled_batch(states,mean,scale,left,ga)
        b=coupled_batch(states,mean,scale,right,gb)
        for i in range(3):self.assertTrue(torch.equal(a[i],b[i]))
        self.assertEqual(a[3],b[3])
        self.assertLessEqual(max(a[3].count(i) for i in range(3))-min(a[3].count(i) for i in range(3)),1)
        for batch,index in enumerate(a[3]):
            original=states[index].astype(np.float32)
            received=a[2][batch].numpy()
            self.assertEqual({tuple(r) for r in original},{tuple(r) for r in received})

    def test_same_seed_gives_identical_initial_parameters(self):
        torch.manual_seed(260926);a=DeepSetsVelocity()
        torch.manual_seed(260926);b=DeepSetsVelocity()
        self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))

    def test_factor_psd_and_chart_rejection(self):
        F=torch.randn(32,4,4,dtype=torch.float64)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(F@F.transpose(-1,-2)).min()),-1e-12)
        mean=torch.zeros(22);scale=torch.ones(22);scale[1]=2
        with self.assertRaises(ValueError):validate_chart(mean,scale)


if __name__=='__main__':
    unittest.main(verbosity=2)
