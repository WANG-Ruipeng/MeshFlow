"""Small tensor/geometry-contract tests only; no model construction or CUDA."""
import math, unittest, subprocess, sys, json
import numpy as np
import torch
from . import rl as R, reward as Q


class ExactDetectorProcessTests(unittest.TestCase):
    def test_original_detector_and_finite_surface_distance(self):
        # This new process never imports torch or constructs a model. It checks
        # the real detector and finite-triangle query, rather than fake scans.
        code='''
import json, sys, numpy as np
from r2.pilots.bench_three_mechanisms_v1 import reward as Q
raw=np.asarray([[[20,0,0],[21,0,0],[20,1,0]],
                [[-1,-1,0],[1,-1,0],[0,1,0]],
                [[0,-.5,-1],[0,-.5,1],[0,.5,0]],
                [[5,0,0],[6,0,0],[5,1,0]]],np.float32)
ref=Q.reference(raw,1)
assert ref['local_bad_pairs']==[[0,1]],ref
same=Q.evaluate(raw,raw.copy(),1,ref)
assert same['reward']==0 and same['protection_pass'],same
fixed=raw.copy();fixed[2,:,2]+=4
good=Q.evaluate(raw,fixed,1,ref)
assert good['reward']==1 and good['protection_pass'],good
assert 'torch' not in sys.modules
print(json.dumps(dict(status='PASS',reference_pairs=ref['local_bad_pairs'],
    same_reward=same['reward'],repaired_reward=good['reward'],
    protected_query_count=good['protected_query_count'],torch_imported=False)))
'''
        p=subprocess.run([sys.executable,'-c',code],cwd=str(R.C.ROOT),capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr+p.stdout)
        self.assertEqual(json.loads(p.stdout.strip())['status'],'PASS')


def scans(ff=(),cf=()):
    return {kind:dict(contacts=[dict(pair=list(pair),classification='proper_transverse_penetration') for pair in pairs])
        for kind,pairs in [('FF',ff),('CF',cf)]}


def mesh():
    t=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    return np.stack([t+np.asarray([i*2,0,0],np.float32) for i in range(6)])


class ProbabilityTests(unittest.TestCase):
    def test_joint_density_sum(self):
        mean=torch.zeros((1,3,9),dtype=torch.float32,requires_grad=True)
        a=torch.full_like(mean,.01).detach();mask=torch.tensor([[False,True,True]])
        p=R.joint_log_prob(a,mean,mask)
        expected=-18*(.01**2)/(2*R.SIGMA**2)-18*math.log(R.SIGMA)-9*math.log(2*math.pi)
        self.assertAlmostEqual(float(p),expected,places=5)
        p.sum().backward();self.assertTrue(torch.equal(mean.grad[:,0],torch.zeros((1,9))))
        self.assertTrue(bool((mean.grad[:,1:]!=0).all()))

    def test_outside_mean_no_adapter_gradient(self):
        x=torch.zeros((1,3,9),dtype=torch.float32)
        vb=torch.ones_like(x);vc=torch.full_like(x,5.,requires_grad=True)
        m=torch.tensor([[False,True,False]]);known=torch.tensor([[True,False,False]])
        mu=R.mixed_mean(x,vb,vc,m,known)
        self.assertTrue(torch.equal(mu[:,0],x[:,0]));self.assertTrue(torch.equal(mu[:,2],x[:,2]+R.DT*vb[:,2]))
        mu.sum().backward();self.assertTrue(torch.equal(vc.grad[:,2],torch.zeros((1,9))))
        self.assertTrue(torch.allclose(vc.grad[:,1],torch.full((1,9),R.DT)))

    def test_action_detached_required(self):
        x=torch.zeros((1,1,9),requires_grad=True);m=torch.ones((1,1),dtype=torch.bool)
        with self.assertRaises(ValueError):R.joint_log_prob(x,x,m)

    def test_mask_c_conflict_rejected(self):
        x=torch.zeros((1,1,9));m=torch.ones((1,1),dtype=torch.bool)
        with self.assertRaises(ValueError):R.mixed_mean(x,x,x,m,m)

    def test_rloo_continuous(self):
        r=np.asarray([1.,.2,-.3,-1.]);a=R.rloo(r)
        self.assertTrue(np.allclose(a,[r[i]-np.delete(r,i).mean() for i in range(4)]))
        self.assertAlmostEqual(a.sum(),0.)
        self.assertTrue(np.array_equal(R.rloo([0.]*4),np.zeros(4)))

    def test_kl_sum_selected(self):
        x=torch.zeros((1,3,9));y=x.clone();y[:,1]=.01;y[:,2]=100
        m=torch.tensor([[False,True,False]])
        self.assertAlmostEqual(float(R.analytic_kl(x,y,m)),18.,places=5)

    def test_empty_gaussian_dimension(self):
        x=torch.zeros((1,2,9));m=torch.zeros((1,2),dtype=torch.bool)
        self.assertEqual(float(R.joint_log_prob(x,x,m)),0.)


class RewardTests(unittest.TestCase):
    def test_mask_preserves_complete_endpoints_and_cap(self):
        m,p=Q.mask_from_pairs(20,2,[(0,1),(2,3),(4,5),(6,7),(8,9)])
        self.assertEqual(int(m.sum()),8);self.assertFalse(m[:2].any())
        for i,j in p:self.assertTrue(m[i+2] and m[j+2])

    def test_mask_stable_pair_order(self):
        a=Q.mask_from_pairs(20,2,[(9,10),(0,1),(3,4)])[0]
        b=Q.mask_from_pairs(20,2,[(3,4),(0,1),(9,10)])[0]
        self.assertTrue(np.array_equal(a,b))

    def test_zero_reference_no_fake_improvement(self):
        a=mesh();s=scans();ref=Q.reference(a,1,s)
        result=Q.evaluate(a,a,1,ref,s,distance_fn=lambda p,t:np.zeros(len(p)))
        self.assertEqual(result['reward'],0);self.assertTrue(result['mask_empty']);self.assertTrue(result['protection_pass'])

    def test_local_pair_removed_reward(self):
        a=mesh();ref=Q.reference(a,1,scans(ff=[(0,1)]))
        result=Q.evaluate(a,a,1,ref,scans(),distance_fn=lambda p,t:np.zeros(len(p)))
        self.assertEqual(result['reward'],1.);self.assertEqual(result['eliminated_local_FF_pairs'],[[0,1]])

    def test_new_cf_blocks_reward(self):
        a=mesh();ref=Q.reference(a,1,scans(ff=[(0,1)]))
        result=Q.evaluate(a,a,1,ref,scans(cf=[(0,2)]),distance_fn=lambda p,t:np.zeros(len(p)))
        self.assertEqual(result['reward'],-1.);self.assertIn('CF_PAIR_COUNT_INCREASE',result['rejection_reasons'])

    def test_new_outside_conflict_blocks_reward(self):
        a=mesh();ref=Q.reference(a,1,scans(ff=[(0,1)]))
        result=Q.evaluate(a,a,1,ref,scans(ff=[(3,4)]),distance_fn=lambda p,t:np.zeros(len(p)))
        self.assertEqual(result['reward'],-1.);self.assertIn('OUTSIDE_FF_PAIR_COUNT_INCREASE',result['rejection_reasons'])

    def test_deletion_not_legal(self):
        a=mesh();ref=Q.reference(a,1,scans(ff=[(0,1)]))
        with self.assertRaises(ValueError):Q.evaluate(a,a[:-1],1,ref,scans(),distance_fn=lambda p,t:np.zeros(len(p)))

    def test_degenerate_block(self):
        a=mesh();b=a.copy();b[2]=b[2,0];ref=Q.reference(a,1,scans(ff=[(0,1)]))
        result=Q.evaluate(a,b,1,ref,scans(),distance_fn=lambda p,t:np.zeros(len(p)))
        self.assertEqual(result['reward'],-1.);self.assertIn('NEW_DEGENERATE_FACE',result['rejection_reasons'])

    def test_coverage_block(self):
        a=mesh();ref=Q.reference(a,1,scans(ff=[(0,1)]))
        result=Q.evaluate(a,a,1,ref,scans(),distance_fn=lambda p,t:np.full(len(p),Q.SURFACE_TOLERANCE*2))
        self.assertEqual(result['reward'],-1.);self.assertIn('PROTECTED_SURFACE_COVERAGE_LT98PCT',result['rejection_reasons'])

    def test_frozen_noise_mask_and_repeat(self):
        m=np.asarray([False,True,False,True]);a=Q.draw_policy_noise(123,4,m);b=Q.draw_policy_noise(123,4,m)
        self.assertTrue(np.array_equal(a,b));self.assertFalse(a[:,~m].any());self.assertEqual(a.dtype,np.float32)

    def test_original_cpu_scoring_cannot_import_into_gpu_worker(self):
        with self.assertRaises(RuntimeError):Q.scan(mesh(),1)


if __name__=='__main__':
    if torch.cuda.is_initialized():raise RuntimeError('CPU tests must not initialize CUDA')
    unittest.main()
