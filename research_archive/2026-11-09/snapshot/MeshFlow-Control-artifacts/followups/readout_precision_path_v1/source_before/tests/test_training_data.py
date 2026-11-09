"""Synthetic CPU-only data/coupling/resume contracts, no model construction."""
import copy
import itertools
import unittest
import numpy as np
import torch
from meshflow_control.data.dataset import gaussian, training_time, permutations, collate, model_inputs
from meshflow_control.data.io import array_hash
from meshflow_control.data.stream import make_plan, HybridStream, SEED, NAMESPACE
from meshflow_control.training.coupling import nested_ot_map, to_target_slots, to_noise_slots, coupling_path, validate_map
from meshflow_control.training.losses import masked_fm_loss

def manifest():
    parents=["parent_"+str(i).zfill(2) for i in range(32)]
    tasks=[]
    for i,uid in enumerate(parents):
        for j in range(15 if i<5 else 14):
            tasks.append(dict(uid=uid,task_id=uid+"_"+str(j),N=160,K=32 if j%2==0 else 64,
                ratio=.2 if j%2==0 else .4,role="train",status="READY"))
    return {"splits":{"train":parents},"tasks":tasks}

class TrainingDataTests(unittest.TestCase):
    def test_plan_fixed_distinct_parents_exposures(self):
        p=make_plan(manifest())
        self.assertEqual(p,make_plan(manifest()))
        self.assertEqual(len(p["records"]),8000)
        self.assertEqual(set(p["object_counts"].values()),{250})
        self.assertNotIn(SEED,{20261004,20261005,20261006,20261007,20261008,20261009})
        for i in range(0,8000,8):
            rows=p["records"][i:i+8]
            self.assertEqual(len({r["uid"] for r in rows}),8)
            self.assertEqual(sum(r["alpha"]==1. for r in rows),4)
        self.assertEqual(len({r["gaussian_seed"] for r in p["records"]}),8000)

    def test_noise_precision_and_rng_isolation(self):
        np.random.seed(123); before=np.random.get_state()
        z=gaussian(41,3); t=training_time(17); free,corners,order=permutations(7,6,2)
        after=np.random.get_state()
        self.assertTrue(np.array_equal(before[1],after[1])); self.assertEqual(before[2:],after[2:])
        self.assertEqual(z.dtype,np.float64)
        np.testing.assert_array_equal(z,np.random.RandomState(41).randn(3,3,3))
        self.assertEqual(t.dtype,np.float32)
        self.assertTrue(np.all(np.sort(corners,1)==np.arange(3)))
        self.assertEqual(sorted(order.tolist()),list(range(6)))

    def test_ot_bruteforce_cost_and_inverse(self):
        rng=np.random.RandomState(19); y=rng.randn(3,3,3).astype(np.float32);z=rng.randn(3,3,3)
        face,corner=nested_ot_map(y,z); paired=to_target_slots(z,face,corner)
        actual=np.linalg.norm(y.astype(np.float64)-paired,axis=(1,2)).sum()
        perms=list(itertools.permutations(range(3)))
        best=min(sum(min(np.linalg.norm(y[i].astype(np.float64)-z[f[i],list(c)]) for c in perms)
                     for i in range(3)) for f in perms)
        self.assertAlmostEqual(actual,best,places=12)
        np.testing.assert_array_equal(to_noise_slots(paired,face,corner),z)
        np.testing.assert_array_equal(to_target_slots(to_noise_slots(y,face,corner),face,corner),y)

    def test_hybrid_order_and_velocity(self):
        y=np.arange(18,dtype=np.float32).reshape(2,3,3)/7
        z=gaussian(8,2);eps=gaussian(9,2); t=np.float32(.7)
        face,corner=nested_ot_map(y,z);target=to_noise_slots(y*np.float32(2),face,corner)
        mixed,xt,u=coupling_path(z,target,eps,t)
        np.testing.assert_array_equal(mixed,np.float32(np.sqrt(.75))*z.astype(np.float32)+np.float32(.5)*eps.astype(np.float32))
        np.testing.assert_array_equal(u,target-mixed)
        np.testing.assert_array_equal(xt,(np.float32(1)-t)*mixed+t*target)
        self.assertEqual(mixed.dtype,np.float32)

    def test_invalid_mapping_and_time(self):
        with self.assertRaises(ValueError):validate_map(np.array([0,0],np.int64),np.tile(np.arange(3),(2,1)))
        a=np.ones((2,3,3),np.float32)
        with self.assertRaises(ValueError):coupling_path(a,a,a,np.nan)
        with self.assertRaises(ValueError):coupling_path(a,a,a,.5,lam=.5)

    def test_global_free_denominator_padding_gradients(self):
        v1=torch.arange(36,dtype=torch.float32).reshape(1,4,9).requires_grad_()
        v2=torch.ones(1,3,9,requires_grad=True)
        valid1=torch.tensor([[True,True,True,False]]);known1=torch.tensor([[True,False,False,False]])
        valid2=torch.tensor([[True,True,False]]);known2=torch.tensor([[True,False,False]])
        denom=27
        loss=masked_fm_loss(v1,torch.zeros_like(v1),valid1,known1,denominator=denom)+masked_fm_loss(v2,torch.zeros_like(v2),valid2,known2,denominator=denom)
        loss.backward()
        expected=torch.zeros_like(v1);expected[:,1:3]=2*v1.detach()[:,1:3]/denom
        torch.testing.assert_close(v1.grad,expected)
        self.assertEqual(float(v1.grad[:,0].abs().sum()+v1.grad[:,3].abs().sum()),0.)
        self.assertAlmostEqual(float(v2.grad[0,1,0]),2/27,places=7)

    def test_model_input_allowlist_and_padding(self):
        sample=dict(xt=np.ones((3,9),np.float32),u=np.zeros((3,9),np.float32),
            context=np.zeros((3,9),np.float32),t=np.float32(.2),y=np.int64(3),
            known_mask=np.array([1,0,0],bool),valid_mask=np.ones(3,bool),
            UID="private",Gamma="private",GT="private")
        batch=collate([sample],pad_to=6)
        args=model_inputs(batch)
        self.assertEqual(len(args),5);self.assertEqual(tuple(args[0].shape),(1,6,9))
        self.assertEqual(int(args[3].sum()),3)
        self.assertTrue(all(isinstance(a,torch.Tensor) for a in args))
        self.assertTrue(torch.equal(args[0][:,3:],torch.zeros(1,3,9)))

    def test_restore_strict_cursor_identity(self):
        stream=object.__new__(HybridStream)
        stream.arm="A";stream.batch_index=0;stream.plan_sha256="p";stream.implementation_sha256="i"
        stream.data=type("FakeData",(),{"manifest_sha256":"d"})()
        state=stream.state_dict();state.update(batch_index=500,sample_index=4000)
        stream.load_state_dict(state);self.assertEqual(stream.batch_index,500)
        bad=dict(state, sample_index=4001)
        with self.assertRaises(ValueError):stream.load_state_dict(bad)
        bad=dict(state, plan_sha256="changed")
        with self.assertRaises(ValueError):stream.load_state_dict(bad)


    def test_recovery_rejects_uncovered_or_duplicate_execution(self):
        from meshflow_control.training.trainer import validate_recovery
        events=[dict(event=k,step=s) for s in (1,2) for k in ("INPUT_BEGIN","INPUT_DONE","UPDATE_BEGIN","UPDATE_DONE")]
        validate_recovery(events,2)
        for bad in (events[:-1],events+[events[0]],events+[dict(event="INPUT_BEGIN",step=3)]):
            with self.assertRaises(RuntimeError):validate_recovery(bad,2)

if __name__=="__main__":unittest.main()
