"""CPU-only data/loss checks; no real model or optimizer update is executed."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from native_t1.data import (PatchCase,SamplingCondition,load_archive,file_sha256,
    NativeTrainingStream,mixed_k_plan,shared_sample_hash,native_collate,model_inputs)
from native_t1.training import masked_fm_loss


def fake_cases():
    result={}
    for parent in (0,1):
        pre=np.random.RandomState(70+parent).randn(112,3,3).astype(np.float32)
        full=(pre*2).reshape(112,9)
        for K in (2,4,8,12):
            ids=np.arange(K,dtype=np.int64); free=np.arange(K,112,dtype=np.int64)
            result[parent,K]=PatchCase(parent,K,full[ids].reshape(K,3,3),full,full[free],ids,free,
                float(np.linalg.norm(np.ptp(full.reshape(-1,3),axis=0))),pre[free])
    return result


class NativeDataTests(unittest.TestCase):
    def test_archive_identity_and_model_condition_allowlist(self):
        cases=fake_cases(); arrays={}
        for (parent,K),case in cases.items():
            arrays[f'parent{parent}_target']=case.full_target
            for key in ('constraints','free_target','source_face_ids','free_source_ids','pre_ot_free'):
                arrays[f'parent{parent}_K{K}_{key}']=getattr(case,key)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'inputs.npz'; np.savez(path,**arrays)
            loaded=load_archive(path,expected_sha256=file_sha256(path))
            self.assertEqual(set(loaded.cases),set(cases))
            np.testing.assert_array_equal(loaded.cases[1,12].constraints,cases[1,12].constraints)
            condition=loaded.cases[1,12].sampling_condition()
            self.assertEqual(set(vars(condition)),{'constraints','N'})
            self.assertFalse(condition.constraints.flags.writeable)
            self.assertEqual(condition.known_mask.sum(),12)
            with self.assertRaises(ValueError): load_archive(path,expected_sha256='0'*64)

    def test_wrong_decomposition_and_dtype_fail_closed(self):
        case=fake_cases()[0,12]
        wrong=case.constraints.copy(); wrong[0,0,0]+=1
        with self.assertRaises(ValueError):
            PatchCase(0,12,wrong,case.full_target,case.free_target,case.source_face_ids,case.free_source_ids,case.bbox_diagonal)
        with self.assertRaises(ValueError): SamplingCondition(case.constraints.astype(np.float64))
        with self.assertRaises(ValueError):
            PatchCase(0,12,case.constraints,case.full_target,case.free_target,case.source_face_ids,case.free_source_ids[::-1],case.bbox_diagonal)
        with self.assertRaises(ValueError):
            PatchCase(0,12,case.constraints,case.full_target,case.free_target,case.source_face_ids,case.free_source_ids,case.bbox_diagonal,case.pre_ot_free+1)

    def test_mixed_k_schedule_preserves_global_denominator(self):
        plan=[mixed_k_plan(i) for i in range(3)]
        self.assertEqual([sum((112-K)*9 for _,K in p) for p in plan],[7524,7452,7488])
        self.assertEqual(Counter(pair for p in plan for pair in p),Counter({(p,K):4 for p in (0,1) for K in (4,8,12)}))

    def test_free_only_stream_independent_donor_rng_and_restore(self):
        cases=fake_cases()
        correct=NativeTrainingStream(cases,'Native_correct',10)
        independent=NativeTrainingStream(cases,'Native_independent',10)
        for parent,K in ((0,4),(1,8),(0,12)):
            a=correct._sample(parent,K); b=independent._sample(parent,K)
            self.assertEqual(shared_sample_hash(a),shared_sample_hash(b))
            self.assertEqual(a['free_gaussian_before_OT'].shape,(112-K,3,3))
            self.assertEqual(sorted(a['source_face_ids'].tolist()),list(range(112)))
            self.assertEqual(np.count_nonzero(a['u'][a['known_mask']]),0)
            np.testing.assert_array_equal(a['xt'][~a['known_mask']],b['xt'][~b['known_mask']])
            batch=native_collate([a]); args=model_inputs(batch)
            self.assertEqual(len(args),5)
            self.assertIs(args[0],batch['xt'])
            self.assertTrue(torch.equal(batch['xt'][batch['known_mask']],batch['context'][batch['known_mask']]))
        saved=correct.state_dict(); a=correct._sample(1,12); correct.load_state_dict(saved); b=correct._sample(1,12)
        self.assertEqual(shared_sample_hash(a),shared_sample_hash(b))

    def test_masked_loss_excludes_known_padding_and_scales_accumulation(self):
        velocity=torch.ones((1,4,9),dtype=torch.float32,requires_grad=True)
        target=torch.zeros_like(velocity)
        valid=torch.tensor([[True,True,True,False]])
        known=torch.tensor([[True,False,False,False]])
        loss=masked_fm_loss(velocity,target,valid,known,denominator=36)
        self.assertEqual(float(loss),.5)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(velocity.grad[:,[0,3]])),0)
        torch.testing.assert_close(velocity.grad[:,1:3],torch.full((1,2,9),2/36))
        with self.assertRaises(ValueError): masked_fm_loss(velocity,target,valid,known,denominator=9)
        with self.assertRaises(ValueError): masked_fm_loss(velocity,target,valid,valid)


if __name__=='__main__': unittest.main()
