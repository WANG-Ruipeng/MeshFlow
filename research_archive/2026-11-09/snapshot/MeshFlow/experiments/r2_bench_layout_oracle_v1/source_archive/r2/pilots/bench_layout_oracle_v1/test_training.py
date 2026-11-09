"""CPU-only schedule/optimizer ownership tests; no generator forward or CUDA."""
from __future__ import annotations
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from . import common as C, prepare as P, training as T
sys.path.insert(0,str(C.ROOT/'r2/src'))


class TinyOwnedParameters(torch.nn.Module):
    readout_mode='none'
    def __init__(self):
        super().__init__()
        self.first=torch.nn.Parameter(torch.ones(2,3))
        self.second=torch.nn.Parameter(torch.full((3,),2.))
        self.eval()


def initialized_optimizer(model, step):
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)
    for p in model.parameters():
        opt.state[p]=dict(step=torch.tensor(float(step)),exp_avg=torch.full_like(p,.25),exp_avg_sq=torch.full_like(p,.125))
    return opt


class TrainingContracts(unittest.TestCase):
    def test_new_branch_does_not_reorder_or_reset_inherited_optimizer(self):
        from r2.pilots.bench_full_seat_curriculum_v1.training_state import validate_optimizer
        model=TinyOwnedParameters(); opt=initialized_optimizer(model,25500)
        original_names=[n for n,_ in model.named_parameters()]
        original_ids=[id(p) for p in model.parameters()]
        old_state={id(p):{k:v.clone() for k,v in s.items()} for p,s in opt.state.items()}
        model.layout_branch=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(2))])
        view=T.BaseParameters(model)
        self.assertEqual([n for n,_ in view.named_parameters()],original_names)
        self.assertEqual([id(p) for p in view.parameters()],original_ids)
        self.assertEqual(validate_optimizer(view,opt,25500),[original_names])
        for p,s in opt.state.items():
            for k,v in s.items(): self.assertTrue(torch.equal(v,old_state[id(p)][k]))
        with self.assertRaises(ValueError): validate_optimizer(model,opt,25500)

    def test_layout_optimizer_has_separate_local_cursor(self):
        model=TinyOwnedParameters(); model.layout_branch=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2))])
        opt=initialized_optimizer(model.layout_branch,1)
        T.check_layout_optimizer(model,opt,1)
        with self.assertRaises(ValueError): T.check_layout_optimizer(model,opt,25501)
        opt.param_groups[0]['lr']=2e-5
        with self.assertRaises(ValueError): T.check_layout_optimizer(model,opt,1)

    def test_incomplete_ledger_cannot_resume(self):
        counts=dict(train_forward=8,train_backward=8,optimizer_update=1)
        rows=[]
        for kind in ('train_forward','train_backward'):
            rows += [dict(event='CALL_BEGIN',kind=kind,context=dict(arm='PLAIN',step=1,microbatch=i)) for i in range(8)]
        rows += [dict(event='CALL_BEGIN',kind='optimizer_update',context=dict(arm='PLAIN',step=1))]
        ledger=SimpleNamespace(pending=[],failed=[],attempted=counts.copy(),successful=counts.copy(),rows=rows,
            limits=dict(train_forward=8000,train_backward=8000,optimizer_update=1000))
        T.require_cursor(ledger,1)
        ledger.successful['train_backward']=7
        with self.assertRaises(RuntimeError): T.require_cursor(ledger,1)
        ledger.successful['train_backward']=8; ledger.pending=[1]
        with self.assertRaises(RuntimeError): T.require_cursor(ledger,1)

    def test_actual_parent_schedule_pairing_and_group_fm_coefficients(self):
        # Real old streams are read only. New files and resource calls are mocked.
        from r2.pilots.bench_three_mechanisms_v1 import common as old
        manifest=C.read(old.OUT/'data/manifest.json')
        panel=C.read(old.OUT/'data/eval_manifest.json')
        source=C.read(C.PRIOR/'data/schedule_PLAIN1500.json')
        coefs=C.read(C.PRIOR/'data/coefficients_PLAIN1500.json')
        written={}
        def read(path):
            if path==C.OUT/'data/manifest.json': return copy.deepcopy(manifest)
            if path==C.OUT/'data/eval_manifest.json': return copy.deepcopy(panel)
            if path==C.PRIOR/'data/schedule_PLAIN1500.json': return copy.deepcopy(source)
            if path==C.PRIOR/'data/coefficients_PLAIN1500.json': return copy.deepcopy(coefs)
            raise AssertionError(path)
        def write(path,value,**kwargs): written[str(path)]=copy.deepcopy(value)
        with patch.object(C,'read',side_effect=read), patch.object(C,'write',side_effect=write), \
             patch.object(C,'verify'),patch.object(C,'sha',return_value='CPU_TEST_ONLY'), \
             patch.object(C,'register_resource',return_value={'model_calls':0}):
            P.prepare()
        schedule=written[str(C.OUT/'data/schedule.json')]['steps']
        coeff=written[str(C.OUT/'data/coefficients.json')]
        self.assertEqual(len(schedule),1000); self.assertEqual(len(coeff),1000)
        self.assertEqual([e['stage_step'] for e in schedule],list(range(1,1001)))
        T.validate_stream(schedule,coeff)
        with self.assertRaises(ValueError): T.validate_stream(schedule[:-1],coeff)
        broken=copy.deepcopy(schedule); broken[0]['samples'][1]['layout_drop']=not broken[0]['samples'][0]['layout_drop']
        with self.assertRaises(ValueError): T.validate_stream(broken,coeff)
        for e,c,original in zip(schedule,coeff,source['steps']):
            self.assertEqual([r['parent_id'] for r in e['samples']],[r['parent_id'] for r in original['samples']])
            for start in (0,2,4,6):
                a,b=e['samples'][start:start+2]
                self.assertEqual(a['parent_id'],b['parent_id'])
                self.assertEqual(a['case_sha256'],b['case_sha256'])
                self.assertEqual(a['layout_drop'],b['layout_drop'])
                self.assertEqual(a['layout_drop'],C.seed('layout_drop',e['stage_step'],start//2)%5==0)
            D=c['denominator']; counts=c['micro_free_scalar_counts']; gains=c['micro_gains']; groups=c['micro_groups']
            T.validate_allocation(c,counts,groups,gains,e['stage_step'])
            for group,weight in c['reference_group_mse_weights'].items():
                self.assertAlmostEqual(sum(n*g for n,g,k in zip(counts,gains,groups) if k==group)/D,weight,places=12)
        badcounts=coeff[0]['micro_free_scalar_counts'].copy(); badcounts[0]+=9
        with self.assertRaises(ValueError):
            T.validate_allocation(coeff[0],badcounts,coeff[0]['micro_groups'],coeff[0]['micro_gains'],1)


if __name__=='__main__': unittest.main()
