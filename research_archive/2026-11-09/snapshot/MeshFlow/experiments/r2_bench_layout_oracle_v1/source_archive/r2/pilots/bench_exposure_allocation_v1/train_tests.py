"""CPU contract tests: no neural forward, GPU, backward or optimizer update.

Uses parameter-only AdamW fixtures, real CPU target construction and the retained
checkpoint's tensors/metadata, without constructing a generator.
"""
from __future__ import annotations
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from . import training as M
from . import stream as Q
import numpy as np
import torch
from meshflow_control.training.schedule40 import make_optimizer


def contract():
    return dict(updates=1500,source_global_step=19500,final_global_step=21000,
                full_steps=[1500],checkpoint_every=250)


def pool():
    return [dict(parent_id=g+str(i),condition_id=g+str(i),replay_group=g)
            for g,n in (('OLD',22),('NEW',2)) for i in range(n)]


def update(ledger,reg,step):
    for mi in range(8):
        context=dict(arm=ledger.arm,**M.cursor(reg,step),microbatch=mi)
        for kind in ('train_forward','train_backward'):
            ledger.call(kind,lambda:None,context)
    ledger.call('optimizer_update',lambda:None,dict(arm=ledger.arm,**M.cursor(reg,step)))


class ContractTests(unittest.TestCase):
    def test_exact_exposure_totals(self):
        for arm,old,new in ((Q.ARMS[0],6000,6000),(Q.ARMS[1],9000,3000)):
            s=Q.make_schedule(pool(),arm)
            r=Q.validate(s,pool())
            self.assertEqual((r['old_instances'],r['priority_instances']),(old,new))

    def test_slot_rng_augmentation_paired(self):
        a,b=[Q.make_schedule(pool(),arm,40) for arm in Q.ARMS]
        for x,y in zip(a['steps'],b['steps']):
            for u,v in zip(x['samples'],y['samples']):
                for key in ('alpha','scale_xyz','noise_seed','epsilon_seed','time_seed','permutation_seed'):
                    self.assertEqual(u[key],v[key])
        self.assertNotEqual(a['steps'],b['steps'])

    def test_schedule_no_process_rng_mutation(self):
        n=np.random.get_state();t=torch.get_rng_state().clone()
        Q.make_schedule(pool(),Q.ARMS[1],40)
        now=np.random.get_state()
        self.assertEqual(n[0],now[0]);self.assertTrue(np.array_equal(n[1],now[1]));self.assertEqual(n[2:],now[2:])
        self.assertTrue(torch.equal(t,torch.get_rng_state()))

    def test_reject_quota_seed_or_augmentation_mutation(self):
        base=Q.make_schedule(pool(),Q.ARMS[0],40)
        for key,value in (('replay_group','NEW'),('noise_seed',-1),('alpha',1.2),('parent_id','UNKNOWN')):
            bad=copy.deepcopy(base);bad['steps'][0]['samples'][0][key]=value
            with self.assertRaises(ValueError):Q.validate(bad,pool(),40)
        with self.assertRaises(ValueError):Q.validate(base,pool(),40,Q.ARMS[1])

    def test_no_new_pool_or_multicondition(self):
        r=pool();r[-1]['parent_id']=r[0]['parent_id']
        with self.assertRaises(ValueError):Q.make_schedule(r,Q.ARMS[0])
        r=pool();r.append(dict(r[0],condition_id='second_C'))
        with self.assertRaises(ValueError):Q.make_schedule(r,Q.ARMS[0])

    def test_global_and_native_accounting(self):
        r=contract();a=M.cursor(r,0);b=M.cursor(r,1500)
        self.assertEqual((a['global_step'],a['native_total_step']),(19500,7500))
        self.assertEqual((b['global_step'],b['native_total_step'],b['seam_total_step']),(21000,9000,0))
        with self.assertRaises(ValueError):M.cursor(r,1501)
        with self.assertRaises(ValueError):M.stage_spec(dict(r,updates=3000))

    def test_exact_boundary_resume_and_wrong_arm(self):
        r=contract()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'calls.jsonl'
            a=M.T.historical.CallLedger(p,M.limits(r),Q.ARMS[0],'first',synchronize=False)
            M.require_cursor(a,r,0);update(a,r,1);M.require_cursor(a,r,1)
            b=M.T.historical.CallLedger(p,M.limits(r),Q.ARMS[0],'resume',synchronize=False)
            M.require_cursor(b,r,1);update(b,r,2);M.require_cursor(b,r,2)
            with self.assertRaises(RuntimeError):M.require_cursor(b,r,1)
            with self.assertRaises(ValueError):M.T.historical.CallLedger(p,M.limits(r),Q.ARMS[1],'wrong',synchronize=False)

    def test_partial_failed_and_extra_calls_prohibit_resume(self):
        r=contract()
        with tempfile.TemporaryDirectory() as temp:
            a=M.T.historical.CallLedger(Path(temp)/'calls',M.limits(r),Q.ARMS[0],'partial',synchronize=False)
            a.call('train_forward',lambda:None,dict(**M.cursor(r,1),microbatch=0))
            with self.assertRaises(RuntimeError):M.require_cursor(a,r,0)
            with self.assertRaises(RuntimeError):a.call('head_forward',lambda:None,{})
        with tempfile.TemporaryDirectory() as temp:
            a=M.T.historical.CallLedger(Path(temp)/'calls',M.limits(r),Q.ARMS[0],'failed',synchronize=False)
            def fail():raise RuntimeError('synthetic failure, not a neural call')
            with self.assertRaises(RuntimeError):a.call('train_forward',fail,dict(**M.cursor(r,1),microbatch=0))
            with self.assertRaises(RuntimeError):M.require_cursor(a,r,0)

    def test_inherited_adamw_roundtrip_no_update(self):
        class ParametersOnly(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(2));self.readout_mode='none'
        model=ParametersOnly().eval();o=make_optimizer(model,None)
        o.state[model.weight]=dict(step=torch.tensor(19500.),exp_avg=torch.tensor([.1,.2]),exp_avg_sq=torch.tensor([.3,.4]))
        saved=copy.deepcopy(o.state_dict());clone=make_optimizer(model,None);clone.load_state_dict(saved)
        M.validate_optimizer(model,clone,19500)
        self.assertEqual(M.T.base.tree_hash(saved),M.T.base.tree_hash(clone.state_dict()))
        with self.assertRaises(ValueError):M.validate_optimizer(model,clone,0)
        with self.assertRaises(ValueError):M.validate_optimizer(model,make_optimizer(model,None),19500)

    def test_no_new_geometric_or_network_objective(self):
        source=Path(M.__file__).read_text()
        calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)]
        losses=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='corner_masked_fm_loss']
        self.assertEqual(len(losses),1)
        self.assertIn("D.sample_for_training(cases[r['condition_id']],r,'PLAIN')",source)
        self.assertIn("_build_restore(p,BASE_STEP+step,'cuda')",source)


def main():
    if torch.cuda.is_initialized():raise RuntimeError('CPU gate cannot own CUDA')
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not result.wasSuccessful():raise SystemExit(1)
    source_hash=M.digest(M.SOURCE)
    if source_hash!=M.SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('Coverage source bytes changed')
    reg=dict(contract(),**M.SOURCE_IDENTITIES,source_checkpoint=str(M.SOURCE))
    payload=M._source_payload(reg)
    optimizer=payload['optimizer'];groups=optimizer['param_groups']
    if len(groups)!=1 or any(float(s['step'])!=19500 for s in optimizer['state'].values()):raise ValueError('Wrong inherited AdamW steps')
    if not all(k in payload['rng'] for k in ('python','numpy','torch_cpu','torch_cuda')):raise ValueError('Incomplete RNG')
    full_load=dict(schema=payload['schema'],global_step=payload['global_step'],optimizer_states=len(optimizer['state']),
        optimizer_group_count=len(groups),optimizer_step=19500,generator_state_sha256=M.state_hash(payload['generator_state']))
    del payload,optimizer
    old=M.T.read(M.SOURCE_ROOT/'data/manifest.json')
    rows=old['train']; actual=[]
    first={}
    schedule=Q.make_schedule(rows,Q.ARMS[0])
    for entry in schedule['steps']:
        for r in entry['samples']:
            first.setdefault((r['condition_id'],r['augmentation']),r)
    for r in rows:
        case=M.D.load_case(r)
        for variant in ('CLEAN','X_WIDTH'):
            sample=M.D.sample_for_training(case,first[r['condition_id'],variant],'PLAIN')
            audit=M.verify_sample(sample,'NATIVE')
            if not 128<=int(sample['N'])<=256:raise ValueError('Frozen face budget changed')
            actual.append(dict(condition_id=r['condition_id'],augmentation=variant,N=int(sample['N']),K=int(sample['K']),audit=audit))
    if len(actual)!=48:raise ValueError('Expected real24C identity/width CPU checks')
    if torch.cuda.is_initialized():raise RuntimeError('CPU tests initialized CUDA')
    report=dict(status='PASS',schema='bench_exposure_allocation_training_cpu_gate_v1',tests_run=result.testsRun,
        retained_checkpoint_tensor_validation=full_load,source_identities=M.SOURCE_IDENTITIES,
        actual_cpu_samples=actual,OT_calls=48,CPU_only=True,CUDA_initialized=False,
        neural_model_constructions=0,model_forwards=0,backwards=0,optimizer_updates=0,
        parameter_only_optimizer_fixture=True,
        source_hashes={str(Path(__file__).with_name(f)):M.digest(Path(__file__).with_name(f)) for f in ('training.py','stream.py','train_tests.py')})
    M.T.write(M.OUT/'training_cpu_gate.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('actual_cpu_samples','source_hashes')},indent=2))


if __name__=='__main__':main()
