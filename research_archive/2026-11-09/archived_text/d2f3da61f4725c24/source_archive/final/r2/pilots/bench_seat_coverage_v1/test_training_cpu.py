"""CPU schedule/recovery/role/AdamW tests, without loading any large model.

The retained source is checked by bytes and sidecar only here. Actual tensor
hashes and moment restoration are enforced before the first formal update.
"""
from __future__ import annotations
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
import torch
from meshflow_control.training.schedule40 import make_optimizer
from . import train as M
from . import training_state as S
from . import schedule as Q
from .common import OUT,SOURCE,SOURCE_IDENTITIES,read,sha


def contract(updates=2):
    return dict(updates=updates,full_steps=[0,updates],source_global_step=18000,final_global_step=18000+updates)


def pool():
    return [dict(condition_id=g+str(p)+'_'+str(c),parent_id=g+str(p),replay_group=g)
            for g in ('OLD','NEW') for p in range(3 if g=='OLD' else 2) for c in range(2)]


def update(ledger,reg,step):
    for micro in range(8):
        ctx=dict(**S.cursor(reg,step),microbatch=micro)
        for kind in ('train_forward','train_backward'):ledger.call(kind,lambda:None,ctx)
    ledger.call('optimizer_update',lambda:None,S.cursor(reg,step))


class Tests(unittest.TestCase):
    def test_new_branch_cursor_and_budget(self):
        r=contract();self.assertEqual(S.cursor(r,0)['global_step'],18000)
        self.assertEqual(S.cursor(r,2)['global_step'],18002)
        self.assertEqual(S.cursor(r,2)['native_total_step'],6002)
        with self.assertRaises(ValueError):S.cursor(r,3)
        r['full_steps']=[0,1]
        with self.assertRaises(ValueError):S.stage_spec(r)

    def test_schedule_reproducible_and_parent_balanced(self):
        rows=pool();a=Q.make_schedule(rows,90)
        self.assertEqual(a,Q.make_schedule(rows,90))
        self.assertEqual(Q.validate(a,rows,90)['exposures'],720)
        for g in ('OLD','NEW'):
            counts=[a['condition_exposure'][r['condition_id']] for r in rows if r['replay_group']==g]
            self.assertLessEqual(max(counts)-min(counts),2)

    def test_schedule_rejects_wrong_quota_identity_seed(self):
        rows=pool();a=Q.make_schedule(rows,20)
        for key,value in (('replay_group','NEW'),('noise_seed',-1),('alpha',1.5)):
            bad=copy.deepcopy(a);bad['steps'][0]['samples'][0][key]=value
            with self.assertRaises(ValueError):Q.validate(bad,rows,20)

    def test_distinct_parent_groups_and_minimum(self):
        rows=pool();rows[0]['parent_id']='NEW0'
        with self.assertRaises(ValueError):Q.groups_for(rows)
        with self.assertRaises(ValueError):Q.groups_for([dict(condition_id='x',parent_id='p',replay_group='NEW')])

    def test_exact_recovery_rejects_stale_and_extra_calls(self):
        r=contract()
        with tempfile.TemporaryDirectory(dir=OUT/'cpu_tests') as tmp:
            ledger=M.T.historical.CallLedger(Path(tmp)/'calls.jsonl',S.limits(r),'NATIVE','synthetic',synchronize=False)
            S.require_cursor(ledger,r,0);update(ledger,r,1);S.require_cursor(ledger,r,1)
            with self.assertRaises(RuntimeError):S.require_cursor(ledger,r,0)
            with self.assertRaises(RuntimeError):ledger.call('head_forward',lambda:None,{})
            restored=M.T.historical.CallLedger(ledger.path,S.limits(r),'NATIVE','restore',synchronize=False)
            S.require_cursor(restored,r,1);update(restored,r,2);S.require_cursor(restored,r,2)
            with self.assertRaises(RuntimeError):restored.call('optimizer_update',lambda:None,{})

    def test_partial_update_and_wrong_global_rejected(self):
        r=contract()
        with tempfile.TemporaryDirectory(dir=OUT/'cpu_tests') as tmp:
            ledger=M.T.historical.CallLedger(Path(tmp)/'calls.jsonl',S.limits(r),'NATIVE','synthetic',synchronize=False)
            ledger.call('train_forward',lambda:None,dict(**S.cursor(r,1),microbatch=0))
            with self.assertRaises(RuntimeError):S.require_cursor(ledger,r,0)
        with tempfile.TemporaryDirectory(dir=OUT/'cpu_tests') as tmp:
            ledger=M.T.historical.CallLedger(Path(tmp)/'calls.jsonl',S.limits(r),'NATIVE','synthetic',synchronize=False)
            update(ledger,r,1);ledger.rows[0]['context']['global_step']=12001
            with self.assertRaises(RuntimeError):S.require_cursor(ledger,r,1)

    def test_native_owner_metadata_does_not_fix_free(self):
        n,k=128,2;anchor=np.full((n,3),-1,np.int64);anchor[:k]=np.arange(6).reshape(2,3)
        known=np.arange(n)<k
        s=dict(N=n,K=k,anchor_ids=anchor,known_mask=known,valid_mask=np.ones(n,bool),
            occupied_labels=np.asarray([True,False]),occupied_for_FM=np.zeros(2,bool),free_coordinate_count=9*(n-k))
        self.assertEqual(S.verify_sample(s,'NATIVE')['active_seam_slots'],0)
        s['anchor_ids'][k,:2]=[0,1]
        with self.assertRaises(ValueError):S.verify_sample(s,'NATIVE')

    def test_inherited_single_group_adamw_without_forward(self):
        class ParametersOnly(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(2));self.readout_mode='none'
        model=ParametersOnly().eval();optimizer=make_optimizer(model,None)
        optimizer.state[model.weight]=dict(step=torch.tensor(18000.),exp_avg=torch.tensor([.1,.2]),exp_avg_sq=torch.tensor([.3,.4]))
        saved=optimizer.state_dict();clone=make_optimizer(model,None);clone.load_state_dict(saved)
        S.validate_optimizer(model,clone,18000)
        self.assertEqual(M.T.base.tree_hash(saved),M.T.base.tree_hash(clone.state_dict()))
        with self.assertRaises(ValueError):S.validate_optimizer(model,clone,0)
        with self.assertRaises(ValueError):S.validate_optimizer(model,make_optimizer(model,None),18000)


def main():
    if torch.cuda.is_initialized():raise RuntimeError('CPU check must start without CUDA')
    (OUT/'cpu_tests').mkdir(parents=True,exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)
    if sha(SOURCE)!=SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('Retained source file bytes changed')
    receipt=read(SOURCE.with_suffix('.json'))
    for key,hkey in (('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')):
        if receipt[key]!=SOURCE_IDENTITIES[hkey]:raise ValueError('Source sidecar differs')
    if torch.cuda.is_initialized():raise RuntimeError('CPU tests initialized CUDA')
    report=dict(status='PASS',schema='bench_seat_coverage_training_cpu_gate_v1',tests_run=result.testsRun,
        source_identity=SOURCE_IDENTITIES,source_global_step=receipt['global_step'],CPU_only=True,
        CUDA_initialized=False,large_model_loads=0,model_forwards=0,backwards=0,optimizer_updates=0,OT_calls=0,
        real_source_tensor_load='DEFERRED: mandatory hashes/restoration checks before first formal update',
        source_hashes={str(Path(__file__).with_name(f)):sha(Path(__file__).with_name(f)) for f in
            ('common.py','train.py','training_state.py','schedule.py','test_training_cpu.py')})
    M.T.write(OUT/'training_cpu_gate.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
