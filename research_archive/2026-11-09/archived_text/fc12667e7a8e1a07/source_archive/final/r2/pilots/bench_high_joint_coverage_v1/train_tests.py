"""Bounded CPU validation: original B state restore and one real max-N OT.

No neural forward, backward, optimizer update, GPU initialization, or synthetic
training. Synthetic call-ledger fixtures perform only Python no-op functions.
"""
from __future__ import annotations
import ast,copy,gc,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from . import common as C,runtime as R,stream as Q,training as M


def contract():
    return dict(updates=1500,source_global_step=21000,final_global_step=22500,
                full_steps=[1500],checkpoint_every=250)


def pool():return C.read(C.OUT/'data/manifest.json')['train']


def fixture_update(ledger,reg,step):
    for mi in range(8):
        for kind in ('train_forward','train_backward'):
            ledger.call(kind,lambda:None,dict(arm=ledger.arm,**M.cursor(reg,step),microbatch=mi))
    ledger.call('optimizer_update',lambda:None,dict(arm=ledger.arm,**M.cursor(reg,step)))


class ContractTests(unittest.TestCase):
    def test_budget_and_source_cursor(self):
        reg=contract()
        self.assertEqual((M.cursor(reg,0)['global_step'],M.cursor(reg,0)['native_total_step']),(21000,9000))
        self.assertEqual((M.cursor(reg,1500)['global_step'],M.cursor(reg,1500)['native_total_step']),(22500,10500))
        self.assertEqual(M.limits(reg)['train_forward'],12000)
        for value in (-1,1501):
            with self.assertRaises(ValueError):M.cursor(reg,value)
        with self.assertRaises(ValueError):M.stage_spec(dict(reg,source_global_step=22500))
        with self.assertRaises(ValueError):M.stage_spec(dict(reg,updates=3000))

    def test_exact_paired_stream_and_mutations(self):
        rows=pool();s=Q.make_schedule(rows,'WIDE_HIGH')
        self.assertEqual(Q.validate_frozen_narrow(s,rows)['status'],'PASS')
        for key,value in (('replay_group','HIGH'),('noise_seed',-1),('alpha',1.2),('parent_id','UNKNOWN')):
            bad=copy.deepcopy(s);bad['steps'][0]['samples'][0][key]=value
            with self.assertRaises(ValueError):Q.validate(bad,rows)
        for arm in ('CONTROL_OLD','NATURAL_HIGH'):
            with self.assertRaises(ValueError):Q.make_schedule(rows,arm)
        with self.assertRaises(ValueError):Q.make_schedule(rows,'WIDE_HIGH',3000)
        with self.assertRaises(ValueError):Q.groups_for(rows+[dict(rows[0],condition_id='DUPLICATE_PARENT')])

    def test_preserves_global_rng(self):
        n=np.random.get_state();t=torch.get_rng_state().clone()
        Q.make_schedule(pool(),'WIDE_HIGH');now=np.random.get_state()
        self.assertEqual(n[0],now[0]);self.assertTrue(np.array_equal(n[1],now[1]));self.assertEqual(n[2:],now[2:])
        self.assertTrue(torch.equal(t,torch.get_rng_state()))

    def test_exact_resume_and_no_replay(self):
        reg=contract()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture_calls.jsonl'
            a=M.T.historical.CallLedger(path,M.limits(reg),'WIDE_HIGH','fixture',synchronize=False)
            M.require_cursor(a,reg,0);fixture_update(a,reg,1);M.require_cursor(a,reg,1)
            b=M.T.historical.CallLedger(path,M.limits(reg),'WIDE_HIGH','resume_fixture',synchronize=False)
            M.require_cursor(b,reg,1);fixture_update(b,reg,2);M.require_cursor(b,reg,2)
            with self.assertRaises(RuntimeError):M.require_cursor(b,reg,1)
            with self.assertRaises(ValueError):M.T.historical.CallLedger(path,M.limits(reg),'NATURAL_HIGH','wrong',synchronize=False)
        with tempfile.TemporaryDirectory() as temp:
            a=M.T.historical.CallLedger(Path(temp)/'fixture_partial',M.limits(reg),'WIDE_HIGH','partial',synchronize=False)
            a.call('train_forward',lambda:None,dict(**M.cursor(reg,1),microbatch=0))
            with self.assertRaises(RuntimeError):M.require_cursor(a,reg,0)
            with self.assertRaises(RuntimeError):a.call('head_forward',lambda:None,{})

    def test_original_objective_unchanged(self):
        text=Path(M.__file__).read_text();tree=ast.parse(text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='corner_masked_fm_loss']
        self.assertEqual(len(calls),1)
        self.assertIn("denominator=sum(int(s['free_coordinate_count']) for s in samples)",text)
        self.assertIn("sample_for_training(cases[r['condition_id']],r,'PLAIN')",text)
        prior=Path(M.__file__).parents[1]/'bench_natural_n_recipe_v1/training.py'
        def body(path,name):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('_source_payload','_build_restore','require_cursor','limits','cursor','validate_payload'):
            self.assertEqual(ast.dump(body(Path(M.__file__),name)),ast.dump(body(prior,name)))


def main():
    names=('common.py','runtime.py','stream.py','training.py','train_tests.py','prepare.py','register.py')
    hashes={str(Path(__file__).with_name(n)):C.sha(Path(__file__).with_name(n)) for n in names}
    dest=C.OUT/'training_cpu_gate.json'
    if dest.exists():
        saved=C.read(dest)
        if saved['source_hashes']!=hashes or saved['manifest_sha256']!=C.sha(C.OUT/'data/manifest.json'):
            raise ValueError('Completed bounded gate changed; no silent repeat')
        return saved
    ledger=C.OUT/'cpu_ot_calls.jsonl'
    if ledger.exists():raise RuntimeError('Prior CPU OT attempt exists; no automatic replay')
    if torch.cuda.is_initialized():raise RuntimeError('CPU gate must not initialize CUDA')
    torch.set_num_threads(2);before=C.resources();runtime_receipt=R.cpu_gate()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not tests.wasSuccessful():raise SystemExit(1)
    if C.sha(C.SOURCE)!=C.SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('Original B source changed')
    payload=M._source_payload(dict(contract(),**C.SOURCE_IDENTITIES,source_checkpoint=str(C.SOURCE)))
    model,optimizer=M._build_restore(payload,21000,'cpu')
    if len(optimizer.param_groups)!=1 or len(optimizer.state)!=194 or any(float(s['step'])!=21000 for s in optimizer.state.values()):
        raise ValueError('Original B optimizer grouping/steps differ')
    process_rng=M.rng_state();cpu_rng=copy.deepcopy(payload['rng']);cpu_rng['torch_cuda']=[]
    M.restore_rng(cpu_rng)
    if M.T.base.tree_hash(M.rng_state())!=M.T.base.tree_hash(cpu_rng):raise ValueError('CPU source RNG restore differs')
    M.restore_rng(process_rng)
    restoration=dict(schema=payload['schema'],global_step=21000,native_total_step=9000,
        generator_strict_load=True,optimizer_tensor_and_parameter_order_exact=True,optimizer_groups=1,
        optimizer_states=194,optimizer_step=21000,CPU_RNG_restore_exact=True,
        saved_CUDA_RNG_arrays_authenticated_not_restored=len(payload['rng']['torch_cuda']))
    del model,optimizer,payload;gc.collect()
    manifest=C.read(C.OUT/'data/manifest.json');Q.validate_manifest(manifest)
    schedule=C.read(C.OUT/'data/schedule_WIDE_HIGH.json');Q.validate(schedule,manifest['train'])
    maximum=max(manifest['train'],key=lambda r:int(r['N']))
    slot=next(r for e in schedule['steps'] for r in e['samples'] if r['condition_id']==maximum['condition_id'] and r['augmentation']=='CLEAN')
    case=M.D.load_case(maximum)
    M.append_event(ledger,dict(event='CPU_OT_BEGIN',index=1,budget=1,registered_stream_row=slot,condition_id=maximum['condition_id'],N=int(maximum['N'])))
    sample=R.sample_for_training(case,slot,'PLAIN');audit=R.verify_sample(sample)
    if int(sample['production_OT_calls'])!=1:raise ValueError('Exactly one real CPU OT packing required')
    if int(sample['N'])!=int(maximum['N']) or int(sample['free_coordinate_count'])!=9*(int(maximum['N'])-int(maximum['K'])):
        raise ValueError('Maximum real-N sample mask/free-coordinate contract differs')
    M.append_event(ledger,dict(event='CPU_OT_COMPLETE',index=1,OT_calls=1,N=int(sample['N']),K=int(sample['K'])))
    if torch.cuda.is_initialized():raise RuntimeError('CPU gate initialized CUDA')
    report=dict(status='PASS',schema='bench_high_joint_training_cpu_gate_v1',source_identities=C.SOURCE_IDENTITIES,
        source_hashes=hashes,retained_checkpoint_tensor_validation=restoration,tests_run=tests.testsRun,
        runtime_cpu_gate=str(C.OUT/'runtime_cpu_gate.json'),runtime_cpu_gate_sha256=C.sha(C.OUT/'runtime_cpu_gate.json'),
        manifest_sha256=C.sha(C.OUT/'data/manifest.json'),actual_cpu_samples=[dict(condition_id=maximum['condition_id'],
            N=int(sample['N']),K=int(sample['K']),audit=audit,source_case_sha256=maximum['case_sha256'],OT_calls=1)],
        OT_calls=1,OT_attempts=1,CPU_OT_ledger_sha256=C.sha(ledger),model_forwards=0,model_backwards=0,
        optimizer_updates=0,CUDA_initialized=False,CPU_only=True,CPU_threads=2,resources_before=before,
        resources_after=C.resources(),neural_model_constructions=1,synthetic_ledger_fixtures_not_scientific_calls=True)
    C.write_new(dest,report);print(json.dumps({k:report[k] for k in ('status','tests_run','OT_calls','model_forwards')}));return report


if __name__=='__main__':main()
