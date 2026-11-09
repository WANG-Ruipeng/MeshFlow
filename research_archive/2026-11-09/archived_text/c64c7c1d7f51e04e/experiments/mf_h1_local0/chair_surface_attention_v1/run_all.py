"""Recoverable finite gate/audit controller; frozen failures cannot trigger training."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import *
from .registration import prepare,verify_identity,code_snapshot

MODULE='experiments.mf_h1_local0.chair_surface_attention_v1'


def execute(out,key,args):
    path=out/'controller'/f'{key}.json'
    if path.exists():
        old=read_json(path)
        if old['status']=='DONE':return old
        raise RuntimeError('Prior attempted stage is not complete; automatic replay prohibited: '+key)
    log=out/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
    state=dict(stage=key,status='RUNNING',started_unix=time.time(),command=[sys.executable,'-B','-m',*args],log=str(log))
    atomic_json(path,state)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if key.startswith('score'):env['CUDA_VISIBLE_DEVICES']=''
    with log.open('a',buffering=1) as f:
        p=subprocess.Popen(state['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
        state['pid']=p.pid;atomic_json(path,state);rc=p.wait()
    state.update(status='DONE' if rc==0 else 'BLOCKED',returncode=rc,finished_unix=time.time())
    atomic_json(path,state);return state


def finalize(out):
    from .ledger import Budget
    from .reporting import finalize as report
    from .delivery import finish
    numeric=read_json(out/'tests/relations/relations_gate_summary.json')
    synthetic=read_json(out/'tests/relations/relations_synthetic.json')
    attention=read_json(out/'tests/attention/results.json')
    real_stage=read_json(out/'stages/precheck.json') if (out/'stages/precheck.json').exists() else dict(status='NOT_RUN')
    real=real_stage.get('result',{})
    # This controller seals the actually observed finite failure. It does not
    # invent downstream checkpoints or permit a changed result to resume training.
    blocked=numeric['status']!='PASS' or synthetic['status']!='PASS' or attention['status']!='PASS' or real.get('status')!='PASS'
    if not blocked:raise RuntimeError('No gate failure found; inspect actual stage evidence before a scientific continuation')
    status='IMPLEMENTATION_BLOCKED' if numeric['status']!='PASS' or real.get('status')!='COST_BLOCKED' else 'COST_BLOCKED'
    gate=dict(status=status,formal_training_allowed=False,
        numeric=dict(status=numeric['status'],summary_path='tests/relations/relations_gate_summary.json',
            maxima=numeric['aggregate_maxima'],ordinary_threshold=.01,reference_threshold=.001,
            accuracy_pass_fixtures=sum(f['accuracy_pass'] for f in numeric['fixtures']),reference_pass_fixtures=sum(f['reference_convergence_pass'] for f in numeric['fixtures']),
            reference_scope='Fixed subsets only; one subset not converged, no certified true-integral bound'),
        synthetic=dict(status=synthetic['status'],path='tests/relations/relations_synthetic.json'),
        attention=dict(status=attention['status'],path='tests/attention/results.json'),
        real_model=dict(stage_status=real_stage['status'],result_status=real.get('status','NOT_RUN'),path='stages/precheck.json',actual_forwards=real.get('actual_forwards',0),actual_backwards=real.get('actual_zero_update_backwards',0),errors=real.get('errors',[]),comparisons=real.get('comparisons',[])),
        cost=dict(real.get('cost',dict(status='NOT_RUN')),path='precheck/result.json'),
        continuation_after_numeric_failure='Only the already-budgeted independent real-model wiring/cost audit; no formal training/sampling',
        changes_to_nodes_bandwidth_data_thresholds=0,extra_I_P_model_calls=0)
    atomic_json(out/'gate_results.json',gate)
    for arm in ARMS:
        atomic_json(out/'stages'/f'train_{arm}.json',dict(status='NOT_RUN',reason=status,optimizer_updates=0,surface_step=0))
        atomic_json(out/'stages'/f'eval_{arm}.json',dict(status='NOT_RUN',reason='No trained endpoint because gate failed',sampled=0))
    atomic_json(out/'trajectory_quadrature_results.json',dict(status='NOT_RUN',reason='No formal SURFACE trajectory because gate failed',extra_model_calls=0))
    counts=Budget(out,'finalize').snapshot()
    counts['isolated_CPU_tests']=dict(module_optimizer_updates=attention['isolated_module_optimizer_updates'],
        module_backwards=attention['isolated_module_backward_calls'],relation_autograd_checks=2,
        attention_status=attention['status'],relations_synthetic_status=synthetic['status'],
        relation_evaluations_synthetic=synthetic.get('relation_evaluations'),relation_evaluations_real=numeric['relation_evaluations'])
    counts['summary']['isolated_module_test_updates']=attention['isolated_module_optimizer_updates']
    atomic_json(out/'accounting.json',counts)
    verification=verify_identity(out);atomic_json(out/'registration/source_verification_after.json',verification)
    report(out);finish(out)
    atomic_json(out/'RUN_STOPPED.json',dict(status=status,stopped_unix=time.time(),protocol_sha256=digest(out/'protocol.json'),
        formal_optimizer_updates=0,trained='NOT_RUN',new_endpoint_evaluation='NOT_RUN',START='READ_ONLY_CACHE_SCORED',
        stop_reason='Registered numerical gate failed; fixed permitted audit finished; no budget extension',automatic_resume_training=False))
    print('FIXED_TASK_STOPPED '+status,flush=True)
    return gate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--through',choices=('prepare','precheck','all'),default='all');a=ap.parse_args();out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(out)
        freeze_json(out/'registration/evaluation_requests.json',[dict(r,checkpoint_label=label) for label in LABELS for r in records()])
        if a.through=='prepare':return
        # Already executed component tests have independent frozen source hashes;
        # never rerun failed numerical fixtures or an isolated test update.
        for path in ('tests/relations/relations_gate_summary.json','tests/relations/relations_synthetic.json','tests/attention/results.json'):
            if not (out/path).exists():raise RuntimeError('Required durable CPU gate result unavailable: '+path)
        assert read_json(out/'tests/relations/relations_synthetic.json')['status']=='PASS'
        assert read_json(out/'tests/attention/results.json')['status']=='PASS'
        if not (out/'registration/implementation_hashes.json').exists():code_snapshot(out)
        verify_identity(out)
        if not (out/'RUN_STOPPED.json').exists():
            execute(out,'precheck',[MODULE+'.worker','--out',str(out)])
        if a.through=='precheck':return
        from .cache_start import run as cache_start
        cache_start(out)
        if not (out/'evaluation/START/scoring/summary.json').exists():
            state=execute(out,'score_START',[MODULE+'.scoring','--out',str(out),'--label','START','--generated',str(out/'evaluation/START/generated_rows.json')])
            if state['status']!='DONE':raise RuntimeError('CPU source scoring incomplete; preserve evidence')
        finalize(out)


if __name__=='__main__':main()
