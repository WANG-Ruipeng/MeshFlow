"""One foreground controller for all authorized stages, with durable identities."""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var]='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import difflib
import json
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time
import traceback
from .common import REPO, LIMITS, GPU_LOCK, atomic_json, read_json, digest, json_hash, freeze_json, append_event, controller_lock

MODULE='experiments.mf_h1_local0.control_mvp48_budget_single'

def stages():
    result=[]
    def add(key,module,args,deps=()):
        result.append(dict(key=key,module=module,args=args,dependencies=list(deps)))
    add('wiring','precheck',[])
    for model in ('Native0','T1_500','Geo_shared_500','T1_1000','Geo_1000','Full_1000'):
        add('curve_'+model,'evaluation',['--model-id',model,'--phase','curve'],['wiring'])
        if model in ('Native0','T1_1000','Geo_1000','Full_1000'):
            add('swap_'+model,'evaluation',['--model-id',model,'--phase','swap'],['curve_'+model])
    for step,prior in ((2000,None),(3500,2000),(5000,3500)):
        for arm in ('T1','Geo','Full'):
            model=f'{arm}_{step}'
            deps=['wiring'] if prior is None else [f'train_{arm}_{prior}']
            add('train_'+model,'training',['--arm',arm,'--target',str(step)],deps)
            add('curve_'+model,'evaluation',['--model-id',model,'--phase','curve'],['train_'+model])
            if step==5000:
                add('swap_'+model,'evaluation',['--model-id',model,'--phase','swap'],['curve_'+model])
    add('fit_T1_1000','evaluation',['--model-id','T1_1000','--phase','fit'],['wiring'])
    for step in (500,1000):
        model=f'T1_fit2_probe_{step}'
        add('train_'+model,'training',['--arm','T1_fit2_probe','--target',str(step)],
            ['wiring'] if step==500 else ['train_T1_fit2_probe_500'])
        add('fit_'+model,'evaluation',['--model-id',model,'--phase','fit'],['train_'+model])
    for arm in ('T1','Geo','Full'):
        model=f'{arm}_5000'
        add('holdout_'+model,'evaluation',['--model-id',model,'--phase','holdout'],
            ['train_T1_5000','train_Geo_5000','train_Full_5000'])
    return result

def code_snapshot(out,verify=False):
    own=Path(__file__).parent
    files=sorted(own.glob('*.py'))
    hashes={str(p.relative_to(REPO)):digest(p) for p in files}
    path=out/'registration/code_hashes.json'
    if verify:
        if read_json(path)!=hashes:
            raise RuntimeError('This run code changed after freezing; preserving state and stopping')
        return hashes
    freeze_json(path,hashes)
    chunks=[]
    for p in files:
        chunks.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(keepends=True),
            fromfile='/dev/null',tofile='b/'+str(p.relative_to(REPO))))
    (out/'code.diff').write_text(''.join(chunks),encoding='utf-8')
    return hashes

def frozen_protocol(source,out,gate):
    from .prepare import prepare
    prepared=prepare(source,out)
    plan_files={name:digest(out/name) for name in ('train_plan.json','fit_plan.json','panels.json')}
    old=read_json(source/'registration/train_protocol.json')['protocol']
    protocol=dict(schema='control_mvp48_budget_single_v1',source=str(source),out=str(out),
        source_data_revision='repair_v2',source_data_ready_sha256=gate['data_ready_sha256'],
        source_train_ready_sha256=gate['train_ready_sha256'],source_train_protocol=old,
        restore_mode='RESTORE_ALL',source_checkpoints={k:v['hashes'] for k,v in gate['endpoints'].items()},
        original_fork_sha256=gate['fork']['checkpoint_sha256'],frozen_files=plan_files,
        main_arms=['T1','Geo','Full'],start_step=1000,checkpoints=[1000,2000,3500,5000],
        main_updates_per_arm=4000,fit2_updates=1000,budgets=LIMITS,
        effective_batch=8,microbatch=1,optimizer=old['optimizer'],loss=old['loss'],
        full_ramp_index='logical_step_minus_500',sampler='Euler50',NFE=50,
        precision=old['precision'],determinism=old['determinism'],gpu_workers=1,gpu_lock=GPU_LOCK,
        source_plan_extension='new shared stateless seeds; original plan ends at 1000',
        historical_missing='NOT_AVAILABLE; no reconstruction by training',
        final_selection='all main arms at 5000',stages=stages(),
        repetitions_authorized_max=4,repetitions_planned=0,
        source_scorer_identity=read_json(source/'registration/eval_protocol.json')['scorer_identity'])
    freeze_json(out/'protocol.json',protocol)
    return protocol

def validate_frozen_inputs(source,out,protocol):
    for name,sha in protocol['frozen_files'].items():
        if digest(out/name)!=sha: raise ValueError('Frozen plan changed: '+name)
    for name,sha in protocol['source_scorer_identity'].items():
        if digest(REPO/name)!=sha: raise ValueError('Protected scorer changed: '+name)
    ready=read_json(source/'TRAIN_READY.json')
    for name,sha in ready['source_hashes'].items():
        if digest(REPO/name)!=sha: raise ValueError('Protected shared code changed: '+name)
    if digest(source/'DATA_READY.json')!=protocol['source_data_ready_sha256'] or digest(source/'TRAIN_READY.json')!=protocol['source_train_ready_sha256']:
        raise ValueError('Source readiness identity changed')

def worker_status(out,stage):
    out=Path(out)
    if stage['module']=='evaluation':
        args=stage['args']; model=args[args.index('--model-id')+1]; phase=args[args.index('--phase')+1]
        return read_json(out/f'evaluation/{phase}/{model}/ledger.json')
    if stage['module']=='precheck': return read_json(out/'registration/wiring_precheck.json')
    if stage['module']!='training': raise ValueError('Unknown managed worker')
    args=stage['args'];arm=args[args.index('--arm')+1];target=int(args[args.index('--target')+1])
    summary=out/'training'/f'{arm}_to{target}_summary.json'
    status_path=summary if summary.exists() else out/'training'/f'{arm}_status.json'
    if not status_path.exists():
        return dict(status='STOPPED',reason='Training worker has no persisted completion record')
    actual=read_json(status_path)
    if actual.get('status')!='SEGMENT_COMPLETE' or int(actual.get('logical_step',-1))<target:
        return dict(actual,status='STOPPED',worker_status=actual.get('status'),reason='Training target not completed')
    if actual.get('arm')!=arm:
        raise ValueError('Training summary arm differs')
    checkpoint=out/'checkpoints'/f'{arm}_step{target}.pt'
    sidecar=checkpoint.with_suffix('.json')
    if not checkpoint.is_file() or not sidecar.is_file():
        return dict(status='STOPPED',reason='Fixed checkpoint or identity sidecar missing')
    record=read_json(sidecar);metadata=record['metadata']
    sha=digest(checkpoint)
    if record['checkpoint_sha256']!=sha:
        raise ValueError('Fixed checkpoint SHA differs: '+str(checkpoint))
    if metadata['variant']!=arm or int(metadata['logical_step'])!=target:
        raise ValueError('Fixed checkpoint arm/step differs')
    protocol_hash=json_hash(read_json(out/'protocol.json'))
    if metadata['protocol_sha256']!=protocol_hash or actual.get('protocol_sha256')!=protocol_hash:
        raise ValueError('Fixed checkpoint/summary protocol identity differs')
    if Path(record['path']).resolve()!=checkpoint.resolve():
        raise ValueError('Fixed checkpoint sidecar path differs')
    return dict(actual,status='PASS',worker_status='SEGMENT_COMPLETE',verified_checkpoint_sha256=sha,
                verified_checkpoint=str(checkpoint),checkpoint_metadata=metadata)


def orphan_budget_workers(out, process_table=None):
    """Report live workers for this output; never terminate someone else's process."""
    if process_table is None:
        process_table=subprocess.run(['ps','-eo','pid,ppid,args'],capture_output=True,text=True,check=True).stdout
    expected=Path(out).resolve(); found=[]
    for line in process_table.splitlines()[1:]:
        fields=line.strip().split(None,2)
        if len(fields)!=3: continue
        try: pid,ppid=int(fields[0]),int(fields[1]);argv=shlex.split(fields[2])
        except (ValueError,TypeError): continue
        if pid==os.getpid() or '-m' not in argv: continue
        index=argv.index('-m')
        if index+1>=len(argv) or argv[index+1] not in {MODULE+'.'+x for x in ('training','evaluation','precheck')}:
            continue
        output=None
        if '--out' in argv:
            position=argv.index('--out')
            if position+1<len(argv): output=argv[position+1]
        else:
            output=next((x.split('=',1)[1] for x in argv if x.startswith('--out=')),None)
        if output is None or (REPO/Path(output)).resolve()==expected:
            found.append(dict(pid=pid,ppid=ppid,command=fields[2],reason='Live budget worker may own this output'))
    return found


def wait_managed_child(child):
    """On controller interruption, ask only its own child to save, then reap it."""
    try:
        return child.wait()
    except BaseException:
        if child.poll() is None:
            try:child.send_signal(signal.SIGTERM)
            except ProcessLookupError:pass  # Child exited between poll and signal.
            previous={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
            last_notice=time.monotonic()
            try:
                while child.poll() is None:
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if time.monotonic()-last_notice>=30:
                            print('Waiting for owned worker to save safe progress and exit.',flush=True)
                            last_notice=time.monotonic()
            finally:
                for sig,handler in previous.items():signal.signal(sig,handler)
        raise


def execution_budget_audit(out):
    """Verify actual cumulative endpoints and all requested evaluation denominators."""
    out=Path(out);errors=[]; arms={};total_updates=training_forwards=0
    registered=stages();expected_keys={x['key'] for x in registered}
    state=read_json(out/'run_state.json')
    if set(state['stages'])!=expected_keys:
        errors.append('Controller stage set is incomplete or has unregistered stages')
    for arm,step,want in (('T1',5000,4000),('Geo',5000,4000),('Full',5000,4000),('T1_fit2_probe',1000,1000)):
        try:
            stage=next(x for x in registered if x['key']==f'train_{arm}_{step}')
            actual=worker_status(out,stage)
            counts=actual.get('counts',{})
            updates=int(counts.get('optimizer_updates',-1));forwards=int(counts.get('forward_attempts',-1))
            if actual.get('status')!='PASS' or int(actual.get('logical_step',-1))!=step:
                errors.append(arm+' did not complete its exact fixed endpoint')
            if updates!=want or forwards!=updates*8:
                errors.append(arm+' optimizer/forward counts differ from fixed effective8 budget')
            for key in ('forward_returns','backward_attempts','backward_returns'):
                if int(counts.get(key,-1))!=forwards: errors.append(arm+' incomplete '+key)
            if int(counts.get('optimizer_step_attempts',-1))!=updates:
                errors.append(arm+' optimizer attempts differ from validated updates')
            meta=actual.get('checkpoint_metadata',{})
            if meta.get('counts')!=counts or int(meta.get('actual_new_updates',-1))!=want:
                errors.append(arm+' final checkpoint counts differ from completed worker')
            arms[arm]=dict(updates=updates,forward_attempts=forwards,status=actual.get('status'))
            total_updates+=max(0,updates);training_forwards+=max(0,forwards)
        except (OSError,ValueError,KeyError,StopIteration) as exc:
            errors.append(arm+': '+repr(exc))
    generations=teacher_forwards=sampling_calls=teacher_calls=0;eval_rows=[]
    sizes=dict(curve=(64,48),swap=(16,0),fit=(8,6),holdout=(32,0))
    for stage in registered:
        if stage['module']!='evaluation':continue
        args=stage['args'];phase=args[args.index('--phase')+1];model=args[args.index('--model-id')+1]
        try:
            data=worker_status(out,stage);sc=data.get('sampling_counts',{});tc=data.get('teacher_counts',{})
            complete=int(sc.get('complete_rollouts',0));teachers=int(tc.get('teacher_returns',0))
            gen_calls=int(sc.get('model_forward_attempts',0));t_calls=int(tc.get('model_forward_attempts',0))
            generations+=complete;teacher_forwards+=teachers;sampling_calls+=gen_calls;teacher_calls+=t_calls
            if data.get('status')=='NOT_AVAILABLE':
                if model not in ('Native0','T1_500','Geo_shared_500') or complete or teachers or gen_calls or t_calls:
                    errors.append(stage['key']+' has invalid historical NOT_AVAILABLE status')
            elif data.get('status')=='PASS':
                rows=data.get('rows',[]);trows=data.get('teacher_rows',[])
                if len(rows)!=sizes[phase][0] or len(trows)!=sizes[phase][1]:errors.append(stage['key']+' request denominator differs')
                if len({r['case_id'] for r in rows})!=len(rows) or len({r['case_id'] for r in trows})!=len(trows):
                    errors.append(stage['key']+' duplicate request IDs')
                allowed=('PASS','NOT_AVAILABLE') if phase=='swap' else ('PASS',)
                if any(r.get('status') not in allowed for r in rows) or any(r.get('status')!='PASS' for r in trows):
                    errors.append(stage['key']+' contains unfinished/failed requests')
                if complete!=sum(r.get('status')=='PASS' for r in rows) or teachers!=len(trows):
                    errors.append(stage['key']+' complete counters differ from actual rows')
                if int(sc.get('rollout_attempts',0))!=complete or int(tc.get('teacher_attempts',0))!=teachers:
                    errors.append(stage['key']+' attempts differ from completed work')
                if gen_calls!=complete*50 or t_calls!=teachers:
                    errors.append(stage['key']+' actual forward budget differs from Euler50/teacher definition')
                for key in ('model_forward_returns','backbone_forward_attempts','backbone_forward_returns'):
                    if int(sc.get(key,0))!=gen_calls or int(tc.get(key,0))!=t_calls:
                        errors.append(stage['key']+' incomplete '+key)
                if any(not r.get('score_path') for r in rows if r.get('status')=='PASS'):
                    errors.append(stage['key']+' raw generation missing its score')
            else:errors.append(stage['key']+' is not a completed evaluation')
            eval_rows.append(dict(stage=stage['key'],status=data.get('status'),complete_rollouts=complete,
                                 teacher_returns=teachers,sampling_forward_attempts=gen_calls,teacher_forward_attempts=t_calls))
        except (OSError,ValueError,KeyError) as exc:errors.append(stage['key']+': '+repr(exc))
    try:
        wiring=read_json(out/'registration/wiring_precheck.json')
        wiring_calls=int(wiring.get('forward_attempts',wiring.get('forwards_attempted',-1)))
        returned=int(wiring.get('forward_returns',wiring.get('forwards_returned',-1)))
        if wiring.get('status')!='PASS' or wiring_calls<0 or returned!=wiring_calls or wiring.get('optimizer_updates')!=0:
            errors.append('Wiring counters are incomplete or include an optimizer update')
    except (OSError,ValueError,KeyError) as exc:
        errors.append('wiring: '+repr(exc));wiring_calls=0
    total=training_forwards+sampling_calls+teacher_calls+wiring_calls
    if total_updates!=13000 or training_forwards!=104000:errors.append('Main/probe update total differs from 12000+1000')
    for actual,limit,name in ((generations,1192,'generations'),(teacher_forwards,738,'teacher'),(wiring_calls,80,'wiring'),(total,164618,'all model forwards')):
        if actual>limit:errors.append(name+' exceeds its hard budget')
    return dict(schema='control_mvp48_budget_execution_audit_v1',status='PASS' if not errors else 'FAIL',errors=errors,
                arms=arms,optimizer_updates=total_updates,training_forward_attempts=training_forwards,
                complete_rollouts=generations,teacher_returns=teacher_forwards,sampling_forward_attempts=sampling_calls,
                teacher_forward_attempts=teacher_calls,wiring_forward_attempts=wiring_calls,model_forward_attempts=total,
                limits=LIMITS,evaluation=eval_rows,
                planned_forward_ceiling=1192*50+104000+738+80,repeat_forward_reserve=4*50)


def bootstrap_report(out,state):
    # A failed preflight still has a truthful artifact; final reporting replaces this.
    lines=['# Control MVP48 All-in-One', '',
           'Status: '+state['status'], '',
           'This file is an execution status record, not evidence that training has completed.',
           'Actual task counts must be read from the durable worker ledgers; queued stages are not results.', '',
           'Fixed limits: main 12000 + fit2 1000 optimizer updates; 1192 planned generations; 738 teacher forwards.',
           'Protected source assets and public defaults are unchanged.', '',
           'Resume command:', '```bash',state['restart_command'],'```']
    if state.get('error'): lines.extend(['',state['error']])
    (out/'ALL_IN_ONE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run(source,out):
    source=source.resolve();out=out.resolve()
    if source==out or not out.is_relative_to(REPO/'native_t1/runs'):
        raise ValueError('Outputs must use an isolated run directory')
    for name in ('registration','logs','training','evaluation','checkpoints'):
        (out/name).mkdir(parents=True,exist_ok=True)
    with controller_lock(out):
        path=out/'run_state.json'
        state=read_json(path) if path.exists() else dict(schema='control_mvp48_budget_state_v1',status='PREFLIGHT',stages={},created_unix=time.time())
        state['restart_command']=f'cd /mnt/e/MeshFlow && {sys.executable} -B -m {MODULE}.run_all --source {source} --out {out}'
        state['pid']=os.getpid();state['last_started_unix']=time.time()
        atomic_json(path,state)
        bootstrap_report(out,state)
        try:
            from experiments.mf_h1_local0.control_mvp48.eval_checkpoint_gate import inspect_ready
            gate=inspect_ready(source,REPO)
            atomic_json(out/'registration/source_identity.json',gate)
            processes=subprocess.run(['ps','-eo','pid,ppid,args'],capture_output=True,text=True,check=True).stdout
            atomic_json(out/'registration/processes_before.json',dict(time_unix=time.time(),processes=processes,
                disk_free_bytes=shutil.disk_usage(out).free,duplicate_control='exclusive controller flock',gpu_workers=1))
            orphaned=orphan_budget_workers(out,processes)
            if orphaned: raise RuntimeError('Existing budget workers must exit before resuming: '+json.dumps(orphaned))
            if shutil.disk_usage(out).free < 25*1024**3: raise RuntimeError('Less than 25 GiB disk headroom; no assets will be deleted')
            protocol=frozen_protocol(source,out,gate)
            state['protocol_sha256']=json_hash(protocol)
            code_snapshot(out)
            validate_frozen_inputs(source,out,protocol)
            if state['status']=='COMPLETE':
                budget=execution_budget_audit(out);atomic_json(out/'registration/execution_budget_audit.json',budget)
                if budget['status']!='PASS':raise RuntimeError('Completed state failed actual budget audit: '+json.dumps(budget['errors']))
                print('Already complete with matching identities; zero additional work.',flush=True)
                return state
            state['status']='RUNNING';atomic_json(path,state)
            for stage in stages():
                key=stage['key'];old=state['stages'].get(key,{})
                if old.get('status') in ('PASS','NOT_AVAILABLE'):
                    actual=worker_status(out,stage)
                    if actual.get('status')!=old['status']:raise RuntimeError('Persisted completion no longer matches worker: '+key)
                    continue
                # A missing historical Native0 curve also makes its swap NA; let
                # the same model worker persist that NA instead of inventing a failure.
                acceptable=('PASS','NOT_AVAILABLE') if stage['module']=='evaluation' else ('PASS',)
                failures=[d for d in stage['dependencies'] if state['stages'].get(d,{}).get('status') not in acceptable]
                if failures:
                    state['stages'][key]=dict(status='BLOCKED_DEPENDENCY',dependencies=failures)
                    atomic_json(path,state);continue
                code_snapshot(out,verify=True);validate_frozen_inputs(source,out,protocol)
                orphaned=orphan_budget_workers(out)
                if orphaned:raise RuntimeError('Live budget worker blocks duplicate dispatch: '+json.dumps(orphaned))
                command=[sys.executable,'-B','-m',MODULE+'.'+stage['module'],'--source',str(source),'--out',str(out),*stage['args']]
                state['active_stage']=key;state['stages'][key]=dict(status='RUNNING',command=command,started_unix=time.time())
                atomic_json(path,state);append_event(out/'execution_ledger.jsonl',dict(event='STAGE_BEGIN',stage=key,command=command))
                print(json.dumps(dict(event='STAGE_BEGIN',stage=key,time_unix=time.time())),flush=True)
                tick=time.perf_counter()
                with (out/'logs'/f'{key}.log').open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        state['worker_pid']=child.pid;atomic_json(path,state)
                        rc=wait_managed_child(child)
                    except BaseException:
                        # Also cover an interruption while recording the newly owned PID.
                        if child.poll() is None:
                            try:child.send_signal(signal.SIGTERM)
                            except ProcessLookupError:pass
                            wait_managed_child(child)
                        raise
                    finally:
                        state['worker_pid']=child.pid if child.poll() is None else None
                        atomic_json(path,state)
                record=dict(status='PASS' if rc==0 else 'STOPPED',returncode=rc,elapsed_seconds=time.perf_counter()-tick,
                            completed_unix=time.time(),log=f'logs/{key}.log')
                if rc==0:
                    actual=worker_status(out,stage)
                    if actual is not None:
                        record['status']=actual.get('status','STOPPED')
                        if record['status'] not in ('PASS','NOT_AVAILABLE'):record['status']='STOPPED'
                state['stages'][key]=record;state['worker_pid']=None
                atomic_json(path,state);append_event(out/'execution_ledger.jsonl',dict(event='STAGE_END',stage=key,**record))
                print(json.dumps(dict(event='STAGE_END',stage=key,**record)),flush=True)
            code_snapshot(out,verify=True);validate_frozen_inputs(source,out,protocol)
            complete=set(state['stages'])=={s['key'] for s in stages()} and all(r['status'] in ('PASS','NOT_AVAILABLE') for r in state['stages'].values())
            if complete:
                budget=execution_budget_audit(out);atomic_json(out/'registration/execution_budget_audit.json',budget)
                if budget['status']!='PASS':raise RuntimeError('Final actual budget audit failed: '+json.dumps(budget['errors']))
            state['status']='COMPLETE' if complete else 'PARTIAL_STOPPED'
            state['active_stage']=None;state['finished_unix']=time.time();atomic_json(path,state)
        except BaseException as exc:
            state.update(status='STOPPED',error=repr(exc),traceback=traceback.format_exc())
            atomic_json(path,state);bootstrap_report(out,state)
            append_event(out/'execution_ledger.jsonl',dict(event='CONTROLLER_STOPPED',error=repr(exc)))
            raise
        finally:
            if (Path(__file__).parent/'reporting.py').exists():
                command=[sys.executable,'-B','-m',MODULE+'.reporting','--source',str(source),'--out',str(out)]
                with (out/'logs/reporting.log').open('a',encoding='utf-8') as log:
                    result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    state['reporting_status']='STOPPED';atomic_json(path,state)
                    print('Reporting failed; see logs/reporting.log',flush=True)
        return state

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=REPO/'native_t1/runs/control_mvp48_v1')
    p.add_argument('--out',type=Path,default=REPO/'native_t1/runs/control_mvp48_budget_single_v1');p.add_argument('--status',action='store_true')
    a=p.parse_args()
    if a.status:
        print(json.dumps(read_json(a.out/'run_state.json'),indent=2));return
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Controller received signal '+str(signum))
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:state=run(a.source,a.out)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps(state,indent=2),flush=True)
    if state['status']!='COMPLETE':raise SystemExit(2)

if __name__=='__main__':main()
