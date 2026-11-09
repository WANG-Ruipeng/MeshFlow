"""Single resumable controller for the preregistered Chair experiment."""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import difflib
import json
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from .event_audit import audit_training_events
from .common import REPO, LIMITS, GPU_LOCK, atomic_json, read_json, digest, json_hash, freeze_json, append_event, controller_lock
from experiments.mf_h1_local0.control_mvp48_budget_single.run_all import wait_managed_child

MODULE='experiments.mf_h1_local0.chair_domain_control_v1'
ARMS=('T1','Geo','Full')


def stages():
    result=[]
    def add(key,module,args,deps=()):
        result.append(dict(key=key,module=module,args=args,dependencies=list(deps)))
    add('wiring','precheck',[])
    starts=[]
    for arm in ARMS:
        key=f'eval_{arm}_0'; starts.append(key)
        add(key,'evaluation',['--arm',arm,'--phase-step','0'],['wiring'])
    for phase,prior in ((1000,0),(2500,1000)):
        for arm in ARMS:
            deps=starts if prior==0 else [f'eval_{arm}_{prior}']
            add(f'train_{arm}_{phase}','training',['--arm',arm,'--target',str(phase)],deps)
            add(f'eval_{arm}_{phase}','evaluation',['--arm',arm,'--phase-step',str(phase)],[f'train_{arm}_{phase}'])
    for arm in ARMS:
        add(f'train_{arm}_3000','training',['--arm',arm,'--target','3000'],[f'eval_{arm}_2500'])
    for arm in ARMS:
        add(f'eval_{arm}_3000','evaluation',['--arm',arm,'--phase-step','3000'],[f'train_{a}_3000' for a in ARMS])
    add('reporting','reporting',[],[s['key'] for s in result])
    return result


def code_snapshot(out,verify=False):
    files=sorted(Path(__file__).parent.glob('*.py'))
    hashes={str(p.relative_to(REPO)):digest(p) for p in files}
    target=out/'registration/code_hashes.json'
    if verify:
        if read_json(target)!=hashes: raise RuntimeError('Frozen experiment code changed')
    else:
        freeze_json(target,hashes)
        chunks=[]
        for p in files:
            chunks.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(p.relative_to(REPO))))
        content=''.join(chunks)
        diff=out/'code.diff'
        if diff.exists() and diff.read_text(encoding='utf-8')!=content: raise ValueError('Frozen code.diff changed')
        if not diff.exists(): diff.write_text(content,encoding='utf-8')
    return hashes


def frozen_protocol(source,out,gate):
    from .prepare import prepare
    prepare(source,out)
    for name in ('data_audit.json','source_restore_audit.json','training_cpu_tests.json','preparation_validation.json','width_cpu_math_tests.json','controller_tests.json','event_audit_cpu_tests.json','reporting_cpu_tests.json'):
        if read_json(out/'registration'/name).get('status')!='PASS': raise ValueError('Required finite CPU check not PASS: '+name)
    old=read_json(source/'registration/train_protocol.json')['protocol']
    protected={str(p.relative_to(REPO)):digest(p) for p in sorted((REPO/'experiments/mf_h1_local0/control_mvp48_budget_single').glob('*.py'))}
    protocol=dict(schema='chair_domain_control_v1',source=str(source),out=str(out),
        prompt_sha256=digest(out/'registration/requested_protocol.md'),source_data_revision='repair_v2',
        source_data_ready_sha256=gate['data_ready_sha256'],source_train_ready_sha256=gate['train_ready_sha256'],
        source_train_protocol=old,restore_mode='RESTORE_ALL',source_checkpoints={k:v['hashes'] for k,v in gate['endpoints'].items()},
        original_fork_sha256=gate['fork']['checkpoint_sha256'],
        frozen_files={name:digest(out/name) for name in ('train_plan.json','task_manifest.json','panels.json')},
        protected_adapter_files=protected,main_arms=list(ARMS),start_step=1000,
        phase_checkpoints=[0,1000,2500,3000],conditional_total_checkpoints=[1000,2000,3500,4000],
        main_updates_per_arm=3000,budgets=LIMITS,effective_batch=8,microbatch=1,
        optimizer=old['optimizer'],loss=old['loss'],full_ramp='always 1; original warmup is complete',
        full_ramp_index='500 + phase_step',sampler='Euler50',NFE=50,precision=old['precision'],determinism=old['determinism'],
        gpu_workers=1,gpu_lock=GPU_LOCK,concurrency_reason='Explicit experiment protocol requires one formal GPU worker',
        generator_inputs='C, free Gaussian/state, N, known mask/role/time, and Geo derived exclusively from current C',
        forbidden_generator_inputs=['alpha','UID','full GT','target width','GT free source face IDs','GT bbox'],
        axis='canonical X; no certified semantic left-right metadata',width_quantile='q95-q05, equal triangle corner weights, numpy linear quantiles',
        width_undefined='W1 <= 1e-8 L0',alpha_hat='sum(C1x*Calphax)/sum(C1x^2), fixed origin',
        alpha_hat_undefined='denominator <= 1e-12 * L0^2 * number_of_C_corners',
        cross_condition_metric='Frozen corresponding GT points to predicted free surface, one-direction RMS/L0 and coverage <= .01 L0; distinct from original symmetric area-weighted metric',
        response_slope='(W1.1-W.9)/(.2*W1)',width_absolute_error='abs(Wpred-Wgt)/L0',
        historical_missing='NA; never regenerate missing historical outputs',
        source_scorer_identity=read_json(source/'registration/eval_protocol.json')['scorer_identity'],
        final_selection='all three fixed phase3000 / cumulative4000 endpoints',
        CPU_analysis_budget=dict(B_groups=72,width_rows=216,cross_condition_cells=648,analytic_candidates=432,analytic_original_score_calls=288,analytic_original_score_cache_hits=144,corresponding_point_distance_calls=1080),
        CPU_preparation_budget=dict(positive_geometry_score_calls=12,unique_teacher_OT=8,width_math_distance_calls=9),
        repetitions_authorized_max=2,repetitions_planned=0,stages=stages())
    freeze_json(out/'protocol.json',protocol)
    return protocol


def validate_frozen_inputs(source,out,protocol):
    for name,sha in protocol['frozen_files'].items():
        if digest(out/name)!=sha: raise ValueError('Frozen input changed: '+name)
    for name,sha in read_json(out/'panels.json')['frozen_files'].items():
        if digest(out/Path(name))!=sha: raise ValueError('Frozen input/reference bank changed: '+name)
    for group in (protocol['source_scorer_identity'],protocol['protected_adapter_files'],read_json(source/'TRAIN_READY.json')['source_hashes']):
        for name,sha in group.items():
            if digest(REPO/name)!=sha: raise ValueError('Protected code changed: '+name)
    for name,record in read_json(source/'DATA_READY.json')['files'].items():
        if digest(source/name)!=record['sha256']: raise ValueError('Protected source data changed: '+name)
    for name,key in (('DATA_READY.json','source_data_ready_sha256'),('TRAIN_READY.json','source_train_ready_sha256')):
        if digest(source/name)!=protocol[key]: raise ValueError('Source readiness changed: '+name)


def worker_status(out,stage):
    out=Path(out);args=stage['args'];kind=stage['module']
    if kind=='reporting':
        report=read_json(out/'reporting_status.json')
        if report.get('status')=='PASS':
            required={'CHAIR_DOMAIN_CONTROL_REPORT.md','A_context_per_case.csv','A_context_per_parent.csv','B_width_response.csv','B_cross_condition_scores.csv','B_analytic_baselines.csv','C_holdout_regression.csv','checkpoint_manifest.json','execution_accounting.json','resource_summary.csv','protected_assets_after.json','restart.txt','analysis_ledger.json','history_reuse.json','figures/manifest.json','code.diff'}
            if not required <= {x['path'] for x in report['artifacts']}: raise ValueError('Report artifact manifest incomplete')
            panels=read_json(out/'panels.json')
            if report['completed_generations']!=panels['counts']['applicable_generation'] or report['completed_teacher']!=288: raise ValueError('Report actual model counts incomplete')
            if report['analytic_candidate_complete']!=432 or report['cross_condition_complete']!=648: raise ValueError('Report actual CPU analysis counts incomplete')
            for item in report['artifacts']:
                if digest(out/item['path'])!=item['sha256']: raise ValueError('Report artifact changed: '+item['path'])
            if read_json(out/'analysis_ledger.json').get('status')!='PASS': raise ValueError('CPU analytics unfinished')
        return report
    if kind=='precheck': return read_json(out/'registration/wiring_precheck.json')
    arm=args[args.index('--arm')+1]
    if kind=='evaluation':
        phase=int(args[args.index('--phase-step')+1])
        ledger=read_json(out/f'evaluation/phase{phase}/{arm}/ledger.json')
        ident=ledger['identity']
        if ident['arm']!=arm or ident['phase_step']!=phase or ident['protocol_sha256']!=digest(out/'protocol.json'):
            raise ValueError('Evaluation identity differs')
        protocol=read_json(out/'protocol.json')
        expected=(Path(protocol['source'])/'checkpoints'/f'{arm}_step1000.pt') if phase==0 else (out/'checkpoints'/f'{arm}_phase{phase}_total{1000+phase}.pt')
        if Path(ident['checkpoint']).resolve()!=expected.resolve(): raise ValueError('Evaluation used wrong registered checkpoint path')
        if digest(expected)!=ident['checkpoint_sha256']: raise ValueError('Evaluation checkpoint changed')
        metadata=ledger['model_metadata'];hashes=ledger['model_hashes']
        if metadata['variant']!=arm or metadata['logical_step']!=1000+phase or (phase and metadata.get('phase_step')!=phase): raise ValueError('Evaluated model arm/phase/total differs')
        if hashes['checkpoint_sha256']!=ident['checkpoint_sha256'] or hashes['model_state_sha256']!=metadata['model_state_sha256']: raise ValueError('Evaluated model hashes differ')
        if phase==0:
            if protocol['source_checkpoints'][arm]['checkpoint_sha256']!=ident['checkpoint_sha256']: raise ValueError('Phase0 source checkpoint changed')
        else:
            sidecar=read_json(expected.with_suffix('.json'))
            if sidecar['checkpoint_sha256']!=ident['checkpoint_sha256'] or sidecar['metadata']['model_state_sha256']!=hashes['model_state_sha256'] or sidecar['metadata']['protocol_sha256']!=json_hash(protocol): raise ValueError('Evaluation checkpoint sidecar/registered protocol differs')
        for row in ledger['rows']+ledger['teacher_rows']:
            for field in ('raw','score'):
                if row.get(field+'_path') and digest(out/row[field+'_path'])!=row[field+'_sha256']:
                    raise ValueError('Evaluation artifact changed')
        return ledger
    if kind!='training': raise ValueError('Unknown worker')
    target=int(args[args.index('--target')+1]);total=1000+target
    summary=out/'training'/f'{arm}_to{target}_summary.json'
    if not summary.exists(): return dict(status='STOPPED',reason='No exact segment completion record')
    actual=read_json(summary)
    if actual.get('status')!='SEGMENT_COMPLETE' or actual.get('phase_step')!=target or actual.get('conditional_total_step')!=total:
        return dict(actual,status='STOPPED',reason='Exact phase and cumulative endpoint not completed')
    if actual.get('arm')!=arm: raise ValueError('Training arm differs')
    cp=out/'checkpoints'/f'{arm}_phase{target}_total{total}.pt'
    if not cp.is_file() or not cp.with_suffix('.json').is_file(): return dict(status='STOPPED',reason='Missing fixed checkpoint')
    record=read_json(cp.with_suffix('.json'));meta=record['metadata'];sha=digest(cp)
    if record['checkpoint_sha256']!=sha: raise ValueError('Fixed checkpoint SHA differs')
    if meta['variant']!=arm or meta['phase_step']!=target or meta['conditional_total_step']!=total or meta['logical_step']!=total:
        raise ValueError('Fixed checkpoint arm/phase/total differs')
    ph=json_hash(read_json(out/'protocol.json'))
    if meta['protocol_sha256']!=ph or actual.get('protocol_sha256')!=ph: raise ValueError('Fixed checkpoint protocol identity differs')
    if Path(record['path']).resolve()!=cp.resolve(): raise ValueError('Fixed checkpoint path differs')
    return dict(actual,status='PASS',verified_checkpoint_sha256=sha,verified_checkpoint=str(cp),checkpoint_metadata=meta)


def orphan_budget_workers(out,process_table=None):
    if process_table is None: process_table=subprocess.run(['ps','-eo','pid,ppid,args'],capture_output=True,text=True,check=True).stdout
    expected=Path(out).resolve();found=[]
    for line in process_table.splitlines()[1:]:
        fields=line.strip().split(None,2)
        if len(fields)!=3: continue
        try: pid,ppid=int(fields[0]),int(fields[1]);argv=shlex.split(fields[2])
        except (ValueError,TypeError): continue
        if pid==os.getpid() or '-m' not in argv: continue
        index=argv.index('-m')
        if index+1>=len(argv) or argv[index+1] not in {MODULE+'.'+x for x in ('run_all','training','evaluation','precheck','reporting')}: continue
        output=argv[argv.index('--out')+1] if '--out' in argv else next((x.split('=',1)[1] for x in argv if x.startswith('--out=')),None)
        if output is None or (REPO/Path(output)).resolve()==expected: found.append(dict(pid=pid,ppid=ppid,command=fields[2]))
    return found


def resources(out,stage,event):
    import psutil
    def capture(argv):
        result=subprocess.run(argv,capture_output=True,text=True)
        return dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    mem=psutil.virtual_memory()
    record=dict(stage=stage,event=event,gpu=capture(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),
        cpu_logical=psutil.cpu_count(),cpu_available=len(os.sched_getaffinity(0)),cpu_percent=psutil.cpu_percent(interval=.1),
        ram_total=mem.total,ram_available=mem.available,load_average=os.getloadavg(),disk_free=shutil.disk_usage(out).free,
        process_table=capture(['ps','-eo','pid,ppid,pcpu,pmem,args']),formal_gpu_workers=1,reason='Explicit protocol cap; no extra worker permitted')
    append_event(out/'resource_dispatch.jsonl',record)
    return record


def execution_budget_audit(out):
    out=Path(out);errors=[];total_updates=training_forwards=0;arms={}
    registered=stages();panels=read_json(out/'panels.json')
    state=read_json(out/'run_state.json')
    if set(state['stages'])!={s['key'] for s in registered}: errors.append('Controller stage set incomplete')
    for arm in ARMS:
        try:
            actual=worker_status(out,next(s for s in registered if s['key']==f'train_{arm}_3000'))
            counts=actual.get('counts',{});u=int(counts.get('optimizer_updates',-1));f=int(counts.get('forward_attempts',-1))
            if actual.get('status')!='PASS' or u!=3000 or f!=24000: errors.append(arm+' fixed updates/forwards differ')
            for key in ('forward_returns','backward_attempts','backward_returns'):
                if counts.get(key)!=f: errors.append(arm+' incomplete '+key)
            if counts.get('optimizer_step_attempts')!=u: errors.append(arm+' optimizer attempts differ')
            if actual.get('checkpoint_metadata',{}).get('counts')!=counts: errors.append(arm+' checkpoint count mismatch')
            total_updates+=max(0,u);training_forwards+=max(0,f);arms[arm]=dict(updates=u,forward_attempts=f)
        except (OSError,ValueError,KeyError) as exc: errors.append(arm+': '+repr(exc))
    generations=teacher_forwards=sampling_calls=teacher_calls=expected=0;eval_rows=[]
    for stage in registered:
        if stage['module']!='evaluation': continue
        try:
            data=worker_status(out,stage);phase=int(stage['args'][-1])
            requests=[dict(r,table=table) for table in ('A','B','C') for r in panels[table] if phase in r['phase_steps']]
            want=sum(r.get('status') not in ('NOT_APPLICABLE','NOT_AVAILABLE') for r in requests);expected+=want
            sc=data.get('sampling_counts',{});tc=data.get('teacher_counts',{});rows=data.get('rows',[]);trows=data.get('teacher_rows',[])
            g=sc.get('complete_rollouts',0);t=tc.get('teacher_returns',0);sf=sc.get('model_forward_attempts',0);tf=tc.get('model_forward_attempts',0)
            generations+=g;teacher_forwards+=t;sampling_calls+=sf;teacher_calls+=tf
            if data.get('status')!='PASS' or len(rows)!=len(requests) or len(trows)!=24: errors.append(stage['key']+' unfinished request matrix')
            if {r['case_id'] for r in rows}!={r['case_id'] for r in requests} or len({r['case_id'] for r in rows})!=len(rows): errors.append(stage['key']+' request identity mismatch')
            if any(r.get('status') not in ('PASS','NOT_APPLICABLE','NOT_AVAILABLE') for r in rows) or any(r.get('status')!='PASS' for r in trows): errors.append(stage['key']+' failed rows')
            if g!=want or g!=sum(r['status']=='PASS' for r in rows) or t!=24: errors.append(stage['key']+' actual denominator mismatch')
            if sc.get('rollout_attempts')!=g or tc.get('teacher_attempts')!=t or sf!=50*g or tf!=t: errors.append(stage['key']+' attempt/NFE mismatch')
            for key in ('model_forward_returns','backbone_forward_attempts','backbone_forward_returns'):
                if sc.get(key)!=sf or tc.get(key)!=tf: errors.append(stage['key']+' incomplete '+key)
            if any(not r.get('score_path') for r in rows if r['status']=='PASS'): errors.append(stage['key']+' missing score')
            eval_rows.append(dict(stage=stage['key'],completed_generations=g,teacher_returns=t,model_forward_attempts=sf+tf))
        except (OSError,ValueError,KeyError) as exc: errors.append(stage['key']+': '+repr(exc))
    try:
        wiring=read_json(out/'registration/wiring_precheck.json');w=int(wiring.get('forward_attempts',wiring.get('forwards_attempted',-1)))
        wr=int(wiring.get('forward_returns',wiring.get('forwards_returned',-1)))
        if wiring.get('status')!='PASS' or not 0<=w<=32 or wr!=w or wiring.get('optimizer_updates')!=0: errors.append('Wiring budget/status mismatch')
    except (OSError,ValueError,KeyError) as exc: errors.append('wiring: '+repr(exc));w=0
    event_audit=audit_training_events(out)
    if event_audit['status']!='PASS': errors.extend(event_audit['errors'])
    total=training_forwards+sampling_calls+teacher_calls+w
    if total_updates!=9000 or training_forwards!=72000 or generations!=expected or teacher_forwards!=288: errors.append('Exact registered execution count mismatch')
    if generations>600 or total>102420: errors.append('Hard model budget exceeded')
    return dict(schema='chair_domain_execution_audit_v1',status='PASS' if not errors else 'FAIL',errors=errors,arms=arms,
        optimizer_updates=total_updates,training_forward_attempts=training_forwards,training_backward_attempts=training_forwards,
        planned_applicable_generations=expected,complete_rollouts=generations,teacher_returns=teacher_forwards,
        sampling_forward_attempts=sampling_calls,teacher_forward_attempts=teacher_calls,wiring_forward_attempts=w,
        model_forward_attempts=total,limits=LIMITS,evaluation=eval_rows,training_event_audit=event_audit,repetitions=0)


def bootstrap_report(out,state):
    lines=['# Chair Domain Control','', 'Execution status: '+state['status'],'',
        'This status record does not claim that queued work or implemented scripts are completed experiments.',
        'Fixed training: 9000 new optimizer updates. Planned generation maximum: 600; repeats planned: 0.',
        '', 'Resume command:', '```bash',state['restart_command'],'```']
    if state.get('error'): lines.extend(['',state['error']])
    (out/'CHAIR_DOMAIN_CONTROL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(source,out,history):
    source,out,history=source.resolve(),out.resolve(),history.resolve()
    if source==out or not out.is_relative_to(REPO/'native_t1/runs'): raise ValueError('Output must be isolated')
    for name in ('registration','logs','training','evaluation','checkpoints'): (out/name).mkdir(parents=True,exist_ok=True)
    with controller_lock(out):
        path=out/'run_state.json'
        state=read_json(path) if path.exists() else dict(schema='chair_domain_state_v1',status='PREFLIGHT',stages={},created_unix=time.time())
        state['restart_command']=f'cd /mnt/e/MeshFlow && {sys.executable} -B -m {MODULE}.run_all --source {source} --out {out} --history {history}'
        state.update(pid=os.getpid(),last_started_unix=time.time());atomic_json(path,state)
        try:
            processes=orphan_budget_workers(out)
            if processes: raise RuntimeError('Existing same-protocol workers: '+json.dumps(processes))
            from experiments.mf_h1_local0.control_mvp48.eval_checkpoint_gate import inspect_ready
            gate=inspect_ready(source,REPO);freeze_json(out/'registration/source_identity.json',gate)
            if shutil.disk_usage(out).free<25*1024**3: raise RuntimeError('Less than 25 GiB free; old assets protected')
            protocol=frozen_protocol(source,out,gate);state['protocol_sha256']=json_hash(protocol)
            code_snapshot(out);validate_frozen_inputs(source,out,protocol)
            if state['status']=='DONE':
                budget=execution_budget_audit(out)
                if budget['status']!='PASS' or worker_status(out,stages()[-1])['status']!='PASS': raise ValueError('Completed run failed identity audit')
                print('Already DONE; identities verified; zero new work.',flush=True);return state
            state['status']='RUNNING';atomic_json(path,state);bootstrap_report(out,state)
            for stage in stages():
                key=stage['key'];old=state['stages'].get(key,{})
                if old.get('status')=='PASS':
                    if worker_status(out,stage).get('status')!='PASS': raise RuntimeError('Saved completion differs: '+key)
                    continue
                failed=[d for d in stage['dependencies'] if state['stages'].get(d,{}).get('status')!='PASS']
                if failed:
                    state['stages'][key]=dict(status='BLOCKED_DEPENDENCY',dependencies=failed);atomic_json(path,state);continue
                code_snapshot(out,verify=True);validate_frozen_inputs(source,out,protocol)
                if orphan_budget_workers(out): raise RuntimeError('Existing worker blocks duplicate dispatch')
                command=[sys.executable,'-B','-m',MODULE+'.'+stage['module'],'--source',str(source),'--out',str(out),*stage['args']]
                if stage['module']=='reporting':
                    state['stages'][key]=dict(status='PENDING');atomic_json(path,state)
                    audit=execution_budget_audit(out);atomic_json(out/'registration/execution_budget_audit.json',audit)
                    if audit['status']!='PASS': raise ValueError('Actual execution audit failed: '+json.dumps(audit['errors']))
                    command+=['--history',str(history)]
                resources(out,key,'BEFORE_DISPATCH')
                state['active_stage']=key;state['stages'][key]=dict(status='RUNNING',command=command,started_unix=time.time());atomic_json(path,state)
                append_event(out/'execution_ledger.jsonl',dict(event='STAGE_BEGIN',stage=key,command=command))
                print(json.dumps(dict(event='STAGE_BEGIN',stage=key,time_unix=time.time())),flush=True)
                tick=time.perf_counter()
                with (out/'logs'/f'{key}.log').open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        state['worker_pid']=child.pid;atomic_json(path,state);rc=wait_managed_child(child)
                    except BaseException:
                        if child.poll() is None:
                            try: child.send_signal(signal.SIGTERM)
                            except ProcessLookupError: pass
                            wait_managed_child(child)
                        raise
                    finally:
                        state['worker_pid']=child.pid if child.poll() is None else None;atomic_json(path,state)
                record=dict(status='STOPPED',returncode=rc,elapsed_seconds=time.perf_counter()-tick,completed_unix=time.time(),log=f'logs/{key}.log')
                if rc==0:
                    actual=worker_status(out,stage);record['status']='PASS' if actual.get('status')=='PASS' else 'STOPPED'
                state['stages'][key]=record;state['worker_pid']=None;atomic_json(path,state)
                resources(out,key,'AFTER_WORKER')
                append_event(out/'execution_ledger.jsonl',dict(event='STAGE_END',stage=key,**record))
                print(json.dumps(dict(event='STAGE_END',stage=key,**record)),flush=True)
            code_snapshot(out,verify=True);validate_frozen_inputs(source,out,protocol)
            complete=set(state['stages'])=={s['key'] for s in stages()} and all(r['status']=='PASS' for r in state['stages'].values())
            state.update(status='DONE' if complete else 'PARTIAL_STOPPED',active_stage=None,finished_unix=time.time());atomic_json(path,state)
            if not complete: bootstrap_report(out,state)
        except BaseException as exc:
            state.update(status='STOPPED',error=repr(exc),traceback=traceback.format_exc());atomic_json(path,state)
            if not (out/'reporting_status.json').exists(): bootstrap_report(out,state)
            append_event(out/'execution_ledger.jsonl',dict(event='CONTROLLER_STOPPED',error=repr(exc)));raise
        return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=REPO/'native_t1/runs/control_mvp48_v1')
    p.add_argument('--out',type=Path,default=REPO/'native_t1/runs/chair_domain_control_v1')
    p.add_argument('--history',type=Path,default=REPO/'native_t1/runs/control_mvp48_budget_single_v1');p.add_argument('--status',action='store_true')
    a=p.parse_args()
    if a.status: print(json.dumps(read_json(a.out/'run_state.json'),indent=2));return
    def interrupted(signum,frame): raise KeyboardInterrupt('Controller received signal '+str(signum))
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try: state=run(a.source,a.out,a.history)
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)
    print(json.dumps(state,indent=2),flush=True)
    if state['status']!='DONE': raise SystemExit(2)


if __name__=='__main__': main()
