#!/usr/bin/env python3
"""Resumable measured two-worker dispatcher for the fixed schedule40 study.

Owns the shared GPU lock. Child workers never share models, optimizers, RNG or writers.
Only this study's child processes are managed; unrelated processes are read-only.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import argparse, fcntl, hashlib, json, math, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from meshflow_control.artifacts import atomic_json, file_sha256
REPO=Path(__file__).resolve().parents[1]
NAME='CHAIR_EARLY_ALIGNMENT_SCHEDULE_40_V1'
DEFAULT='/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1'
DATA='/mnt/e/MeshFlow-Control-artifacts/data'
OFFICIAL='/mnt/e/MeshFlow/native_t1/runs/control_mvp48_v1/checkpoints/official_chair_last.pt'
OLD='/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1'

def read(path): return json.loads(Path(path).read_text())
def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value: raise ValueError('Frozen identity differs: '+str(path))
    else: atomic_json(path,value)

def event(out,kind,**fields):
    p=Path(out)/'audit/controller.jsonl'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(dict(time_unix=time.time(),event=kind,**fields),allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())

def resources():
    def command(args):
        p=subprocess.run(args,capture_output=True,text=True); return dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
    fields=gpu['stdout'].strip().splitlines()[0].split(',') if gpu['code']==0 and gpu['stdout'].strip() else []
    parsed=dict(name=fields[0].strip(),total_mib=float(fields[1]),used_mib=float(fields[2]),free_mib=float(fields[3]),utilization=float(fields[4])) if len(fields)==5 else None
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.split(':')[0] in ('MemTotal','MemAvailable'): memory[line.split(':')[0]]=int(line.split()[1])*1024
    return dict(time_unix=time.time(),gpu=parsed,gpu_query=gpu,cpu_cores=os.cpu_count(),ram=memory,loadavg=list(os.getloadavg()),gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),cpu_processes=command(['ps','-eo','pid,pcpu,pmem,args','--sort=-pcpu']))

def snapshot(out,reason,active=()):
    r=resources(); event(out,'RESOURCES',reason=reason,active=list(active),snapshot=r); return r

def slots(r,peak):
    g=r.get('gpu')
    if not g or peak<=0: return 0
    headroom=max(2048.,.1*g['total_mib'])
    v=math.floor(max(0,g['free_mib']-headroom)/(1.2*peak))
    if r['ram'].get('MemAvailable',0)<12*1024**3 or r['cpu_cores']<4: return 0
    return max(0,v)

def protocol(out):
    p=dict(name=NAME,official_ema=OFFICIAL,official_file_sha256='bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d',logical_updates_per_arm=6000,physical_unique_updates=13000,schedules=dict(FM_ONLY=[0,0,6000],EARLY=[1,1000,6000],LATE=[5001,6000,6000]),common_prefix_updates=5000,condition_ratio=.4,training_parents=26,existing_training_tasks=203,effective_batch=8,microbatch=1,hybrid_lambda=.25,alignment_lambda=.1,alignment_clean_samples_per_active_batch=4,alignment_teacher_forwards_new=0,parameters='FP32',forward='BF16',integration='FP32',deterministic_strict=True,tf32=False,sdpa='math',evaluation=dict(objects=8,noise_per_condition=1,models=6,outputs=48,euler_steps=50,raw=True,source_GT_ranking=False,full_gallery=True),dispatch=dict(max_workers=2,incremental=True,initial_representative_updates=16,headroom='max(2048MiB,.1*totalVRAM)',memory_margin=1.2,cpu_threads_per_gpu_worker=2,cpu_OT_workers=4,per_process_vram_unavailable='Use conservative observed single-worker total device peak; includes desktop. Never sum allocated plus reserved.'),recovery=dict(oom_downgrades_max=1,retries_per_oom_affected_segment_max=1,checkpoint_interval=100,max_completed_replayed_updates_per_segment=99,batch_precision_settings_unchanged=True,serial_oom_or_numerical_failure='STOP_AFFECTED_TASK_RECORD_BLOCKED',no_unrelated_process_termination=True),retained_baseline='START remains default',git_commit_push=False)
    freeze(Path(out)/'protocol.json',p); return p

def run_cpu(out,cmd,name):
    snapshot(out,'before_'+name)
    log=Path(out)/'audit'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); event(out,'CPU_START',name=name,command=cmd)
    with log.open('a') as f: ret=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
    event(out,'CPU_END',name=name,returncode=ret.returncode,wall_seconds=time.perf_counter()-start)
    if ret.returncode: raise RuntimeError(name+' failed; see '+str(log))

def prepare(out):
    from meshflow_control.data.schedule40 import prepare_training_plan,prepare_teacher_registration
    protocol(out); snapshot(out,'experiment_preparation')
    p=prepare_training_plan(DATA,Path(out)/'stream')
    t=prepare_teacher_registration(DATA,Path(out)/'stream',Path(OLD),Path(out)/'teacher')
    run_cpu(out,[sys.executable,'-B','-m','meshflow_control.data.schedule40','--precompute','--data',DATA,'--out',str(Path(out)/'stream'),'--workers','4'],'prepare_OT')
    run_cpu(out,[sys.executable,'-B',str(REPO/'tools/evaluate_schedule40.py'),'--stage','prepare','--out',str(out)],'prepare_evaluation')
    atomic_json(Path(out)/'audit/prepared.json',dict(status='COMPLETE',plan=p,teacher=t))

def training_jobs(out):
    jobs=[]
    for seg,end in [('EARLY',6000),('COMMON',5000),('FM_ONLY',6000),('LATE',6000)]:
        jobs.append(dict(name=seg,end=end,phase='training',dependencies=[] if seg in ('EARLY','COMMON') else ['COMMON']))
    return jobs

def export_path(out,segment): return Path(out)/'inference'/(segment+'_generator_step6000.pt')
def completed(out,job):
    p=Path(out)/job['phase']/job['name']/'completed.json'
    return p.is_file() and read(p).get('status')=='COMPLETE'

def job_command(out,job,attempt):
    name=job['name']; out=Path(out)
    if job['phase']=='training':
        cmd=[sys.executable,'-B',str(REPO/'tools/train_schedule40.py'),'--segment',name,'--official',OFFICIAL,'--data-manifest',str(Path(DATA)/'train_manifest.json'),'--stream-root',str(out/'stream'),'--teacher-registration',str(out/'teacher'),'--out',str(out),'--attempt-id',attempt]
        latest=out/'training'/name/'latest.pt'
        if latest.exists(): cmd+=['--resume',str(latest)]
        elif name in ('FM_ONLY','LATE'): cmd+=['--fork',str(out/'checkpoints/COMMON_step5000.pt')]
    else:
        cmd=[sys.executable,'-B',str(REPO/'tools/evaluate_schedule40.py'),'--stage','sample','--out',str(out),'--label',name,'--checkpoint',job['checkpoint'],'--attempt',attempt]
    return cmd

def progress(out,job):
    p=Path(out)/job['phase']/job['name']/'progress.json'
    try: return read(p) if p.exists() else {}
    except (OSError,json.JSONDecodeError): return {}

def dispatch(out,jobs):
    out=Path(out); state_path=out/'audit/dispatch_state.json'; state_path.parent.mkdir(parents=True,exist_ok=True)
    state=read(state_path) if state_path.exists() else dict(max_workers=2,oom_downgrades=0,attempts={},failed={},phase='')
    # Never silently resume a process interrupted without a completed/failure receipt.
    active={}; device_peak=0.; first_owner=None; representative=False; last_print=0; started=time.perf_counter()
    remaining=[j for j in jobs if not completed(out,j)]
    while remaining or active:
        r=snapshot(out,'dispatch_tick',[x for x in active])
        if len(active)==1 and r['gpu']:
            device_peak=max(device_peak,r['gpu']['used_mib'])
            j=next(iter(active.values()))['job']; pr=progress(out,j)
            count=pr.get('physical_optimizer_updates_this_attempt',pr.get('completed',0))
            representative=(int(count)>=16 if j['phase']=='training' else int(count)>=1) and pr.get('attempt_id',pr.get('attempt'))==next(iter(active.values()))['attempt']
            device_peak=max(device_peak,1.2*float(pr.get('cuda_peak_reserved_bytes',0))/1024**2)
        for name,owner in list(active.items()):
            rc=owner['process'].poll()
            if rc is None: continue
            owner['log'].close(); elapsed=time.perf_counter()-owner['start']; job=owner['job']
            event(out,'WORKER_END',name=name,phase=job['phase'],attempt=owner['attempt'],returncode=rc,wall_seconds=elapsed,worker_count_at_finish=len(active))
            del active[name]; snapshot(out,'worker_finished',active)
            failure_key=job['phase']+':'+name
            if rc==0 and completed(out,job):
                state['failed'].pop(failure_key,None); atomic_json(state_path,state); continue
            folder=out/job['phase']/name
            failures=list(folder.glob('**/*failure*.json'))+list(folder.glob('**/*failed*.json'))+list(folder.glob('**/final_status.json'))
            found=[]
            for p in failures:
                try:
                    d=read(p)
                    if d.get('attempt_id',d.get('attempt',owner['attempt']))==owner['attempt']: found.append(d)
                except Exception: pass
            oom=any(x.get('status')=='OOM' or 'OutOfMemoryError' in x.get('error','') for x in found)
            if oom and owner['attempt']=='initial' and owner.get('parallel_exposed',False) and (state['oom_downgrades']==0 or owner.get('launched_before_oom_downgrade',False)):
                state.update(max_workers=1,oom_downgrades=1); event(out,'OOM_DOWNGRADE',failed_worker=name,new_max_workers=1,reason='User explicitly authorized two-to-one fallback; no batch/precision change',failure_receipts=found)
                state['failed'][failure_key]=dict(status='OOM_PENDING_SINGLE_RETRY',attempt=owner['attempt']); remaining.insert(0,job)
            else:
                state['failed'][failure_key]=dict(status='BLOCKED',attempt=owner['attempt'],returncode=rc,oom=oom,failure_receipts=found)
                event(out,'BLOCKED_WORKER',name=name,reason='No silent retry: serial OOM, numerical/code failure, or missing completion receipt',details=state['failed'][failure_key])
                # Preserve independent running work; descendants cannot be dispatched.
            atomic_json(state_path,state)
        ready=[j for j in remaining if not any(j['name']==n for n in active) and all(completed(out,next(x for x in jobs if x['name']==d)) for d in j['dependencies']) and state['failed'].get(j['phase']+':'+j['name'],{}).get('status')!='BLOCKED']
        if ready and len(active)<state['max_workers']:
            if not active: allowed=bool(r['gpu'] and r['gpu']['free_mib']>=max(2048,.1*r['gpu']['total_mib'])+4096 and r['ram']['MemAvailable']>=12*1024**3)
            else: allowed=representative and slots(r,device_peak)>=1
            if allowed:
                job=ready[0]; name=job['name']; attempt_key=job['phase']+':'+name; tries=state['attempts'].get(attempt_key,0)
                if tries>0 and state['failed'].get(job['phase']+':'+name,{}).get('status')!='OOM_PENDING_SINGLE_RETRY': raise RuntimeError('Unclosed prior attempt requires explicit accounting review: '+name)
                if tries>=2: raise RuntimeError('One recovery maximum exceeded')
                if tries and job['phase']=='training':
                    failed=state['failed'][job['phase']+':'+name]
                    folder=out/'training'/name; saved=read(folder/'latest.json') if (folder/'latest.json').exists() else {'global_step':5000 if name in ('FM_ONLY','LATE') else 0}
                    failure=read(folder/'attempts'/failed['attempt']/'final_status.json')
                    completed_step=int(failure['step']); recovery_step=int(saved['global_step']); replayed=completed_step-recovery_step
                    if not 0<=replayed<=99:
                        state['failed'][job['phase']+':'+name].update(status='BLOCKED',reason='Replay distance exceeds frozen99-update maximum'); atomic_json(state_path,state); remaining.remove(job); event(out,'REPLAY_BLOCKED',name=name,replayed=replayed); continue
                    event(out,'RECOVERY_ACCOUNTING',name=name,failed_completed_step=completed_step,recovery_step=recovery_step,completed_updates_to_reexecute=replayed,partial_failed_call_counted_separately=True)
                attempt='initial' if tries==0 else 'oom_serial_retry'; state['attempts'][attempt_key]=tries+1; state['phase']=job['phase']; atomic_json(state_path,state)
                cmd=job_command(out,job,attempt); logpath=out/'audit'/(job['phase']+'_'+name+'_'+attempt+'.log'); f=logpath.open('a')
                event(out,'WORKER_START',name=name,phase=job['phase'],attempt=attempt,command=cmd,worker_count=len(active)+1,observed_single_device_peak_mib=device_peak,memory_proxy_scope='total device peak includes desktop, conservative upper bound; process memory unavailable',resources=r)
                proc=subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
                active[name]=dict(job=job,process=proc,log=f,start=time.perf_counter(),attempt=attempt,launched_before_oom_downgrade=state['oom_downgrades']==0); remaining.remove(job)
                if len(active)==1: representative=False
                else:
                    for item in active.values(): item['parallel_exposed']=True
                if first_owner is None: first_owner=name
                atomic_json(out/'audit/active_workers.json',dict(controller_pid=os.getpid(),workers={k:dict(pid=v['process'].pid,phase=v['job']['phase'],attempt=v['attempt']) for k,v in active.items()}))
        atomic_json(out/'audit/active_workers.json',dict(controller_pid=os.getpid(),workers={k:dict(pid=v['process'].pid,phase=v['job']['phase'],attempt=v['attempt']) for k,v in active.items()}))
        if not active and remaining and not ready:
            event(out,'DEPENDENCIES_BLOCKED',remaining=[j['name'] for j in remaining]); break
        if time.monotonic()-last_print>=30:
            print(json.dumps(dict(phase=jobs[0]['phase'],active={k:progress(out,v['job']) for k,v in active.items()},pending=[j['name'] for j in remaining],device=r['gpu'],max_workers=state['max_workers']),default=str),flush=True); last_print=time.monotonic()
        time.sleep(5)
    summary=dict(status='COMPLETE' if all(completed(out,j) for j in jobs) else 'PARTIAL_BLOCKED',phase=jobs[0]['phase'],wall_seconds=time.perf_counter()-started,completed=[j['name'] for j in jobs if completed(out,j)],failed=state['failed'],oom_downgrades=state['oom_downgrades'])
    atomic_json(out/'audit'/(jobs[0]['phase']+'_dispatch_receipt.json'),summary); return summary

def sample_jobs(out):
    paths=dict(START5000='/mnt/e/MeshFlow/native_t1/checkpoints/chair_hybrid_start_total5000.pt',HIST_FM6000=OLD+'/inference/A_generator_step1000.pt',HIST_FREE_ALIGN6000=OLD+'/inference/B_generator_step1000.pt',**{s:str(export_path(out,s)) for s in ('FM_ONLY','EARLY','LATE')})
    return [dict(name=k,phase='sampling',checkpoint=v,dependencies=[]) for k,v in paths.items()]

def freeze_sources(out):
    files=[REPO/'tools/run_schedule40.py',REPO/'tools/train_schedule40.py',REPO/'tools/evaluate_schedule40.py',*sorted((REPO/'src/meshflow_control').rglob('*.py'))]
    hashes={str(p.relative_to(REPO)):file_sha256(p) for p in files if p.is_file()}
    freeze(Path(out)/'audit/source_hashes_before.json',hashes)
    import shutil
    for p in files:
        if p.is_file():
            dest=Path(out)/'source_code'/p.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists(): shutil.copyfile(p,dest)
            elif file_sha256(dest)!=file_sha256(p): raise ValueError('Source snapshot differs')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default=DEFAULT); p.add_argument('--stage',choices=['prepare','train','sample','evaluate','all'],default='all'); a=p.parse_args(); out=Path(a.out).resolve()
    if out.name!='chair_early_alignment_schedule40_v1' or out.parent.name!='experiments': raise ValueError('Wrong output directory')
    out.mkdir(parents=True,exist_ok=True); protocol(out)
    try:
        if a.stage in ('prepare','all'): prepare(out)
        if a.stage in ('train','sample','all'):
            freeze_sources(out)
            with open('/tmp/wrp_meshflow_jedge5_diag_gpu0.lock','a+') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if a.stage in ('train','all'):
                    result=dispatch(out,training_jobs(out))
                    if result['status']!='COMPLETE': print(json.dumps(result)); return 2
                if a.stage in ('sample','all'):
                    result=dispatch(out,sample_jobs(out))
                    if result['status']!='COMPLETE': print(json.dumps(result)); return 2
        if a.stage in ('evaluate','all'):
            for s in ('score','render'): run_cpu(out,[sys.executable,'-B',str(REPO/'tools/evaluate_schedule40.py'),'--stage',s,'--out',str(out)],s)
        event(out,'CONTROLLER_PHASE_COMPLETE',phase=a.stage); return 0
    except BaseException as exc:
        event(out,'CONTROLLER_FAILURE',phase=a.stage,error=repr(exc),traceback=traceback.format_exc()); raise
if __name__=='__main__': sys.exit(main())