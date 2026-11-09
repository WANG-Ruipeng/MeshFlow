#!/usr/bin/env python3
"""Measured two-worker dispatcher for pre-registered finite recipe jobs.

Only owns jobs in an immutable JSON wave. Old schedule40 sources remain unchanged.
Generic process scheduling; scientific settings belong to frozen worker registrations.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse,fcntl,json,math,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from meshflow_control.artifacts import atomic_json,file_sha256

REPO=Path(__file__).resolve().parents[1]
DEFAULT=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
LOCK='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'

def read(p): return json.loads(Path(p).read_text())
def freeze(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if read(p)!=value: raise ValueError('Frozen registration changed: '+str(p))
    else: atomic_json(p,value)

def event(out,kind,**kw):
    p=Path(out)/'audit/controller.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(dict(time_unix=time.time(),event=kind,**kw),allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())

def resources():
    def command(args):
        p=subprocess.run(args,capture_output=True,text=True)
        return dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    q=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
    fields=q['stdout'].strip().splitlines()[0].split(',') if q['code']==0 and q['stdout'].strip() else []
    gpu=dict(name=fields[0].strip(),total_mib=float(fields[1]),used_mib=float(fields[2]),free_mib=float(fields[3]),utilization=float(fields[4])) if len(fields)==5 else None
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.split(':')[0] in ('MemTotal','MemAvailable'): mem[line.split(':')[0]]=int(line.split()[1])*1024
    return dict(time_unix=time.time(),gpu=gpu,gpu_query=q,cpu_cores=os.cpu_count(),ram=mem,loadavg=list(os.getloadavg()),
        gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        cpu_processes=command(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']))

def snapshot(out,why,active=()):
    r=resources();event(out,'RESOURCES',reason=why,active=list(active),snapshot=r);return r

def slots(r,observed_single_device_peak):
    g=r.get('gpu')
    if not g or observed_single_device_peak<=0:return 0
    if r['ram'].get('MemAvailable',0)<12*1024**3 or r['cpu_cores']<4:return 0
    if r['loadavg'][0]>.8*r['cpu_cores']:return 0
    if g['utilization']>=90:return 0
    return max(0,math.floor(max(0,g['free_mib']-max(2048,.1*g['total_mib']))/(1.2*observed_single_device_peak)))

def progress(job):
    p=Path(job['progress_path'])
    try:return read(p) if p.exists() else {}
    except (OSError,json.JSONDecodeError):return {}

def completed(job):
    p=Path(job['completed_path'])
    if not p.is_file():return False
    d=read(p)
    return d.get('status')=='COMPLETE' and all(Path(a).is_file() for a in job.get('required_completion_files',[]))

def validate_wave(path,out):
    d=read(path)
    if d.get('schema')!='recipe_factorial_dispatch_wave_v1':raise ValueError('Wrong wave schema')
    if d.get('max_workers')!=2 or not d.get('jobs'):raise ValueError('Fixed maximum2 and nonempty jobs required')
    jobs=d['jobs']
    if len({j['name'] for j in jobs})!=len(jobs):raise ValueError('Duplicate owned job name')
    for j in jobs:
        if j['phase'] not in ('training','sampling'):raise ValueError('Unknown phase')
        for key in ('status_root','progress_path','completed_path'):
            p=Path(j[key]).resolve()
            if not p.is_relative_to(Path(out).resolve()):raise ValueError('Job writes outside registered run')
        if j['command'][0]!=sys.executable:raise ValueError('Worker interpreter identity differs')
        if not Path(j['command'][2]).resolve().is_relative_to(REPO):raise ValueError('Worker script outside source checkout')
        for p,h in j.get('source_hashes',{}).items():
            if file_sha256(p)!=h:raise ValueError('Registered worker source changed: '+p)
    freeze(Path(out)/'audit/waves'/(d['wave']+'.json'),d)
    return d

def failure_receipts(job,attempt):
    result=[]
    for p in Path(job['status_root']).glob('**/*.json'):
        if not any(x in p.name for x in ('failure','failed','final_status')):continue
        try:
            d=read(p)
            if d.get('attempt_id',d.get('attempt',attempt))==attempt:result.append(d)
        except (OSError,json.JSONDecodeError):pass
    return result

def is_oom(receipts):
    return any(x.get('status')=='OOM' or 'OutOfMemoryError' in str(x.get('error','')) for x in receipts)

def recovery_allowed(owner,state,oom):
    return bool(oom and owner['attempt']=='initial' and owner.get('parallel_exposed',False)
        and (state['oom_downgrades']==0 or owner.get('launched_before_oom_downgrade',False)))

def clear_solo_eligibility(active):
    cleared=[]
    if len(active)==1:
        name,owner=next(iter(active.items()))
        if owner.get('parallel_exposed',False):
            owner['parallel_exposed']=False;cleared.append(name)
    return cleared

def command(job,attempt):
    cmd=list(job['command'])+[job.get('attempt_option','--attempt-id'),attempt]
    latest=job.get('latest_checkpoint')
    if latest and Path(latest).exists():
        if attempt=='initial':raise ValueError('Initial attempt cannot silently resume an existing checkpoint')
        cmd+=[job.get('resume_option','--resume'),latest]
    return cmd

def replay_count(job,receipts):
    if job['phase']!='training':return None
    saved=read(job['latest_receipt']) if Path(job['latest_receipt']).exists() else {'global_step':0}
    steps=[int(x.get('step',x.get('global_step',0))) for x in receipts]
    return max(steps,default=0)-int(saved.get('global_step',saved.get('step',0)))

def proc_identity(pid):
    try:
        tail=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),starttime=int(tail[19]),ppid=int(tail[1]),state=tail[0])
    except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError,ValueError):
        return None

def check_prior_owners(out):
    p=Path(out)/'audit/active_workers.json'
    if not p.exists():return
    d=read(p)
    if d.get('workers'):
        live=[]
        for name,owner in d['workers'].items():
            now=proc_identity(owner['pid']);prior=owner.get('process_identity')
            if now and (not prior or now['starttime']==prior['starttime']):live.append(name)
        raise RuntimeError('Unclosed previous owner registry requires review; live='+str(live))

def dispatch(out,wave):
    out=Path(out);jobs=wave['jobs'];sp=out/'audit/dispatch_state.json';sp.parent.mkdir(parents=True,exist_ok=True)
    check_prior_owners(out)
    state=read(sp) if sp.exists() else dict(max_workers=2,oom_downgrades=0,attempts={},failed={})
    remaining=[j for j in jobs if not completed(j)];active={};device_peak=0.;representative=False
    start=time.perf_counter();last_print=0;halt=False
    while remaining or active:
        r=snapshot(out,'dispatch_tick',active)
        if len(active)==1 and r['gpu']:
            device_peak=max(device_peak,r['gpu']['used_mib'])
            owner=next(iter(active.values()));pr=progress(owner['job'])
            count=int(pr.get('physical_optimizer_updates_this_attempt',pr.get('completed',0)))
            match=pr.get('attempt_id',pr.get('attempt'))==owner['attempt']
            representative=match and count>=(16 if owner['job']['phase']=='training' else 1)
            # Reserved counters include no desktop/context. Device total is the conservative observed proxy.
            device_peak=max(device_peak,1.2*float(pr.get('cuda_peak_reserved_bytes',0))/1024**2)
        finished=[(n,o,o['process'].poll()) for n,o in list(active.items()) if o['process'].poll() is not None]
        # Snapshot same-poll exits before changing active or eligibility.
        for name,owner,rc in finished:
            owner['log'].close();job=owner['job'];key=wave['wave']+':'+name
            event(out,'WORKER_END',wave=wave['wave'],name=name,phase=job['phase'],attempt=owner['attempt'],returncode=rc,wall_seconds=time.perf_counter()-owner['start'])
            del active[name];snapshot(out,'worker_finished',active)
            if rc==0 and completed(job):
                state['failed'].pop(key,None);atomic_json(sp,state);continue
            receipts=failure_receipts(job,owner['attempt']);oom=is_oom(receipts)
            if recovery_allowed(owner,state,oom):
                state.update(max_workers=1,oom_downgrades=1)
                state['failed'][key]=dict(status='OOM_PENDING_SINGLE_RETRY',attempt=owner['attempt'],receipts=receipts)
                remaining.insert(0,job)
                event(out,'OOM_DOWNGRADE',wave=wave['wave'],name=name,maximum=1,receipts=receipts)
            else:
                state['failed'][key]=dict(status='BLOCKED',attempt=owner['attempt'],returncode=rc,oom=oom,receipts=receipts)
                halt=True
                event(out,'BLOCKED_WORKER',wave=wave['wave'],name=name,reason='No authorized retry for solo OOM, numerical/code failure or ambiguous completion')
            atomic_json(sp,state)
        for name in clear_solo_eligibility(active):
            event(out,'PARALLEL_RECOVERY_ELIGIBILITY_CLEARED',wave=wave['wave'],name=name,reason='Survivor now solo; historical overlap does not authorize later retry')
        if remaining and not halt and len(active)<state['max_workers']:
            if not active:
                allowed=bool(r['gpu'] and r['gpu']['free_mib']>=max(2048,.1*r['gpu']['total_mib'])+4096
                    and r['ram'].get('MemAvailable',0)>=12*1024**3 and r['loadavg'][0]<.8*r['cpu_cores'])
            else:allowed=representative and slots(r,device_peak)>=1
            if allowed:
                job=remaining[0];name=job['name'];key=wave['wave']+':'+name;tries=state['attempts'].get(key,0)
                if tries and state['failed'].get(key,{}).get('status')!='OOM_PENDING_SINGLE_RETRY':
                    raise RuntimeError('Unclosed prior attempt requires accounting review: '+key)
                if tries>=2:raise RuntimeError('One recovery maximum exceeded')
                if tries:
                    replay=replay_count(job,state['failed'][key]['receipts'])
                    if replay is not None and not 0<=replay<=99:
                        state['failed'][key]['status']='BLOCKED';atomic_json(sp,state);halt=True
                        event(out,'REPLAY_BLOCKED',name=name,replay=replay);continue
                    event(out,'RECOVERY_ACCOUNTING',name=name,completed_updates_to_reexecute=replay,partial_failed_calls_counted=True)
                attempt='initial' if tries==0 else 'oom_serial_retry'
                state['attempts'][key]=tries+1;atomic_json(sp,state)
                cmd=command(job,attempt);lp=out/'audit/logs'/(wave['wave']+'_'+name+'_'+attempt+'.log');lp.parent.mkdir(parents=True,exist_ok=True);f=lp.open('a')
                event(out,'WORKER_START',wave=wave['wave'],name=name,phase=job['phase'],attempt=attempt,command=cmd,worker_count=len(active)+1,observed_single_device_peak_mib=device_peak,resources=r)
                proc=subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                active[name]=dict(job=job,process=proc,log=f,start=time.perf_counter(),attempt=attempt,process_identity=proc_identity(proc.pid),parallel_exposed=False,launched_before_oom_downgrade=state['oom_downgrades']==0)
                remaining.remove(job)
                if len(active)==1:representative=False;device_peak=0.
                else:
                    for o in active.values():o['parallel_exposed']=True
        atomic_json(out/'audit/active_workers.json',dict(controller_pid=os.getpid(),wave=wave['wave'],workers={n:dict(pid=o['process'].pid,phase=o['job']['phase'],attempt=o['attempt'],process_identity=o.get('process_identity')) for n,o in active.items()}))
        if halt and not active:break
        if time.monotonic()-last_print>=60:
            print(json.dumps(dict(wave=wave['wave'],active={n:progress(o['job']) for n,o in active.items()},pending=[j['name'] for j in remaining],gpu=r['gpu'],max_workers=state['max_workers']),default=str),flush=True)
            last_print=time.monotonic()
        if active or remaining:time.sleep(5)
    result=dict(status='COMPLETE' if all(completed(j) for j in jobs) else 'PARTIAL_BLOCKED',wave=wave['wave'],wall_seconds=time.perf_counter()-start,completed=[j['name'] for j in jobs if completed(j)],remaining=[j['name'] for j in remaining],failed=state['failed'],oom_downgrades=state['oom_downgrades'])
    atomic_json(out/'audit'/('dispatch_'+wave['wave']+'.json'),result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(DEFAULT));p.add_argument('--jobs');p.add_argument('--resources-only',action='store_true');a=p.parse_args()
    out=Path(a.out).resolve()
    if out!=DEFAULT.resolve():raise ValueError('Wrong experiment root')
    if a.resources_only:snapshot(out,'manual_preflight');return 0
    if not a.jobs:raise ValueError('Explicit frozen jobs wave required')
    wave=validate_wave(a.jobs,out)
    try:
        with open(LOCK,'a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=dispatch(out,wave)
        return 0 if result['status']=='COMPLETE' else 2
    except BaseException as e:
        event(out,'CONTROLLER_FAILURE',wave=wave['wave'],error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':sys.exit(main())