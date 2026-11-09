"""Recoverable entry and resource observer. Never automatically retries science."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_full_prior_motif_guidance_v1'
PACKAGE = 'r2.pilots.bench_full_prior_motif_guidance_v1'

def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
def append(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def gpu_snapshot():
    command=['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    p=subprocess.run(command,capture_output=True,text=True,timeout=20)
    result={'utc':stamp(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    if p.returncode==0:
        a=p.stdout.strip().splitlines()[0].split(',')
        result.update(name=a[0].strip(),total_mib=float(a[1]),used_mib=float(a[2]),free_mib=float(a[3]),utilization_percent=float(a[4]))
    return result
def full_resources():
    from r2.pilots.bench_adaptation_strength_v1.common import resources
    result=resources();result['parsed_gpu']=gpu_snapshot();return result

def observe_child(command,stage,worker,dispatch=None):
    folder=OUT/'workflow'/'attempts'/(stage+'__'+worker+'__'+str(time.time_ns()))
    folder.mkdir(parents=True)
    before=full_resources();write(folder/'resources_before.json',before)
    write(folder/'attempt.json',{'utc':stamp(),'stage':stage,'worker':worker,'command':command,
          'dispatch':str(dispatch) if dispatch else None,'policy':'NO_AUTOMATIC_RETRY',
          'resources_scope':'host aggregate, child RSS and child-scoped CUDA allocator; WDDM process VRAM may be unavailable'})
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    start=time.perf_counter();samples=[]
    with (folder/'stdout.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(folder/'pid.json',{'pid':child.pid,'parent_pid':os.getpid(),'utc':stamp()})
        print(json.dumps({'stage':stage,'worker':worker,'pid':child.pid,'log':str(folder/'stdout.log')}),flush=True)
        import psutil
        while child.poll() is None:
            sample=gpu_snapshot()
            try:
                proc=psutil.Process(child.pid)
                sample['child_rss_bytes']=proc.memory_info().rss
                sample['child_cpu_seconds']=sum(proc.cpu_times()[:2])
            except psutil.Error:pass
            samples.append(sample);append(folder/'resources.jsonl',sample)
            time.sleep(5)
        returncode=child.wait()
    after=full_resources();write(folder/'resources_after.json',after)
    result={'status':'COMPLETE' if returncode==0 else 'FAILED','returncode':returncode,
            'utc':stamp(),'seconds':time.perf_counter()-start,'stage':stage,'worker':worker,
            'samples':len(samples),'gpu_peak_used_mib':max((s.get('used_mib',0) for s in samples),default=None),
            'gpu_min_free_mib':min((s.get('free_mib',float('inf')) for s in samples),default=None),
            'peak_child_rss_bytes':max((s.get('child_rss_bytes',0) for s in samples),default=None),
            'per_process_vram_not_inferred_from_host_delta':True,'automatic_retries':0}
    write(folder/'completion.json',result);append(OUT/'workflow'/'ledger.jsonl',dict(result,attempt=str(folder)))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if returncode:raise SystemExit(returncode)
    return result

def sample(args):
    jobs=Path(args.jobs_json).resolve();registration=OUT/'registration.json'
    if not jobs.is_file() or not registration.is_file():raise ValueError('Prepared jobs and registration required')
    snap=full_resources();g=snap['parsed_gpu']
    if g['returncode']!=0:raise RuntimeError('GPU resources unknown; do not dispatch')
    headroom=max(2048.,g['total_mib']*.1)
    needed=headroom+(1.2*args.observed_peak_mib if args.observed_peak_mib else 0.)
    if g['free_mib']<needed:raise RuntimeError('Insufficient free VRAM; no worker dispatched')
    # Per-cell exclusive ownership and completed-artifact checks live in generate.py.
    dispatch=OUT/'dispatch'/(args.worker+'__'+str(time.time_ns())+'.json')
    write(dispatch,{'status':'GO','utc':stamp(),'registration_sha256':sha(registration),
          'jobs_sha256':sha(jobs),'jobs_path':str(jobs),'headroom_mib':headroom,
          'minimum_free_mib':needed,'worker_id':args.worker,'workers_in_this_dispatch':1,
          'observed_comparable_process_peak_mib':args.observed_peak_mib or None,
          'first_authorized_task_measurement':not bool(args.observed_peak_mib),
          'resource_snapshot':snap,'scope':'one incremental GPU worker; no extra samples or retries'})
    cmd=[sys.executable,'-B','-u','-m',PACKAGE+'.generate','--jobs-json',str(jobs),'--dispatch',str(dispatch)]
    return observe_child(cmd,'RAW',args.worker,dispatch)

def status():
    counts={}
    for name,pattern in [('raw','raw/*/receipt.json'),('post','post/*/receipt.json'),('scores','scores/**/row.json')]:
        counts[name]=len(list(OUT.glob(pattern)))
    result={'utc':stamp(),'output':str(OUT),'counts':counts,'lambda_locked':(OUT/'lock.json').exists(),
            'training_updates':0,'resource':gpu_snapshot()}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

def _cpu_task(job):
    from . import evaluate as E
    phase,task=job
    result=E.score_raw(task) if phase=='raw' else E.post(task)
    print(json.dumps({'cpu_phase':phase,'cell_id':task['cell_id'],'status':result.get('status','COMPLETE')}),flush=True)
    return {'cell_id':task['cell_id'],'status':result.get('status','COMPLETE')}

def cpu(args):
    import concurrent.futures
    import multiprocessing
    import psutil
    from . import prepare as P
    from . import evaluate as E
    if args.phase=='post' and args.mode!='final':raise ValueError('No calibration D0/POST')
    if args.mode=='calibration':tasks=E._calibration_tasks()
    else:tasks=P.selected_tasks() if args.phase=='post' else P.load_tasks(resolved=True)
    before=full_resources();started=time.perf_counter()
    folder=OUT/'workflow'/'cpu'/('{}_{}__{}'.format(args.phase,args.mode,time.time_ns()))
    folder.mkdir(parents=True);write(folder/'resources_before.json',before)
    # First is an already-budgeted real score; reuse completed rows through the evaluator.
    first=_cpu_task((args.phase,tasks[0]))
    import resource
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    available=psutil.virtual_memory().available
    count=min(args.workers,max(1,int(available/(max(rss,256<<20)*1.2))),len(tasks)-1)
    if not 1<=count<=2:raise ValueError('This phase permits at most two CPU workers')
    write(folder/'dispatch.json',{'workers':count,'threads_per_worker':1,'first_parent_peak_rss_bytes':rss,
          'RAM_available_bytes':available,'tasks':len(tasks),'phase':args.phase,'mode':args.mode})
    jobs=[(args.phase,t) for t in tasks[1:]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=[first,*pool.map(_cpu_task,jobs)]
    done={'status':'COMPLETE','cells':len(results),'seconds':time.perf_counter()-started,'workers':count,
          'phase':args.phase,'mode':args.mode,'results':results,'model_calls':0,'automatic_retries':0}
    write(folder/'completion.json',done);write(folder/'resources_after.json',full_resources())
    print(json.dumps({k:v for k,v in done.items() if k!='results'}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('sample');q.add_argument('--jobs-json',required=True);q.add_argument('--worker',required=True)
    q.add_argument('--observed-peak-mib',type=float,default=0.)
    sub.add_parser('status');sub.add_parser('resources')
    q=sub.add_parser('prepare')
    q=sub.add_parser('evaluate');q.add_argument('options',nargs=argparse.REMAINDER)
    q=sub.add_parser('cpu');q.add_argument('--mode',choices=('calibration','final'),required=True)
    q.add_argument('--phase',choices=('raw','post'),required=True);q.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.command=='sample':sample(a)
    elif a.command=='status':status()
    elif a.command=='resources':
        path=OUT/'workflow'/('resources_'+str(time.time_ns())+'.json');write(path,full_resources());print(path)
    elif a.command=='prepare':observe_child([sys.executable,'-B','-u','-m',PACKAGE+'.prepare','prepare'],'PREPARE','cpu')
    elif a.command=='cpu':cpu(a)
    elif a.command=='evaluate':
        options=a.options[1:] if a.options[:1]==['--'] else a.options
        observe_child([sys.executable,'-B','-u','-m',PACKAGE+'.evaluate',*options],'EVALUATE','cpu_or_d0')
if __name__=='__main__':main()
