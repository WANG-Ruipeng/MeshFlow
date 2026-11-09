"""Incremental measured CPU dispatch; only registered jobs, no silent retries."""
from __future__ import annotations
import json,os,subprocess,sys,time
from . import runtime as R

def run():
    reg,rh,by_id=R.load_jobs();folder=R.OUT/'dispatch';folder.mkdir(exist_ok=True)
    done=folder/'complete.json'
    if done.exists():
        old=R.read(done);R.verify(old['completions'])
        if old['registration_sha256']!=rh:raise ValueError('Dispatch identity changed')
        return old
    pending=[];completed=[]
    for j in reg['jobs']:
        d=R.OUT/'runs'/j['job_id'];p=d/'complete.json'
        if p.exists():R.verify_result(R.read(p));completed.append(j['job_id'])
        elif d.exists():raise RuntimeError('Partial scientific attempt must be reviewed before dispatch: '+j['job_id'])
        else:pending.append(j)
    event_path=folder/'events.jsonl'
    def event(kind,**kw):
        row=dict(utc=R.utc(),kind=kind,**kw)
        with event_path.open('a',encoding='utf-8') as f:f.write(json.dumps(R.clean(row),ensure_ascii=False)+'\n')
        print(json.dumps(R.clean(row),ensure_ascii=False),flush=True)
    active={};tick=time.perf_counter();peak=0;workers=1
    initial=R.resources();event('RESOURCES_INITIAL',resources=initial,already_completed=completed)
    if not (R.OUT/'resources_initial.json').exists():R.write(R.OUT/'resources_initial.json',initial)
    def launch(job,phase):
        snap=R.resources();reserve=max(2*(1<<30),.1*snap['ram_total_bytes'])
        if peak and snap['ram_available_bytes']-reserve<1.2*peak:return False
        logpath=folder/(job['job_id']+'.log')
        if logpath.exists():raise RuntimeError('Existing attempt log: '+str(logpath))
        log=logpath.open('xb');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
        p=subprocess.Popen([sys.executable,'-B','-m','r2.pilots.bench_local_shape_mix_v1.runtime','run-one','--job-id',job['job_id']],cwd=R.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[p.pid]=(p,job,log,time.perf_counter());event('START',job_id=job['job_id'],pid=p.pid,phase=phase,resources=snap,active_workers=len(active));return True
    def collect(pid):
        nonlocal peak
        p,j,log,start=active.pop(pid);log.close();path=R.OUT/'runs'/j['job_id']/'complete.json'
        if p.returncode or not path.exists():
            event('FAILED',job_id=j['job_id'],returncode=p.returncode);return False
        result=R.verify_result(R.read(path));peak=max(peak,result['peak_RSS_KiB']*1024);completed.append(j['job_id'])
        event('COMPLETE',job_id=j['job_id'],seconds=time.perf_counter()-start,peak_RSS_KiB=result['peak_RSS_KiB'],resources_after=R.resources());return True
    if pending:
        first=pending.pop(0);launch(first,'FIRST_AUTHORIZED_JOB_MEASUREMENT');pid=next(iter(active))
        while active[pid][0].poll() is None:time.sleep(1)
        if not collect(pid):raise RuntimeError('First job failed; no retry')
        snap=R.resources();reserve=max(2*(1<<30),.1*snap['ram_total_bytes'])
        slots=int(max(0,snap['ram_available_bytes']-reserve)//(1.2*peak)) if peak else 1
        workers=min(2,slots,max(1,snap['cpu_logical']//2),max(1,len(pending)))
        if workers<1:raise RuntimeError('No measured resource slot')
        event('MEASURED_DISPATCH',workers=workers,observed_process_peak_bytes=peak,resource_scope='CPU fresh subprocess lifetime',concurrent_wall_times_not_isolated_benchmarks=True)
    failed=[]
    while pending or active:
        while pending and len(active)<workers and not failed:
            if not launch(pending[0],'INCREMENTAL_CPU'):break
            pending.pop(0)
        for pid,(p,_,_,_) in list(active.items()):
            if p.poll() is not None and not collect(pid):failed.append(pid)
        if failed and not active:raise RuntimeError('Failure retained; remaining unstarted')
        if pending and not active:raise RuntimeError('No memory slot; unstarted jobs retained')
        if pending or active:time.sleep(1)
    R.verify(reg['protected']);R.verify(reg['source_hashes'])
    result=dict(status='COMPLETE',registration_sha256=rh,jobs=len(completed),workers=workers,seconds=time.perf_counter()-tick,
        peak_worker_RSS_bytes=peak,resources_final=R.resources(),budget=reg['budget'],
        completions={str(R.OUT/'runs'/j['job_id']/'complete.json'):R.sha(R.OUT/'runs'/j['job_id']/'complete.json') for j in reg['jobs']})
    R.write(done,result);event('ALL_COMPLETE',jobs=len(completed),seconds=result['seconds']);return result

if __name__=='__main__':run()
