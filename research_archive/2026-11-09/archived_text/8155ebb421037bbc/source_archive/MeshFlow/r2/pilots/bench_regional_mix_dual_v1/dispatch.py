"""Bounded incremental CPU dispatch of the frozen jobs; no automatic retries."""
from . import common as C
import argparse, json, os, subprocess, sys, time
from r2.pilots.bench_four_routes_v1.resources import snapshot


def receipt(job):
    return C.OUT/'runs'/job['job_id']/'complete.json'


def command(job):
    return [sys.executable,'-B','-m','r2.pilots.bench_regional_mix_dual_v1.runtime','run-one',
            '--jobs',str(C.OUT/'jobs.json'),'--job-id',job['job_id']]


def run():
    from . import runtime
    reg=C.read(C.OUT/'jobs.json');C.verify(reg['protected']);C.verify(reg['source_hashes'])
    folder=C.OUT/'dispatch';folder.mkdir(parents=True,exist_ok=True)
    ledger=folder/'events.jsonl'
    def event(kind,**kw):
        row=dict(utc=C.utc(),kind=kind,**kw)
        with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(C.clean(row),ensure_ascii=False)+'\n')
        print(json.dumps(C.clean(row),ensure_ascii=False),flush=True)
    existing=[j for j in reg['jobs'] if receipt(j).exists()]
    pending=[j for j in reg['jobs'] if not receipt(j).exists()]
    registration_hash=C.sha(C.OUT/'jobs.json')
    for job in existing:
        prior=runtime.verify_result(C.read(receipt(job)))
        if prior['registration_sha256']!=registration_hash or prior['job_sha256']!=runtime.canonical_hash(job):
            raise ValueError('Existing receipt identity differs: '+job['job_id'])
    for j in pending:
        if (C.OUT/'runs'/j['job_id']).exists():raise RuntimeError('Partial job; inspect, do not retry: '+j['job_id'])
    if not pending:
        event('ALREADY_COMPLETE_NO_REPLAY',jobs=len(existing));return
    dispatch_registration=folder/'registration.json'
    if dispatch_registration.exists():
        if C.read(dispatch_registration)['jobs_sha256']!=registration_hash:raise ValueError('Dispatch registration changed')
        event('CONTINUE_UNSTARTED_ONLY',verified_completed=[j['job_id'] for j in existing])
    else:
        C.write(dispatch_registration,dict(jobs_sha256=registration_hash,maximum_workers=2,
            threads_per_worker=1,initial_resources=snapshot(),resumed_complete_jobs=[j['job_id'] for j in existing]))
    started=time.perf_counter();logs=[];active={}
    def launch(job,phase):
        path=folder/(job['job_id']+'.log')
        if path.exists():raise RuntimeError('Attempt log exists: '+str(path))
        log=path.open('xb');logs.append(log)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
        p=subprocess.Popen(command(job),cwd=C.ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        active[p.pid]=(p,job,log,time.perf_counter())
        event('START',job_id=job['job_id'],pid=p.pid,phase=phase,active_workers=len(active))
    # Reused receipts do not estimate scientific workspace. Measure the first physical job.
    physical=[j for j in pending if not (j.get('reuse_of') or j.get('reuse_of_external') or j.get('reuse_original_d0'))]
    first=physical[0] if physical else pending[0]
    pending.remove(first);launch(first,'FIRST_AUTHORIZED_JOB_MEASUREMENT')
    pid=next(iter(active));p,j,log,tick=active[pid]
    while p.poll() is None:time.sleep(1)
    log.close();active.pop(pid)
    if p.returncode:raise RuntimeError('First job failed; no retries: '+j['job_id'])
    if not receipt(j).exists():raise RuntimeError('Missing first completion receipt')
    r=C.read(receipt(j));after=snapshot()
    peak_kib=r.get('peak_RSS_KiB',r.get('peak_rss_kib',0))
    peak_bytes=float(peak_kib)*1024
    if peak_bytes<=0:workers=1
    else:
        reserve=max(2*(1<<30),.1*after['ram']['total'])
        memory_workers=int(max(0,after['ram']['available']-reserve)//(1.2*peak_bytes))
        workers=min(2,memory_workers,max(1,after['cpu_count']//2))
        if workers<1:raise RuntimeError('Insufficient resource headroom after measured job')
    C.write(folder/'measured_dispatch.json',dict(first_job=j['job_id'],first_peak_RSS_KiB=peak_kib,
        chosen_workers=workers,resources_after_first=after,peak_scope='fresh subprocess lifetime RSS, includes all CPU scientific phases',
        parallel_wall_times_not_isolated_benchmarks=True))
    event('COMPLETE',job_id=j['job_id'],seconds=time.perf_counter()-tick,workers_next=workers)
    failed=[]
    while pending or active:
        # Dependencies are immutable cached aliases, so launch only after owner completion.
        while len(active)<workers and pending and not failed:
            ready=next((x for x in pending if not x.get('reuse_of') or (C.OUT/'runs'/x['reuse_of']/'complete.json').exists()),None)
            if ready is None:break
            current=snapshot()
            margin=max(2*(1<<30),.1*current['ram']['total'])
            if peak_bytes>0 and current['ram']['available']-margin<1.2*peak_bytes:
                event('WAIT_MEMORY_HEADROOM',resources=current,observed_peak_bytes=peak_bytes)
                break
            event('RESOURCE_BEFORE_DISPATCH',resources=current,observed_peak_bytes=peak_bytes)
            pending.remove(ready);launch(ready,'BOUNDED_CPU_POOL')
        for pid,(p,j,log,tick) in list(active.items()):
            if p.poll() is not None:
                log.close();active.pop(pid)
                if p.returncode or not receipt(j).exists():failed.append(j['job_id'])
                elif not (j.get('reuse_of') or j.get('reuse_of_external') or j.get('reuse_original_d0')):
                    peak_bytes=max(peak_bytes,float(C.read(receipt(j)).get('peak_RSS_KiB',0))*1024)
                event('FAILED' if j['job_id'] in failed else 'COMPLETE',job_id=j['job_id'],returncode=p.returncode,
                      seconds=time.perf_counter()-tick,resources_after=snapshot())
        if failed and not active:raise RuntimeError('Failure preserved, remaining jobs not dispatched: '+str(failed))
        if pending and not active and not any(not x.get('reuse_of') or (C.OUT/'runs'/x['reuse_of']/'complete.json').exists() for x in pending):
            raise RuntimeError('Unresolved alias dependency')
        if active or pending:time.sleep(1)
    C.verify(reg['protected']);C.verify(reg['source_hashes'])
    C.write(folder/'complete.json',dict(status='COMPLETE',logical_jobs=len(reg['jobs']),budget=reg['budget'],
        seconds=time.perf_counter()-started,workers=workers,resources_final=snapshot(),
        completions={str(receipt(j)):C.sha(receipt(j)) for j in reg['jobs']}))
    event('ALL_COMPLETE',logical_jobs=len(reg['jobs']),seconds=time.perf_counter()-started)


if __name__=='__main__':run()
