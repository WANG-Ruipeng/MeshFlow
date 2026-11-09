"""Explicit completed-cascade handoff and four disjoint CPU-only job lists.

No scientific function is changed. Only the two registered old PIDs may be
stopped, and they are terminated only after every started owned cascade has a
COMPLETE receipt. Any incomplete cascade is resumed immediately. No SIGKILL.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

from . import prepare as P, evaluate as E
from .run import full_resources, observe_child, write, append

VERSION='completed_cascade_handoff_four_disjoint_CPU_workers_v1'
MODULE='r2.pilots.bench_full_prior_motif_guidance_v1.post_handoff_four'
DEST=P.OUT/'workflow/post_four_v1'
OLD={0:6050,1:5564}
ORIGINAL_MODULE='r2.pilots.bench_full_prior_motif_guidance_v1.post_ready_cached'
SOURCE_BARRIER=P.OUT/'workflow/ready_post_cached_0/initial_full_RAW_verification.json'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify(files):
    for p,h in files.items():
        if P.sha(p)!=h:raise ValueError('Immutable evidence changed: '+str(p))


def statbook(paths):
    return {str(P.resolve(p)):[P.resolve(p).stat().st_size,P.resolve(p).stat().st_mtime_ns] for p in paths}


def guard(stats):
    for p,expected in stats.items():
        s=P.resolve(p).stat()
        if [s.st_size,s.st_mtime_ns]!=expected:raise ValueError('Frozen file stat changed: '+p)


def initial():
    import psutil
    path=DEST/'handoff_registration.json'
    if path.exists():
        saved=P.read(path)
        if saved['source_sha256']!=P.sha(__file__):raise ValueError('Handoff source changed')
        return saved
    procs={}
    for partition,pid in OLD.items():
        proc=psutil.Process(pid);cmd=proc.cmdline()
        if ORIGINAL_MODULE not in cmd or cmd[-2:]!=['--partition',str(partition)]:raise ValueError('Not the authorized old worker')
        procs[str(partition)]=dict(pid=pid,create_time=proc.create_time(),command=cmd)
    saved=dict(status='REGISTERED',utc=utc(),version=VERSION,source_sha256=P.sha(__file__),old_processes=procs,
        resources=full_resources(),selected_tasks_sha256=P.sha(P.TASKS),selected_outputs=184,
        requested_workers=4,threads_per_worker=1,scientific_changes=False,
        policy='Stop only verified old PID, inspect all started owned tasks, immediately CONT if any incomplete; TERM+CONT only at a completed-cascade boundary')
    write(path,saved);return saved


def process_identity(expected):
    import psutil
    try:proc=psutil.Process(expected['pid'])
    except psutil.NoSuchProcess:return None
    if proc.status()==psutil.STATUS_ZOMBIE:return None
    if proc.create_time()!=expected['create_time'] or proc.cmdline()!=expected['command']:
        raise ValueError('PID identity changed; no signal allowed')
    return proc


def owned_state(partition):
    """No hashes or scientific computation while a process is briefly stopped."""
    completed=[];incomplete=[];receipt_hashes={}
    for i,t in enumerate(P.selected_tasks()):
        if i%2!=partition:continue
        cell=t['cell_id'];base=P.OUT/'base_pair_caches'/cell;cpu=P.OUT/'cpu'/cell
        if not base.exists() and not cpu.exists():continue
        bp=base/'complete.json';cp=cpu/'complete.json'
        ok=bp.is_file() and cp.is_file()
        if ok:
            b=P.read(bp);c=P.read(cp)
            ok=b.get('status')=='COMPLETE' and b.get('full_pair_scans')==1 and c.get('status')=='COMPLETE'
            ok=ok and c.get('request_id')==t['request_id'] and c.get('expanded_base_scores')==1 and c.get('final_scores')==1
        if ok:
            completed.append(cell)
            receipt_hashes[str(bp)]=P.sha(bp);receipt_hashes[str(cp)]=P.sha(cp)
        else:incomplete.append(cell)
    return dict(partition=partition,started_tasks=len(completed)+len(incomplete),completed=completed,
                incomplete=incomplete,receipt_hashes=receipt_hashes)


def handoff(timeout=50.):
    import psutil
    if not 0<timeout<=60:raise ValueError('A handoff polling wave is bounded to sixty seconds')
    registration=initial();stopped=set();start=time.monotonic()
    def interrupted(signum,frame):raise InterruptedError('Coordinator interrupted; restoring stopped owned workers')
    prior={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while time.monotonic()-start<timeout:
            for partition in OLD:
                path=DEST/f'handoff_{partition}.json'
                if path.exists():continue
                expected=registration['old_processes'][str(partition)];proc=process_identity(expected)
                if proc is None:
                    state=owned_state(partition)
                    if state['incomplete']:raise RuntimeError('Old worker exited with incomplete work; no replay')
                    write(path,dict(status='ALREADY_EXITED_SAFE',utc=utc(),process=expected,**state));continue
                # Observe first; a running cascade never needs to be stopped just to find it incomplete.
                preview=owned_state(partition)
                if preview['incomplete']:continue
                proc=process_identity(expected)
                if proc is None:continue
                stopped_at=time.perf_counter();os.kill(proc.pid,signal.SIGSTOP);stopped.add(proc.pid)
                try:
                    deadline=time.monotonic()+2
                    while proc.status()!=psutil.STATUS_STOPPED:
                        if time.monotonic()>deadline:raise RuntimeError('Cannot confirm process stopped')
                        time.sleep(.01)
                    state=owned_state(partition)
                    if state['incomplete']:
                        append(DEST/'boundary_observations.jsonl',dict(utc=utc(),pid=proc.pid,status='RESUMED_INCOMPLETE',
                               incomplete=state['incomplete'],paused_seconds=time.perf_counter()-stopped_at))
                        continue
                    receipt=dict(status='SAFE_BOUNDARY_HANDOFF',utc=utc(),process=expected,**state,
                        paused_seconds_before_termination=time.perf_counter()-stopped_at,
                        old_observer_expected_returncode=-15,
                        reason='Authorized orchestration handoff after all started owned cascades completed; no model/score replay')
                    write(path,receipt)
                    os.kill(proc.pid,signal.SIGTERM)
                    os.kill(proc.pid,signal.SIGCONT);stopped.discard(proc.pid)
                    try:proc.wait(timeout=3)
                    except psutil.TimeoutExpired:raise RuntimeError('TERM did not finish; no new workers may start')
                    print(json.dumps(dict(status='HANDOFF',partition=partition,pid=proc.pid,completed=len(state['completed']))),flush=True)
                finally:
                    if proc.pid in stopped:
                        try:os.kill(proc.pid,signal.SIGCONT)
                        except ProcessLookupError:pass
                        stopped.discard(proc.pid)
            if all((DEST/f'handoff_{p}.json').is_file() for p in OLD):
                return dict(status='BOTH_HANDOFFS_COMPLETE',seconds=time.monotonic()-start)
            time.sleep(.15)
        return dict(status='WAITING_SAFE_BOUNDARY_ALL_UNFINISHED_WORKERS_RUNNING',seconds=time.monotonic()-start,
                    handed_off=[p for p in OLD if (DEST/f'handoff_{p}.json').exists()])
    finally:
        for pid in stopped:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        for sig,handler in prior.items():signal.signal(sig,handler)


def barrier_cache():
    """Authenticate all 403 protected files in the prior full verification."""
    receipt=P.read(SOURCE_BARRIER);tasks=P.load_tasks(resolved=True)
    expected={P.resolve(t[k]) for t in tasks for k in ('raw_path','raw_receipt_path')}
    expected|={P.TASKS,P.REQUESTS,P.OUT/'lock.json'}
    if receipt.get('status')!='PASS' or receipt.get('outputs')!=200 or len(receipt['files'])!=403:
        raise ValueError('Prior full verification schema mismatch')
    if {P.resolve(p) for p in receipt['files']}!=expected:raise ValueError('403-file protection set differs')
    before=statbook(receipt['files']);verify(receipt['files']);guard(before)
    return (tasks,receipt),before


def install_cache():
    saved,stats=barrier_cache();original=P.verify_raw_barrier
    def cached():guard(stats);return saved
    P.verify_raw_barrier=cached
    return saved,stats,original


def completed(task):
    """Authenticate original producer identities/files without calling a scorer."""
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    cell=task['cell_id'];base=P.OUT/'base_pair_caches'/cell;cpu=P.OUT/'cpu'/cell
    if not (cpu/'complete.json').is_file():
        if base.exists() or cpu.exists():raise RuntimeError('Incomplete started task blocks repartition: '+cell)
        return None
    E._post_proxy()._checked_post(task,P.TASKS)
    cache=P.read(base/'complete.json')
    if cache.get('status')!='COMPLETE' or cache.get('full_pair_scans')!=1:raise ValueError('BASE cache incomplete')
    if cache['mesh_sha256']!=P.sha(P.OUT/'post'/cell/'post.npz'):raise ValueError('BASE cache identity changed')
    verify(cache['files'])
    paths=[P.OUT/'post'/cell/'post.npz',P.OUT/'post'/cell/'pre_geo.npz',P.resolve(task['raw_path']),base/'pairs.json']
    identity=dict(version=CPU.VERSION,label=task['label'],request_id=task['q']['request_id'],
                  registration_sha256=P.sha(P.TASKS),seam_model=False,sources={str(p):P.sha(p) for p in paths})
    receipt=P.read(cpu/'complete.json')
    if receipt.get('status')!='COMPLETE' or receipt['identity']!=identity:raise ValueError('CPU complete identity mismatch')
    for key,value in (('model_calls',0),('D0_calls',0),('GPU_calls',0),('new_noise',0),('new_queries',0),
                      ('expanded_base_scores',1),('final_scores',1)):
        if receipt.get(key)!=value:raise ValueError('CPU complete count mismatch: '+key)
    if [s['stage'] for s in receipt['stages']]!=['HINGE_OPP015','ALPHA','CLEAN_FINAL_POST']:
        raise ValueError('CPU stage chain changed')
    verify(receipt['files']);return dict(cell_id=cell,path=str(cpu/'complete.json'),sha256=P.sha(cpu/'complete.json'),seconds=receipt['seconds'])


def plan():
    registration=initial();path=DEST/'plan.json'
    if path.exists():return P.read(path)
    for part in OLD:
        if not (DEST/f'handoff_{part}.json').is_file():raise RuntimeError('Wait for both handoffs')
        if process_identity(registration['old_processes'][str(part)]) is not None:raise RuntimeError('Old worker still active')
    E.freeze_evaluation_code()
    saved,stats=barrier_cache();tasks=P.selected_tasks();done=[];pending=[]
    if len(tasks)!=184 or len({t['cell_id'] for t in tasks})!=184:raise ValueError('Selected184 changed')
    for t in tasks:
        result=completed(t)
        if result:done.append(result)
        else:pending.append(t)
    # Deterministic round robin after sorting by fixed face count distributes heavy cases.
    pending.sort(key=lambda t:(-int(t['q']['N']),t['cell_id']))
    partitions=[pending[i::4] for i in range(4)]
    for i,jobs in enumerate(partitions):write(DEST/f'jobs_{i}.json',dict(partition=i,task_ids=[t['cell_id'] for t in jobs]))
    result=dict(status='FROZEN',utc=utc(),source_sha256=P.sha(__file__),tasks_sha256=P.sha(P.TASKS),
        selected_tasks=184,already_completed=done,pending_tasks=len(pending),worker_count=4,
        partition_policy='Descending frozen N, deterministic cell ID tie, round robin four disjoint lists',
        partitions={str(i):[t['cell_id'] for t in jobs] for i,jobs in enumerate(partitions)},
        barrier_source=str(SOURCE_BARRIER),barrier_source_sha256=P.sha(SOURCE_BARRIER),protected_files=403,
        initial_403_full_hashes_verified=True,barrier_stats=stats,resources=full_resources(),
        evaluator_sha256=P.sha(E.__file__),prepare_sha256=P.sha(P.__file__),model_calls=0,scoring_replays=0)
    write(path,result);return result


def worker(partition):
    import resource
    p=P.read(DEST/'plan.json')
    if p['source_sha256']!=P.sha(__file__):raise ValueError('Orchestration changed')
    folder=DEST/f'worker_{partition}'
    if folder.exists():raise RuntimeError('Existing worker attempt; no implicit retry')
    folder.mkdir(parents=True)
    saved,stats,original=install_cache()
    tasks={t['cell_id']:t for t in P.selected_tasks()};ids=p['partitions'][str(partition)]
    if P.read(DEST/f'jobs_{partition}.json')['task_ids']!=ids:raise ValueError('Joblist changed')
    write(folder/'initial_verification.json',dict(status='PASS',files=403,full_hashes=True,
        source=str(SOURCE_BARRIER),source_sha256=P.sha(SOURCE_BARRIER),utc=utc()))
    write(folder/'started.json',dict(pid=os.getpid(),partition=partition,task_ids=ids,utc=utc(),
        source_sha256=P.sha(__file__),threads=1,model_calls=0,scientific_functions_modified=False))
    start=time.perf_counter();results=[]
    try:
        for cell in ids:
            guard(stats);tick=time.perf_counter();cpu=time.process_time()
            # E.post retains its original per-output D0/RAW identities and producer cache checks.
            result=E.post(tasks[cell])
            row=dict(cell_id=cell,status=result['status'],utc=utc(),seconds=time.perf_counter()-tick,
                     cpu_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     completed=len(results)+1,total=len(ids),model_calls=0)
            results.append(row);append(folder/'progress.jsonl',row);print(json.dumps(row),flush=True)
        guard(stats)
        write(folder/'complete.json',dict(status='COMPLETE',partition=partition,outputs=len(ids),seconds=time.perf_counter()-start,
            results=results,model_calls=0,scoring_replays=0,final_global_full_barrier_required=True,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except BaseException as exc:
        write(folder/'failure.json',dict(status='STOPPED_NO_REPLAY',utc=utc(),error=repr(exc),completed=results));raise
    finally:P.verify_raw_barrier=original


def observe(partition):
    return observe_child([sys.executable,'-B','-u','-m',MODULE,'worker','--partition',str(partition)],
                         'CPU_POST_FOUR',f'partition_{partition}',DEST/'plan.json')


def launch(partition):
    import psutil
    p=P.read(DEST/'plan.json');path=DEST/f'launch_{partition}.json'
    if path.exists():raise RuntimeError('Already launched; no implicit retry')
    if p['source_sha256']!=P.sha(__file__):raise ValueError('Plan source changed')
    for part,old in initial()['old_processes'].items():
        if process_identity(old) is not None:raise RuntimeError('Old worker still active')
    resource_snapshot=full_resources();ram=psutil.virtual_memory().available;cpu=psutil.cpu_percent(interval=1)
    if ram<8*(1024**3) or (psutil.cpu_count() or 0)<8 or cpu>85:
        raise RuntimeError('CPU/RAM headroom insufficient; do not launch')
    if partition:
        previous=DEST/f'worker_{partition-1}'/'started.json'
        if not previous.exists():raise RuntimeError('Prior worker must finish real initialization before increment')
        previous_pid=P.read(previous)['pid']
        try:previous_rss=psutil.Process(previous_pid).memory_info().rss
        except psutil.NoSuchProcess:previous_rss=0
    else:previous_rss=None
    log=DEST/f'observer_{partition}.log'
    with log.open('x') as handle:
        proc=subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'observe','--partition',str(partition)],
                              cwd=P.ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(status='LAUNCHED',utc=utc(),partition=partition,observer_pid=proc.pid,source_sha256=P.sha(__file__),
        available_RAM_bytes=ram,host_CPU_percent=cpu,previous_worker_RSS_bytes=previous_rss,
        resources=resource_snapshot,incremental=True,model_calls=0)
    write(path,result);return result


def finish():
    path=DEST/'complete.json'
    if path.exists():
        r=P.read(path);verify(r['files']);return r
    p=P.read(DEST/'plan.json');workers=[]
    for i in range(4):
        w=P.read(DEST/f'worker_{i}'/'complete.json')
        if w['status']!='COMPLETE' or w['outputs']!=len(p['partitions'][str(i)]):raise ValueError('Worker incomplete')
        workers.append(w)
    # Exactly one global full validation after all new workers, rather than four repeated full audits.
    actual=P.verify_raw_barrier()
    if actual[1]!=P.read(SOURCE_BARRIER):raise ValueError('Final full RAW verification differs')
    completed_rows=[completed(t) for t in P.selected_tasks()]
    if any(r is None for r in completed_rows):raise ValueError('Missing complete original CPU output')
    resources=full_resources();results=[r for w in workers for r in w['results']]
    first=min(P.read(DEST/f'launch_{i}.json')['utc'] for i in range(4))
    wall=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(first)).total_seconds()
    result=dict(status='COMPLETE',utc=utc(),source_sha256=P.sha(__file__),outputs=184,
        previous_completed=len(p['already_completed']),newly_completed=len(results),workers=4,
        elapsed_since_first_launch_seconds=wall,outputs_per_minute=len(results)/wall*60,
        summed_per_task_seconds=sum(r['seconds'] for r in results),summed_CPU_seconds=sum(r['cpu_seconds'] for r in results),
        final_global_full_RAW_verification=True,model_calls=0,scoring_replays=0,resources=resources,
        files={r['path']:r['sha256'] for r in completed_rows},
        timing_note='Concurrent throughput for this registered remaining task mix; not an isolated compute-speed comparison.')
    write(path,result);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('handoff','plan','launch','observe','worker','finish'))
    ap.add_argument('--partition',type=int,choices=range(4));ap.add_argument('--timeout',type=float,default=50.)
    a=ap.parse_args()
    if a.action=='handoff':r=handoff(a.timeout)
    elif a.action=='plan':r=plan()
    elif a.action=='finish':r=finish()
    else:
        if a.partition is None:ap.error('partition is required')
        r=globals()[a.action](a.partition)
    if r is not None:print(json.dumps({k:v for k,v in r.items() if k not in ('files','resources','partitions','already_completed','barrier_stats')}),flush=True)


if __name__=='__main__':main()
