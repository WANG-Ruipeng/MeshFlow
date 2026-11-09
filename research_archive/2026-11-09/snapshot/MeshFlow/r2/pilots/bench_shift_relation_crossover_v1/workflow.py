"""One finite wave, incremental measured sampling dispatch, no silent retries."""
import argparse, json, math, os, subprocess, sys, time, traceback
from .common import ROOT, OUT, LABEL, read, write_new, atomic_write, resources, sha
from .eval_plan import all_raw_barrier, verify, plan
MODULE = 'r2.pilots.bench_shift_relation_crossover_v1.'
W = OUT / 'workflow'

def gpu_numbers(snapshot):
    line = snapshot['gpu']['stdout'].strip().splitlines()
    if snapshot['gpu']['returncode'] or len(line) != 1: raise RuntimeError('GPU resources unavailable')
    values = line[0].split(',')
    return [float(v.strip().split()[0]) for v in values[1:5]]

def execute():
    gate = read(OUT/'blind_review/gate.json')
    assert gate['status'] == 'PASS' and gate['protocol_sha256'] == sha(OUT/'PROTOCOL.md')
    reg = plan(); verify(reg['protected']); verify(reg['source_hashes'])
    assert read(OUT/'cpu_gate.json')['status'] == 'PASS'
    if (W/'state.json').exists(): raise RuntimeError('Workflow already started; no automatic replay')
    W.mkdir(parents=True, exist_ok=True)
    state = dict(status='RUNNING',pid=os.getpid(),started=time.time(),completed=[],max_sampling_workers=1)
    atomic_write(W/'state.json',state)
    def job(name,module,args=()):
        snap=resources();write_new(W/(name+'_resources.json'),snap)
        command=[sys.executable,'-B','-m',MODULE+module,*args]
        with (W/(name+'.log')).open('xb') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(stage=name,latest_pid=p.pid);atomic_write(W/'state.json',state)
        return name,p,time.monotonic(),time.time()
    def finish(task):
        name,p,tick,start=task; code=p.wait()
        result=dict(stage=name,pid=p.pid,exit_code=code,seconds=time.monotonic()-tick,started_unix=start,finished_unix=time.time())
        write_new(W/(name+'_done.json'),result)
        if code: raise RuntimeError(f'{name} failed with code {code}; no automatic retry')
        state['completed'].append(result);atomic_write(W/'state.json',state)
    first = second = None
    try:
        first=job('raw_shard0','sample',('--label',LABEL,'--shard','0'))
        observation=OUT/'evaluation'/LABEL/'sampling_shard_0/observed_peak_output1.json'
        while not observation.exists() and first[1].poll() is None: time.sleep(.5)
        snap=resources();total,used,free,util=gpu_numbers(snap)
        peak=read(observation)['summary'] if observation.exists() else {}
        # Under WDDM the full device peak is a conservative bound, not a claim
        # that all memory belongs to this worker. Keep the previous 2.8GiB plan floor.
        bound=max(2800.,float(peak.get('total_device_used_peak_mib') or 0))
        headroom=max(2048.,.1*total)
        slots=math.floor(max(0.,free-headroom)/(1.2*bound))
        parallel=(first[1].poll() is None and slots>=1 and util<85 and snap['RAM_available']>=8*(1<<30) and snap['CPU_cores']>=4)
        write_new(W/'incremental_dispatch.json',dict(resources=snap,first_real_peak=peak,
            observed_total_device_bound_mib=bound,process_peak_attribution='NA_WDDM; conservative whole-device bound plus prior planning floor',
            headroom_mib=headroom,incremental_slots=slots,concurrent_second_worker=parallel,
            reason='Two authorized independent request shards; incremental launch after real output peak. Compute saturation or resource limit forces serial.',
            extra_probe_calls=0))
        if parallel:
            second=job('raw_shard1','sample',('--label',LABEL,'--shard','1','--estimated-peak-mib',str(bound)))
            state['max_sampling_workers']=2;atomic_write(W/'state.json',state)
            try: finish(first)
            finally: finish(second)
        else:
            finish(first)
            second=job('raw_shard1','sample',('--label',LABEL,'--shard','1','--estimated-peak-mib',str(bound)))
            finish(second)
        barrier=all_raw_barrier()
        write_new(W/'all_raw_complete.json',dict(status='PASS',outputs=48,new_forwards=1200,
            files={str(p):sha(p) for p in barrier}))
        cpu=job('score_raw','evaluation',('raw','--workers','2'))
        try: finish(job('D0','post_d0',('--label',LABEL)))
        finally: finish(cpu)
        finish(job('score_post','evaluation',('post','--workers','4')))
        finish(job('collect','collect'))
        finish(job('gallery','gallery',('run','--workers','4')))
        state.update(status='NUMERICS_AND_GALLERY_COMPLETE_PENDING_ACTUAL_REVIEW',finished=time.time())
    except BaseException as exc:
        state.update(status='STOPPED_REQUIRES_INSPECTION',error=repr(exc),traceback=traceback.format_exc(),automatic_replay=False,finished=time.time())
        raise
    finally:
        atomic_write(W/'state.json',state)
        write_new(W/'resources_finished.json',resources())

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute: execute()
    else: print(json.dumps(read(W/'state.json') if (W/'state.json').exists() else dict(status='NOT_STARTED')))
