"""Finite registered dependency controller; never retries a failed science job.

This schedules only the already registered two-stage curriculum and72 paired
outputs. Visual review/report remain an explicit final task after READY_REVIEW.
"""
import argparse, json, os, sys, time
from .common import OUT, read, write_new, atomic_write, resources
from .run import launch, status

def ready_gpu(estimate=2800.):
    r=resources(); lines=r['gpu']['stdout'].strip().splitlines()
    if r['gpu']['returncode'] or len(lines)!=1: return False,r
    cells=lines[0].split(','); total,used,free,util=[float(s.strip().split()[0]) for s in cells[1:5]]
    r['decision']=dict(free_mib=free,compute_utilization=util,
        additional_peak_planning_mib=estimate,process_peak_is_measured=False,
        headroom_mib=max(2048.,.1*total),
        estimate_source='Conservative comparable worker estimate; actual WDDM per-process peak unavailable. Last first-worker allocator and aggregate device observations kept separately.')
    return free>=max(2048.,.1*total)+1.2*estimate and r['RAM_available']>=8*2**30,r

def finished(s,key):return s[key]['status']=='COMPLETE'
def fresh(s,key):return s[key]['status']=='NOT_STARTED'

def dispatch(key, estimate=None, explanation='Registered dependency satisfied'):
    if estimate is not None:
        ok,snap=ready_gpu(estimate)
        write_new(OUT/'controller/resources'/(str(time.time_ns())+'_'+key+'.json'),
                  dict(job=key,ready=ok,explanation=explanation,resources=snap))
        if not ok:return False
    launch(key)
    return True

def run():
    reg=read(OUT/'training_registration.json')
    assert reg['stage_updates']=={'NATIVE':6000,'SEAM':3000}
    assert reg['expected_new_raw']==72 and reg['expected_new_d0']==72
    folder=OUT/'controller';start=folder/'started.json'
    if start.exists():raise RuntimeError('Existing controller retained; inspect process before recovery')
    write_new(start,dict(pid=os.getpid(),started=time.time(),automatic_science_retries=False))
    while True:
        s=status()
        bad={k:v for k,v in s.items() if v['status'] in ('STOPPED_REQUIRES_INSPECTION','INTERRUPTED_SAFE','FAILED','OOM')}
        if bad:
            atomic_write(folder/'status.json',dict(status='STOPPED_REQUIRES_INSPECTION',jobs=bad,time=time.time()))
            return
        native=read(OUT/'training/NATIVE/status.json') if (OUT/'training/NATIVE/status.json').exists() else {}
        seam=read(OUT/'training/SEAM/status.json') if (OUT/'training/SEAM/status.json').exists() else {}
        atomic_write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid(),
            native_step=native.get('stage_step'),seam_step=seam.get('stage_step'),
            jobs={k:v['status'] for k,v in s.items()}))
        if finished(s,'native') and fresh(s,'seam'):
            dispatch('seam',4500.,'Native complete; restore identical generator/AdamW/RNG for registered SEAM3000')
        elif fresh(s,'raw-native') and finished(s,'native') and (seam.get('stage_step',0)>=25 or finished(s,'seam')):
            dispatch('raw-native',2800.,'SEAM first25 formal updates measured; independent frozen Native24 sampling')
        elif finished(s,'seam') and fresh(s,'raw-seam'):
            dispatch('raw-seam',2800.,'Registered SEAM endpoint complete; frozen24 requests')
        elif all(finished(s,k) for k in ('raw-common','raw-native','raw-seam')):
            if fresh(s,'score-raw'):
                dispatch('score-raw',explanation='All72 RAW saved; independent CPU scoring,4 workers with memory cap')
            # D0 loading peaks are serialized; these short single-label jobs are
            # overlapped with CPU scoring, not with another large D0 load.
            d0=('d0-common','d0-native','d0-seam')
            running=any(s[k]['status'] in ('LAUNCHED','RUNNING') for k in d0)
            nxt=next((k for k in d0 if fresh(s,k)),None)
            if not running and nxt:
                dispatch(nxt,6000.,'Single D0 model owner; next label only after prior resource release;CPU scoring overlaps')
            if all(finished(s,k) for k in d0) and fresh(s,'score-post'):
                dispatch('score-post',explanation='All72 D0/GEO products saved;4 CPU workers')
            if finished(s,'score-raw') and finished(s,'score-post') and fresh(s,'collect'):
                dispatch('collect')
            if finished(s,'collect') and fresh(s,'gallery'):
                dispatch('gallery')
            if finished(s,'gallery'):
                atomic_write(folder/'status.json',dict(status='READY_REVIEW',time=time.time(),jobs={k:v['status'] for k,v in s.items()},
                    training_complete=True,scoring_complete=True,actual_visual_review_complete=False))
                return
        time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:run()
    else:print(json.dumps(read(OUT/'controller/status.json') if (OUT/'controller/status.json').exists() else {'status':'NOT_STARTED'},indent=2))
