"""Finite-stage recoverable launcher: existing launch records are never replayed."""
import argparse, json, os, subprocess, sys, time, traceback
from .common import ROOT, OUT, MODULE, read, write_new, atomic_write, resources

JOBS = {
    'native': ('train', ('--stage', 'NATIVE')),
    'seam': ('train', ('--stage', 'SEAM')),
    'raw-common': ('sample', ('--label', 'COMMON_S12000')),
    'raw-native': ('sample', ('--label', 'NATIVE_S6000')),
    'raw-seam': ('sample', ('--label', 'SEAM_S3000')),
    'd0-common': ('post_d0', ('--label', 'COMMON_S12000')),
    'd0-native': ('post_d0', ('--label', 'NATIVE_S6000')),
    'd0-seam': ('post_d0', ('--label', 'SEAM_S3000')),
    'score-raw': ('evaluate', ('raw', '--workers', '4')),
    'score-post': ('evaluate', ('post', '--workers', '4')),
    'collect': ('collect', ()),
    'gallery': ('gallery', ('run', '--workers', '4')),
    'pipeline': ('controller', ('--execute',)),
}

def status():
    return {name: read(OUT/'workflow'/name/'status.json')
            if (OUT/'workflow'/name/'status.json').exists() else {'status':'NOT_STARTED'}
            for name in JOBS}

def launch(name):
    folder=OUT/'workflow'/name; marker=folder/'launch.json'
    if marker.exists():
        return read(folder/'status.json') if (folder/'status.json').exists() else read(marker)
    reg=read(OUT/'training_registration.json')
    assert reg['status']=='FROZEN'
    command=[sys.executable,'-B','-m',MODULE+'run','--execute',name]
    write_new(folder/'resources_before.json',resources())
    write_new(marker,dict(command=command,started=time.time(),automatic_retry=False))
    with (folder/'supervisor.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(status='LAUNCHED',pid=p.pid,stage=name,started=time.time())
    write_new(folder/'pid.json',value)
    return value

def execute(name):
    folder=OUT/'workflow'/name; path=folder/'status.json'
    if path.exists(): raise RuntimeError('Existing stage retained; no automatic replay')
    mod,args=JOBS[name]
    state=dict(status='RUNNING',stage=name,pid=os.getpid(),started=time.time())
    atomic_write(path,state)
    try:
        if name in ('native','seam'):
            from .dispatch import release
            state['dispatch']=release(name.upper())
        with (folder/'output.log').open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B','-m',MODULE+mod,*args],cwd=ROOT,
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=p.pid;atomic_write(path,state);rc=p.wait()
        state['exit_code']=rc
        if rc: raise RuntimeError('Stage failed; preserve logs and inspect before recovery')
        if name in ('native','seam'):
            actual=read(OUT/'training'/name.upper()/'status.json')
            if actual['status'] not in ('COMPLETE','INTERRUPTED_SAFE'):
                raise RuntimeError('Worker returned without a complete or safe training cursor')
            state['status']=actual['status']
            state['training_stage_step']=actual['stage_step']
        elif name == 'pipeline':
            actual=read(OUT/'controller/status.json')
            if actual['status'] != 'READY_REVIEW':
                raise RuntimeError('Controller stopped before READY_REVIEW; inspect actual child states')
            state['status']='COMPLETE'
            state['controller_status']=actual['status']
            state['actual_visual_review_complete']=False
        else:
            state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='STOPPED_REQUIRES_INSPECTION',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        state['finished']=time.time();state['seconds']=state['finished']-state['started']
        atomic_write(path,state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',choices=JOBS);p.add_argument('--execute',choices=JOBS)
    a=p.parse_args()
    if a.execute: execute(a.execute)
    else: print(json.dumps(launch(a.launch) if a.launch else status(),indent=2))
