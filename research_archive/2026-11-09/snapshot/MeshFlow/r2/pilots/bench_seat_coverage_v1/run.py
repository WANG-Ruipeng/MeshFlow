"""Recoverable finite launcher. A recorded failed scientific job is never replayed."""
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path
import psutil
from .common import ROOT, OUT, read, write_new, atomic_write, resources
MODULE='r2.pilots.bench_seat_coverage_v1.'
JOBS={
 'train':('train',()),
 'raw-native':('sample',('--label','NATIVE_S6000')),
 'raw-candidate':('sample',('--label','COVERAGE_FINAL')),
 'd0-native':('post_d0',('--label','NATIVE_S6000')),
 'd0-candidate':('post_d0',('--label','COVERAGE_FINAL')),
 'score-raw':('evaluate',('raw','--workers','4')),
 'score-post':('evaluate',('post','--workers','4')),
 'collect':('collect',()),
 'gallery':('gallery',('run','--workers','4')),
 'pipeline':('controller',('--execute',)),
}
def pid_alive(pid, created=None):
    """Inspect only; never signal another process or infer ownership by PID alone."""
    try:
        p=psutil.Process(int(pid))
        return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and (created is None or abs(p.create_time()-created)<1.)
    except (TypeError,ValueError,psutil.NoSuchProcess):return False
    except psutil.AccessDenied:return None

def job_status(folder, alive=pid_alive):
    folder=Path(folder)
    state_path=folder/'status.json'; pid_path=folder/'pid.json'
    if state_path.exists():
        state=read(state_path)
    elif pid_path.exists():
        state={'status':'LAUNCHED',**read(pid_path)}
    elif (folder/'launch.json').exists() or (folder/'resources_before.json').exists():
        return dict(status='LAUNCH_INCOMPLETE_REQUIRES_INSPECTION',reason='Launch/prelaunch exists without supervisor identity; never automatically relaunch',automatic_retry=False)
    else:return {'status':'NOT_STARTED'}
    if state['status'] in ('RUNNING','LAUNCHED'):
        running=alive(state.get('pid'),state.get('process_created'))
        if running is not True:
            return dict(state,status='STALE_PROCESS_REQUIRES_INSPECTION',recorded_status=state['status'],
                reason='Supervisor dead/reused or process visibility unavailable',supervisor_alive=running,automatic_retry=False)
        # A child may have just exited normally while its live supervisor is
        # writing COMPLETE/failed status. The supervisor owns p.wait()/exit-code
        # handling; do not turn that normal short handoff into a false failure.
        if 'child_pid' in state:
            state=dict(state,child_process_alive=alive(state['child_pid'],state.get('child_process_created')))
    return state

def status():
    return {k:job_status(OUT/'workflow'/k) for k in JOBS}
def launch(name):
    folder=OUT/'workflow'/name
    previous=job_status(folder)
    if previous['status']!='NOT_STARTED': return previous
    if read(OUT/'training_registration.json')['status']!='FROZEN': raise ValueError('Not frozen')
    write_new(folder/'resources_before.json',resources())
    command=[sys.executable,'-B','-m',MODULE+'run','--execute',name]
    write_new(folder/'launch.json',dict(command=command,time=time.time(),automatic_retry=False))
    try:
        with (folder/'supervisor.log').open('xb') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result=dict(status='LAUNCHED',pid=p.pid,process_created=psutil.Process(p.pid).create_time(),time=time.time())
        write_new(folder/'pid.json',result)
    except BaseException as exc:
        atomic_write(folder/'status.json',dict(status='STOPPED_REQUIRES_INSPECTION',job=name,
            error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,time=time.time()))
        raise
    return result
def execute(name):
    folder=OUT/'workflow'/name; dest=folder/'status.json'
    if dest.exists(): raise RuntimeError('Existing stage; inspect before any recovery')
    state=dict(status='RUNNING',job=name,pid=os.getpid(),process_created=psutil.Process().create_time(),started=time.time())
    atomic_write(dest,state)
    try:
        if name=='train':
            from .dispatch import release
            state['dispatch']=release()
        module,args=JOBS[name]
        with (folder/'output.log').open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B','-m',MODULE+module,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=p.pid
            try:state['child_process_created']=psutil.Process(p.pid).create_time()
            except psutil.NoSuchProcess:state['child_process_created']=None
            atomic_write(dest,state); rc=p.wait()
        state['exit_code']=rc
        if rc: raise RuntimeError('Worker failed; no automatic retry')
        if name=='train':
            actual=read(OUT/'training/NATIVE/status.json')
            if actual['status']!='COMPLETE': raise RuntimeError('Training returned without complete budget')
        if name=='pipeline' and read(OUT/'controller/status.json')['status']!='READY_REVIEW':
            raise RuntimeError('Pipeline stopped before ready review')
        state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='STOPPED_REQUIRES_INSPECTION',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        state.update(finished=time.time(),seconds=time.time()-state['started'])
        atomic_write(dest,state)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',choices=JOBS);p.add_argument('--execute',choices=JOBS);a=p.parse_args()
    if a.execute: execute(a.execute)
    else: print(json.dumps(launch(a.launch) if a.launch else status(),indent=2))
