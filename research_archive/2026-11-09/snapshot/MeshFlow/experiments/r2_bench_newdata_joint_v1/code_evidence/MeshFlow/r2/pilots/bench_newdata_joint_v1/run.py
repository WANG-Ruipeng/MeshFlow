"""Recoverable finite-stage launcher. Completed stages are read-only; no failed-call replay."""
import argparse,json,os,subprocess,sys,time,traceback
from .common import ROOT,OUT,ARM,LABEL,REFERENCE_LABEL,read,write_new,atomic_write,resources

MODULE='r2.pilots.bench_newdata_joint_v1.'
JOBS={
    'train':('training',()),
    'raw-baseline':('sample',('--label',REFERENCE_LABEL)),
    'raw-baseline-v2':('sample_v2',('--label',REFERENCE_LABEL)),
    'raw-new':('sample_v2',('--label',LABEL)),
    'd0-baseline':('post_d0',('--label',REFERENCE_LABEL)),
    'd0-new':('post_d0',('--label',LABEL)),
    'score-raw':('evaluation',('raw','--workers','4')),
    'score-raw-baseline':('evaluation',('raw','--labels',REFERENCE_LABEL,'--workers','4')),
    'score-post':('evaluation',('post','--workers','4')),
    'collect':('collect',()),
    'gallery':('gallery',('run','--workers','4')),
}

def status():
    return {name:read(OUT/'workflow'/name/'status.json') if (OUT/'workflow'/name/'status.json').exists() else {'status':'NOT_STARTED'} for name in JOBS}

def launch(name):
    folder=OUT/'workflow'/name;marker=folder/'launch.json'
    if marker.exists():
        return read(folder/'status.json') if (folder/'status.json').exists() else read(marker)
    command=[sys.executable,'-B','-m',MODULE+'run','--execute',name]
    write_new(folder/'resources_dispatch.json',resources())
    write_new(marker,dict(command=command,started=time.time(),automatic_retry=False))
    with (folder/'supervisor.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(status='LAUNCHED',pid=p.pid,stage=name,started=time.time())
    write_new(folder/'pid.json',value)
    return value

def execute(name):
    folder=OUT/'workflow'/name;path=folder/'status.json'
    if path.exists():raise RuntimeError('No automatic stage replay')
    mod,args=JOBS[name];state=dict(status='RUNNING',stage=name,pid=os.getpid(),started=time.time())
    atomic_write(path,state)
    try:
        if name=='train':
            from .dispatch import release
            state['dispatch']=release()
        with (folder/'output.log').open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B','-m',MODULE+mod,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=p.pid;atomic_write(path,state);rc=p.wait()
        state['exit_code']=rc
        if rc:raise RuntimeError(f'{name} failed; inspect output.log, no silent replay')
        state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='STOPPED_REQUIRES_INSPECTION',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        state['finished']=time.time();state['seconds']=state['finished']-state['started'];atomic_write(path,state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',choices=JOBS);p.add_argument('--execute',choices=JOBS)
    a=p.parse_args()
    if a.execute:execute(a.execute)
    else:print(json.dumps(launch(a.launch) if a.launch else status(),indent=2))
