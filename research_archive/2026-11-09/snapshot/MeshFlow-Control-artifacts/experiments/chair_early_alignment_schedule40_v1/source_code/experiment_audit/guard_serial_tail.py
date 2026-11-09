#!/usr/bin/env python3
"""Guard only the current training controller's final isolated LATE tail.
No torch imports, model work, scientific-source edits, or GPU-worker signals.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1')
CONTROLLER='/mnt/e/MeshFlow-Control/tools/run_schedule40.py'
WORKER='/mnt/e/MeshFlow-Control/tools/train_schedule40.py'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return None


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def event(folder,kind,**kw):
    row=dict(time_unix=time.time(),guard_pid=os.getpid(),event=kind);row.update(kw)
    with (folder/'events.jsonl').open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row,sort_keys=True),flush=True)
    return row


def proc(pid):
    try:
        folder=Path('/proc')/str(pid)
        tail=folder.joinpath('stat').read_text().rsplit(')',1)[1].split()
        argv=[x.decode(errors='replace') for x in folder.joinpath('cmdline').read_bytes().split(bytes([0])) if x]
        return dict(pid=pid,state=tail[0],ppid=int(tail[1]),starttime=int(tail[19]),argv=argv)
    except (FileNotFoundError,ProcessLookupError):return None


def arg(argv,key):
    try:return argv[argv.index(key)+1]
    except (ValueError,IndexError):return None


def controller_ok(info,pid,starttime):
    return bool(info and info['pid']==pid and info['starttime']==starttime and CONTROLLER in info['argv']
        and arg(info['argv'],'--stage')=='train' and arg(info['argv'],'--out') in (None,str(ROOT)))


def worker_ok(info,pid,controller,starttime=None):
    return bool(info and info['pid']==pid and info['ppid']==controller and (starttime is None or info['starttime']==starttime)
        and WORKER in info['argv'] and arg(info['argv'],'--segment')=='LATE'
        and arg(info['argv'],'--attempt-id')=='initial' and arg(info['argv'],'--out')==str(ROOT))


def ready(completed,active,controller):
    return all(completed.get(x) for x in ('EARLY','COMMON','FM_ONLY')) and bool(active
        and active.get('controller_pid')==controller and set(active.get('workers',{}))=={'LATE'}
        and active['workers']['LATE'].get('phase')=='training' and active['workers']['LATE'].get('attempt')=='initial')


def resources(folder,phase):
    def command(argv):
        try:
            p=subprocess.run(argv,capture_output=True,text=True,timeout=4)
            return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except Exception as e:return dict(error=repr(e))
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key=line.split(':')[0]
        if key in ('MemTotal','MemAvailable','SwapFree'):mem[key]=int(line.split()[1])*1024
    row=dict(time_unix=time.time(),phase=phase,cpu_cores=os.cpu_count(),loadavg=list(os.getloadavg()),ram=mem,
        gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        cpu_processes=command(['ps','-eo','pid,ppid,pcpu,pmem,comm','--sort=-pcpu']))
    with (folder/'resources.jsonl').open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    return row


def run(pid,starttime):
    folder=ROOT/'audit/serial_tail_guard';folder.mkdir(parents=True,exist_ok=True)
    if read(folder/'status.json') is not None:raise RuntimeError('Existing guard receipt: no implicit second guard')
    state=dict(status='WAITING',controller_pid=pid,controller_starttime=starttime,guard_pid=os.getpid(),
        policy='Only stop verified controller when sole LATE tail; never signal GPU worker; serial failure blocks all retries',
        model_calls=0,gpu_worker_signals=0)
    atomic(folder/'status.json',state);event(folder,'GUARD_START',**state)
    paused=False;late_pid=None;late_start=None
    while True:
        tick=time.monotonic();resources(folder,state['status'])
        ci=proc(pid)
        if not controller_ok(ci,pid,starttime):
            state.update(status='IDENTITY_ALERT',controller_observed=ci,controller_was_paused=paused)
            atomic(folder/'status.json',state);event(folder,'IDENTITY_ALERT',**state);return 3
        if not paused:
            completed={}
            for name,end in [('EARLY',6000),('COMMON',5000),('FM_ONLY',6000)]:
                d=read(ROOT/'training'/name/'completed.json')
                completed[name]=bool(d and d.get('status')=='COMPLETE' and d.get('step')==end and d.get('segment')==name)
            active=read(ROOT/'audit/active_workers.json')
            if ready(completed,active,pid):
                late_pid=int(active['workers']['LATE']['pid']);li=proc(late_pid)
                if not worker_ok(li,late_pid,pid) or li['state'] in ('Z','X'):
                    state.update(status='IDENTITY_ALERT',late_observed=li,active=active)
                    atomic(folder/'status.json',state);event(folder,'LATE_IDENTITY_ALERT',**state);return 3
                if ci['state'] in ('T','t'):
                    state.update(status='IDENTITY_ALERT',reason='Controller already stopped by another owner')
                    atomic(folder/'status.json',state);event(folder,'EXTERNAL_STOP_ALERT',**state);return 3
                late_start=li['starttime']
                # Re-read both identities immediately before signaling only controller.
                if not controller_ok(proc(pid),pid,starttime) or not worker_ok(proc(late_pid),late_pid,pid,late_start):
                    event(folder,'PRE_SIGNAL_IDENTITY_ALERT');return 3
                os.kill(pid,signal.SIGSTOP);paused=True
                for _ in range(20):
                    ci=proc(pid)
                    if ci and ci['state'] in ('T','t'):break
                    time.sleep(.05)
                if not controller_ok(ci,pid,starttime) or ci['state'] not in ('T','t'):
                    state.update(status='IDENTITY_ALERT',reason='Controller stop not verified',controller_observed=ci)
                    atomic(folder/'status.json',state);event(folder,'STOP_NOT_VERIFIED',**state);return 3
                state.update(status='CONTROLLER_STOPPED_LATE_RUNNING',late_pid=late_pid,late_starttime=late_start,
                    controller_stop_unix=time.time(),prerequisites_complete=completed)
                atomic(folder/'status.json',state);event(folder,'SIGSTOP_CONTROLLER_ONLY',controller=ci,late=li)
        if paused:
            li=proc(late_pid)
            if li is not None and li['starttime']!=late_start:
                state.update(status='IDENTITY_ALERT',reason='LATE PID reused',late_observed=li)
                atomic(folder/'status.json',state);event(folder,'LATE_PID_REUSE_ALERT',**state);return 3
            done=read(ROOT/'training/LATE/completed.json')
            complete=bool(done and done.get('status')=='COMPLETE' and done.get('step')==6000 and done.get('segment')=='LATE')
            exited=li is None or li['state'] in ('Z','X')
            if li is not None and not exited and not worker_ok(li,late_pid,pid,late_start):
                state.update(status='IDENTITY_ALERT',reason='Live LATE identity changed',late_observed=li)
                atomic(folder/'status.json',state);event(folder,'LATE_IDENTITY_ALERT',**state);return 3
            if complete and exited:
                if not controller_ok(proc(pid),pid,starttime):
                    event(folder,'PRE_CONT_IDENTITY_ALERT');return 3
                os.kill(pid,signal.SIGCONT)
                state.update(status='COMPLETE',controller_resumed_unix=time.time(),late_completed_step=6000)
                atomic(folder/'status.json',state);event(folder,'SIGCONT_CONTROLLER_AFTER_LATE_COMPLETE',**state);return 0
            last=read(ROOT/'training/LATE/status.json') or {}
            failed=last.get('attempt_id')=='initial' and last.get('status') in ('OOM','BLOCKED','FAILED')
            if failed or (exited and not complete):
                if not controller_ok(proc(pid),pid,starttime):
                    event(folder,'PRE_KILL_IDENTITY_ALERT');return 3
                # SIGKILL avoids resuming a stopped dispatcher that might otherwise
                # dispatch its incorrect retry. This signal never targets LATE.
                os.kill(pid,signal.SIGKILL)
                state.update(status='BLOCKED',reason='Serial LATE failure/exit without completion; controller terminated, retries forbidden',
                    controller_terminated_unix=time.time(),late_status=last,late_observed=li)
                atomic(folder/'status.json',state);event(folder,'SIGKILL_VERIFIED_CONTROLLER_ONLY',**state);return 2
        time.sleep(max(0,5-(time.monotonic()-tick)))


def self_test():
    ci=dict(pid=318,starttime=4775,argv=['python',CONTROLLER,'--stage','train'])
    assert controller_ok(ci,318,4775) and not controller_ok(ci,318,4776)
    complete={x:True for x in ('EARLY','COMMON','FM_ONLY')}
    active=dict(controller_pid=318,workers={'LATE':dict(pid=10,phase='training',attempt='initial')})
    assert ready(complete,active,318)
    assert not ready(dict(complete,FM_ONLY=False),active,318)
    assert not ready(complete,dict(active,controller_pid=319),318)
    assert not ready(complete,dict(active,workers=dict(active['workers'],EARLY={})),318)
    li=dict(pid=10,ppid=318,starttime=123,argv=['python',WORKER,'--segment','LATE','--attempt-id','initial','--out',str(ROOT)])
    assert worker_ok(li,10,318,123) and not worker_ok(li,10,318,124)
    print(json.dumps(dict(status='PASS',simulated_logic=True,signals_sent=0,model_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller-pid',type=int,default=318);p.add_argument('--controller-starttime',type=int,default=4775);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:sys.exit(run(a.controller_pid,a.controller_starttime))
