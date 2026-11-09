"""Execution accounting for the isolated terminal-reward pilot."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import json, time, datetime, subprocess, threading, traceback, sys
from . import data as D

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def resource_snapshot():
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    return {'utc':utc(), 'pid':os.getpid(), 'cpu_count':os.cpu_count(), 'load_average':list(os.getloadavg()) if hasattr(os,'getloadavg') else None,
            'CPU_process_load':command(['ps','-eo','pid,ppid,comm,pcpu,pmem,rss,nlwp','--sort=-pcpu']),
            'meminfo':Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,
            'gpu':command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
            'processes':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])}

class Budget:
    def __init__(self,path,max_rollouts,reference_check_policy=None):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.rollouts=0; self.forwards=0; self.max_rollouts=max_rollouts
        self.reference_check_policy=reference_check_policy
        self.handle=self.path.open('a',encoding='utf8',buffering=1)
    def event(self,name,**fields):
        self.handle.write(json.dumps(D.clean(dict(event=name,utc=utc(),pid=os.getpid(),**fields)),ensure_ascii=False)+'\n')
    def reserve(self,kind,context):
        if kind!='rollout_attempt':raise ValueError(kind)
        self.rollouts+=1
        if self.rollouts>self.max_rollouts:raise RuntimeError('Rollout budget exceeded')
        self.event('ROLLOUT_ATTEMPT',attempt=self.rollouts,context=context)
    def call(self,kind,fn,context):
        if kind!='sampling_forward':raise ValueError(kind)
        self.forwards+=1
        if self.forwards>self.max_rollouts*50:raise RuntimeError('Forward budget exceeded')
        self.event('FORWARD_START',kind=kind,count=self.forwards,request_id=context['request_id'],euler_step=context['euler_step'])
        result=fn()
        if self.reference_check_policy is not None:
            import torch
            with self.reference_check_policy.activate('reference'):
                self.event('VALIDATION_FORWARD_START',purpose='first_authorized_state_zero_LoRA_vs_reference')
                reference=fn()
                self.event('VALIDATION_FORWARD_RETURN',purpose='first_authorized_state_zero_LoRA_vs_reference')
            equal=torch.equal(result,reference)
            self.event('ZERO_LORA_CHECK',bitwise_equal=equal,maxabs=float((result.float()-reference.float()).abs().max().item()))
            if not equal:raise RuntimeError('Zero-adapter altered the first actual reference forward')
            self.reference_check_policy=None
        self.event('FORWARD_RETURN',kind=kind,count=self.forwards)
        return result

class Phase:
    """Record process/device memory samples, independently of PyTorch counters."""
    def __init__(self,path,identity):
        self.path=Path(path);self.identity=identity;self.stop=threading.Event();self.samples=[]
    def __enter__(self):
        if (self.path/'completion.json').exists():raise RuntimeError('Completed worker must be reused, not relaunched')
        if (self.path/'attempt.json').exists():raise RuntimeError('Unresolved worker attempt; explicit recovery required')
        self.path.mkdir(parents=True,exist_ok=True)
        self.tick=time.perf_counter(); self.initial=resource_snapshot()
        D.write(self.path/'attempt.json',dict(status='STARTED',identity=self.identity,resources=self.initial))
        def monitor():
            while not self.stop.wait(2):
                try:self.samples.append(resource_snapshot())
                except BaseException as exc:self.samples.append({'utc':utc(),'monitor_error':repr(exc)})
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
        return self
    def __exit__(self,typ,exc,tb):
        self.stop.set();self.thread.join(timeout=12)
        D.write(self.path/'resources.json',dict(initial=self.initial,samples=self.samples,final=resource_snapshot(),
                scope='Global GPU and compute-process polling every 2 seconds; WDDM process bytes may be NA. CUDA counters reported separately.'))
        payload=dict(status='COMPLETE' if exc is None else 'FAILED_NO_AUTO_REPLAY',identity=self.identity,
                     seconds=time.perf_counter()-self.tick,utc=utc())
        if exc is not None:payload.update(error=repr(exc),traceback=''.join(traceback.format_exception(typ,exc,tb)))
        D.write(self.path/('completion.json' if exc is None else 'failure.json'),payload)
        return False
