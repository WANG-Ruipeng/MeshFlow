"""Observed resources, one explicitly authorized GPU worker, no probe jobs."""
from pathlib import Path
import os
import subprocess
import threading
import time
from .common import append_event,atomic_json
from experiments.mf_h1_local0.chair_control_usability_quick_v1.quick_common import resource_snapshot


class Monitor:
    def __init__(self,out,stage):
        self.out=Path(out);self.stage=stage;self.stop=threading.Event();self.samples=[]
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):self.thread.start();return self
    def run(self):
        while not self.stop.is_set():
            row=dict(stage=self.stage,worker_pid=os.getpid())
            for key,args in [('gpu',['--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu']),
                             ('processes',['--query-compute-apps=pid,process_name,used_gpu_memory'])]:
                try:
                    p=subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
                    row[key]=dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
                except Exception as e:row[key]=dict(error=repr(e))
            append_event(self.out/'resources'/f'{self.stage}_monitor.jsonl',row);self.samples.append(row)
            self.stop.wait(5)
    def close(self):
        self.stop.set();self.thread.join(timeout=20)
        values=[];process=[]
        for r in self.samples:
            try:values.append(int(r['gpu']['stdout'].split(',')[2].strip()))
            except (KeyError,ValueError,IndexError):pass
            for line in r.get('processes',{}).get('stdout','').splitlines():
                parts=line.split(',')
                if parts and parts[0].strip()==str(os.getpid()):
                    try:process.append(int(parts[-1].strip()))
                    except ValueError:pass
        result=dict(stage=self.stage,samples=len(self.samples),GPU_workers=1,
            observed_device_used_peak_MiB=max(values,default=None),observed_own_process_peak_MiB=max(process,default=None),
            process_accounting='NA when WSL/WDDM does not expose numeric per-process memory',
            note='Device usage includes unrelated desktop/user processes; not PyTorch allocated+reserved sum',
            dispatch_reason='Task explicitly requires one GPU worker; independent CPU audit overlaps with 2 threads per process')
        atomic_json(self.out/'resources'/f'{self.stage}_summary.json',result);return result


def before_dispatch(out,stage):
    value=resource_snapshot(out,'before_dispatch_'+stage)
    headroom=max(2048,.1*(value['total_mib'] or 16303))
    if value['free_mib'] is None or value['free_mib']<headroom+3072:
        raise RuntimeError('RESOURCE_WAIT: insufficient observed free VRAM; no forced dispatch')
    return value
