"""Read-only resource snapshot; imports no neural framework."""
import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path
import psutil


def snapshot():
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    gpu = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'])
    gp = command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    processes=[]
    for p in psutil.process_iter(['name','memory_info','cmdline','cpu_times']):
        try:
            if any(x in (p.info['name'] or '').lower() for x in ('python','train','render')):
                processes.append(dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss,
                                      cpu_times=list(p.info['cpu_times']),cmdline=p.info['cmdline']))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_count=os.cpu_count(),
                ram=psutil.virtual_memory()._asdict(),cpu_percent=psutil.cpu_percent(1),
                gpu=gpu,gpu_processes=gp,processes=processes,
                process_gpu_memory_limitation='WDDM per-process usage may be N/A; do not equate allocator with process VRAM')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');args=p.parse_args();r=snapshot()
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,ensure_ascii=False)
    print(json.dumps(r,ensure_ascii=False))
