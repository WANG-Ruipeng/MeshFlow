"""Read-only resource snapshot for authorized offline score dispatch."""
from r2.pilots.dedup_n_sensitivity_v1 import common as C
import argparse,json,time
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
r=C.resources();mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (r.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
value=dict(phase=a.phase,cpu_workers_requested=a.workers,resources=r,MemAvailable_bytes=mem.get('MemAvailable'),model_calls=0,
    rule='Only registered quality scores. Each arm measures its first authorized cell, caps workers at 2 and saves dispatch; a second arm requires a fresh resource check. No extra probes.')
dest=C.OUT/'evaluation/execution'/('cpu_dispatch_'+a.phase+'_'+str(time.time_ns())+'.json');C.write(dest,value)
print(json.dumps(dict(path=str(dest),cpu_count=r.get('cpu_logical'),loadavg=r.get('loadavg'),MemAvailable_GiB=mem.get('MemAvailable',0)/(1<<30),gpu=r.get('gpu'),cpu_workers_requested=a.workers)))
