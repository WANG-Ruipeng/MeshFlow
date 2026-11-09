"""One-shot CPU metadata/resource observation; no production mutations/model calls."""
import collections
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
def tail(path,n=100):
    if not path.exists():return []
    with path.open('rb') as f:
        f.seek(0,2);pos=f.tell();data=b''
        while pos and data.count(b'\n')<=n:
            count=min(65536,pos);pos-=count;f.seek(pos);data=f.read(count)+data
    out=[]
    for line in data.splitlines()[-n:]:
        try:out.append(json.loads(line))
        except json.JSONDecodeError:pass
    return out
def read(path):
    for attempt in range(5):
        try:return json.loads(path.read_text(encoding='utf-8-sig'))
        except FileNotFoundError:
            if attempt==4:return {'status':'TRANSIENT_OR_MISSING'}
            time.sleep(.05)
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=10)
    return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())

result=dict(unix=time.time(),dispatch=read(ROOT/'dispatch/status.json'),arms={},
    cpu_count=os.cpu_count(),loadavg=Path('/proc/loadavg').read_text().strip(),
    RAM={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()
         if line.split(':')[0] in {'MemTotal','MemAvailable','SwapTotal','SwapFree'}})
for arm in ('PLAIN','SEAM'):
    d=ROOT/'training'/arm
    records=tail(d/'ledger.jsonl',1200)
    complete=[r for r in records if r.get('event')=='UPDATE_COMPLETE']
    recent=tail(d/'training_log.jsonl',8)
    guards=sorted(d.glob('io_guard_17000_*.jsonl'),key=lambda p:p.stat().st_mtime)
    result['arms'][arm]=dict(status=read(d/'status.json'),recent_training=recent,
        recent_update_ledger=complete[-12:],last_ledger_records=records[-4:],
        IO_guards=[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=tail(p,30)) for p in guards[-3:]],
        recent_event_counts=dict(collections.Counter(r.get('event') for r in records)))
result['GPU']=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,pstate','--format=csv,noheader,nounits'])
result['GPU_processes']=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
ps=cmd(['ps','-eo','pid,ppid,pcpu,pmem,etimes,stat,args'])
result['target_processes']=[l for l in ps['stdout'].splitlines() if 'bench_parts_convergence_v1' in l and 'train17000_resource_check' not in l]
result.update(model_calls=0,GPU_model_calls=0,production_modified=False)
print(json.dumps(result,ensure_ascii=False,indent=2))
