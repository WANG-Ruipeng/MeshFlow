"""Read-only resource snapshots; no CUDA initialization or worker launching."""
from pathlib import Path
import argparse,json,os,subprocess,time

def snapshot(out,phase,note=''):
    def command(args):
        r=subprocess.run(args,text=True,capture_output=True)
        return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.split(':')[0] in ('MemTotal','MemAvailable','SwapTotal','SwapFree'):mem[line.split(':')[0]]=line.split(':')[1].strip()
    value=dict(timestamp=time.time(),phase=phase,note=note,cpu_cores=os.cpu_count(),load_average=os.getloadavg(),memory=mem,
        gpu=command(['nvidia-smi','--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']))
    path=Path(out)/'registration/resource_snapshots.jsonl'
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)
    return value
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--phase',required=True);p.add_argument('--note',default='');a=p.parse_args();snapshot(a.out,a.phase,a.note)
