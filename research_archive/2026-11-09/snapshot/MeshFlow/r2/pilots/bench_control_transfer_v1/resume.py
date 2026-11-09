"""Explicit bounded-stage entry; defaults to read-only status."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_control_transfer_v1'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['status','train','sample','post','score','gallery'],default='status')
    p.add_argument('--arm',choices=['PLAIN','SEAM'])
    p.add_argument('--root',default=str(OUT))
    p.add_argument('--checkpoint',help='Explicit exact-cursor training resume only')
    a=p.parse_args();root=Path(a.root)
    if a.stage=='status':
        result={'root':str(root),'training':{},'evaluation':{}}
        for arm in ('PLAIN','SEAM'):
            path=root/'training'/arm/'status.json'
            result['training'][arm]=json.loads(path.read_text()) if path.exists() else {'status':'NOT_STARTED'}
            result['evaluation'][arm]={x:(root/'evaluation'/arm/(x+'_complete.json')).exists() for x in ('raw','post')}
        result['cleanup_complete']=(root/'cleanup/cleanup_receipt.json').exists()
        result['report_exists']=(root/'REPORT.md').exists()
        print(json.dumps(result,indent=2));return
    if a.stage in ('train','sample','post'):
        if not a.arm:p.error('--arm is required')
        query=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        total,free=map(float,query.stdout.strip().splitlines()[0].split(','))
        if free<max(2048,total*.1)+3000:raise RuntimeError('Insufficient conservative headroom; inspect resources before dispatch')
    module={'train':'train','sample':'sample','post':'postprocess','score':'evaluate','gallery':'gallery'}[a.stage]
    cmd=[sys.executable,'-B','-m','r2.pilots.bench_control_transfer_v1.'+module,'--root',str(root)]
    if a.arm and a.stage in ('train','sample','post'):cmd+=['--arm',a.arm]
    if a.checkpoint:
        if a.stage!='train':p.error('--checkpoint applies only to training')
        cmd+=['--resume',a.checkpoint]
    if a.stage=='score':
        for stage in ('prepare','score','aggregate'):
            subprocess.run(cmd+['--stage',stage],cwd=ROOT,check=True)
    else:subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
