"""Read-only monitoring; no launch, signal, retry or model operations."""
import argparse,json,time
from .common import OUT,read,resources

def snapshot():
    p=OUT/'training/W_MATCHED/status.json'
    r=read(p) if p.exists() else {'status':'PREPARING'}
    d={k:r[k] for k in ('status','stage_step','global_step','elapsed_seconds','error','reason') if k in r}
    log=p.parent/'training_log.jsonl';rows=[]
    if log.exists():
        for line in log.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    if rows:
        tail=rows[-50:]
        d.update(recent_updates=len(tail),FM_mean=sum(x['FM'] for x in tail)/len(tail),
            objective_mean=sum(x['objective'] for x in tail)/len(tail),
            mean_update_seconds=sum(x['update_seconds'] for x in tail)/len(tail),
            mean_input_seconds=sum(x['input_seconds'] for x in tail)/len(tail),
            peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in rows),
            peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rows))
    return dict(training=d,resources=resources(),read_only=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--follow',action='store_true');a=parser.parse_args()
    path=OUT/'monitoring/training_timeline.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    while True:
        s=snapshot()
        if a.follow:
            with path.open('a',encoding='utf8') as f:f.write(json.dumps(s)+'\n')
        print(json.dumps(dict(utc=s['resources']['utc'],gpu=s['resources']['gpu']['stdout'].strip(),**s['training'])),flush=True)
        if not a.follow or s['training']['status'] in ('COMPLETE','FAILED','OOM','INTERRUPTED_SAFE'):return
        time.sleep(50)

if __name__=='__main__':main()
