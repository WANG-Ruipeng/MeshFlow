"""Read-only live observations, never dispatches or signals a process."""
import json
import time
import argparse
from .common import OUT, ARMS, read, resources


def snapshot():
    result={'resources':resources(),'arms':{}}
    for arm in ARMS:
        p=OUT/'training'/arm/'status.json'
        if not p.exists():
            result['arms'][arm]={'status':'NOT_STARTED'};continue
        r=read(p)
        keys=('status','stage_step','global_step','elapsed_seconds','error','reason','resources')
        d={k:r[k] for k in keys if k in r}
        if 'last' in r:
            d['last']={k:r['last'][k] for k in ('FM','input_seconds','update_seconds','peak_allocated_bytes','peak_reserved_bytes','free_coordinate_denominator')}
        log=p.parent/'training_log.jsonl'
        if log.exists():
            lines=log.read_text().splitlines()[-50:]
            # A live writer may have an unfinished final line; never repair it.
            rows=[]
            for line in lines:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
            if rows:
                d['recent_updates']=len(rows)
                d['mean_update_seconds']=sum(x['update_seconds'] for x in rows)/len(rows)
                d['mean_input_seconds']=sum(x['input_seconds'] for x in rows)/len(rows)
                d['recent_allocator_reserved_max']=max(x['peak_reserved_bytes'] for x in rows)
        result['arms'][arm]=d
    return result


def follow():
    path=OUT/'monitoring/training_timeline.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        while True:
            value=snapshot()
            f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps(dict(utc=value['resources']['utc'],gpu=value['resources']['gpu']['stdout'].strip(),
                arms={a:{k:v for k,v in r.items() if k in ('status','stage_step','global_step','mean_update_seconds','mean_input_seconds','error')}
                      for a,r in value['arms'].items()})),flush=True)
            states=[r['status'] for r in value['arms'].values()]
            if all(x=='COMPLETE' for x in states) or any(x in ('FAILED','OOM','INTERRUPTED_SAFE') for x in states):return
            time.sleep(50)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--follow',action='store_true');a=p.parse_args()
    if a.follow:follow()
    else:print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
