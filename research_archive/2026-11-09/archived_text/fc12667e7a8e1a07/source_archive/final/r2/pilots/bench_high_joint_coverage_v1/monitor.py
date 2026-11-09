"""Read-only live observations, never dispatches or signals a process."""
import json
import time
import argparse
import functools
from .common import OUT, ARMS, NARROW_ROOT, NARROW_ARM, read, resources


@functools.lru_cache(maxsize=1)
def maximum_N_slot():
    manifest=read(OUT/'data/manifest.json');schedule=read(OUT/'data/schedule_WIDE_HIGH.json')
    maximum=max(manifest['train'],key=lambda r:int(r['N']))
    step=next(e['step'] for e in schedule['steps'] if any(
        r['condition_id']==maximum['condition_id'] for r in e['samples']))
    return dict(N=int(maximum['N']),K=int(maximum['K']),condition_id=maximum['condition_id'],first_step=step)


def snapshot():
    result={'resources':resources(),'arms':{},'read_only':True,'dispatch_authorized_by_monitor':False}
    prior=NARROW_ROOT/'training'/NARROW_ARM/'status.json'
    result['reused_comparator']={k:v for k,v in read(prior).items() if k in
        ('arm','status','stage_step','global_step','native_total_step')}
    result['reused_comparator']['new_training_in_this_experiment']=False
    maximum=maximum_N_slot();result['maximum_actual_train_N']=maximum
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
            lines=log.read_text().splitlines()
            # A live writer may have an unfinished final line; never repair it.
            rows=[]
            for line in lines:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
            max_update=next((x for x in rows if x['local_step']==maximum['first_step']),None)
            if max_update:
                d['first_max_N_formal_update']={k:max_update[k] for k in ('local_step','global_step',
                    'peak_allocated_bytes','peak_reserved_bytes','update_seconds','free_coordinate_denominator')}
                d['first_max_N_formal_update']['scope']='Completed formal backward and optimizer; allocator counters are not process VRAM'
            d['representative_max_N_update_completed']=max_update is not None
            rows=rows[-50:]
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
