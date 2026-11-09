"""All original score fields; two-model request/condition/parent paired tables."""
import json
from .common import OUT, REFERENCE_LABEL, REFERENCE_ROOT, LABEL, read, sha, write_new, resolve
from .eval_runtime import requests, LABELS, verify
from .evaluation import score_dir, STAGES
from r2.pilots.bench_adaptation_strength_v1.collect_v2 import decorate,numeric_fields,numeric,aggregate,csvwrite

def run():
    out=OUT/'results';done=out/'completion.json'
    if done.exists():
        old=read(done);verify(old['files']);return old
    current=read(OUT/'evaluation/score_registration.json')
    old=read(REFERENCE_ROOT/'evaluation/score_registration.json')
    common=set(current['source_hashes'])&set(old['source_hashes'])
    assert common and current['chain']==old['chain'] and current['stages']==old['stages']
    for p in common: assert current['source_hashes'][p]==old['source_hashes'][p]==sha(p),p
    rows=[];sources={};qs=requests()
    for label in LABELS:
        for q in qs:
            for stage in STAGES:
                folder=score_dir(label,q,stage);receipt=read(folder/'complete.json')
                assert receipt['status']=='PASS' and receipt['model_calls']==0
                path=folder/'row.json';h=sha(path)
                assert any(resolve(p).resolve()==path.resolve() and x==h for p,x in receipt['files'].items())
                src=read(path);row=decorate(src,label,q,stage)
                row['eval_group']=q['eval_group']
                row['result_origin']='REUSED_HISTORICAL' if label==REFERENCE_LABEL and q['baseline_reuse'] else 'NEW_REGISTERED'
                rows.append(row);sources[str(path)]=h
    assert len(rows)==168
    metrics=numeric_fields(rows)
    conditions=aggregate(rows,('label','stage','eval_group','parent_id','condition_id'),metrics)
    parents=aggregate(conditions,('label','stage','eval_group','parent_id'),metrics)
    summary=aggregate(parents,('label','stage','eval_group'),metrics)
    pairs=[]
    for stage in STAGES:
        by={(r['label'],r['request_id']):r for r in rows if r['stage']==stage}
        for q in qs:
            a,b=by[REFERENCE_LABEL,q['request_id']],by[LABEL,q['request_id']]
            r=dict(reference=REFERENCE_LABEL,candidate=LABEL,stage=stage,
                **{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','split','part_type','eval_group')})
            for m in metrics:
                av,bv=a.get(m),b.get(m)
                r[m+'__reference']=av;r[m+'__candidate']=bv
                r[m+'__delta']=float(bv-av) if numeric(av) and numeric(bv) else None
            pairs.append(r)
    pc=aggregate(pairs,('reference','candidate','stage','eval_group','parent_id','condition_id'),[m+'__delta' for m in metrics])
    pp=aggregate(pc,('reference','candidate','stage','eval_group','parent_id'),[m+'__delta' for m in metrics])
    tables=dict(per_output=rows,per_condition=conditions,per_parent=parents,parent_macro_summary=summary,
                paired_per_output=pairs,paired_per_condition=pc,paired_per_parent=pp)
    for name,table in tables.items():csvwrite(out/(name+'.csv'),table)
    result=dict(status='COMPLETE',rows=168,new_score_rows=144,reused_score_rows=24,
        metric_fields=metrics,source_rows=sources,model_calls=0,scoring_calls=0,
        collector_sha256=sha(__file__),files={str(out/(n+'.csv')):sha(out/(n+'.csv')) for n in tables})
    write_new(done,result);return result

if __name__=='__main__':print(json.dumps({k:v for k,v in run().items() if k not in ('files','source_rows','metric_fields')}))
