"""Aggregate saved measurements only, preserving all 124 original fields."""
import csv, collections, math
from pathlib import Path
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C
OUT=C.ROOT/'experiments/r2_seam_followup_v1'
OLD=C.ROOT/'experiments/r2_seam_abc_v1'

def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def num(s):
    if s in ('True','true'):return 1.
    if s in ('False','false'):return 0.
    try:
        v=float(s);return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None

def main():
    inputs=[OLD/'evaluation/per_output.csv',OUT/'layout/evaluation/per_output.csv',OUT/'binding/per_output.csv']
    rows=[r for r in readcsv(inputs[0]) if r['label'] in ('LONG','A_COUPLE','B_POOL47')]
    rows += [r for r in readcsv(inputs[1]) if r['label']=='B_LONG_LAYOUT']
    rows += readcsv(inputs[2])
    ix={(r['label'],r['stage'],r['request_id']):r for r in rows}
    assert len(rows)==len(ix)==336
    metrics=sorted({r['metric'] for r in readcsv(OLD/'evaluation/summary.csv')})
    assert len(metrics)==124
    summaries=[];paired=[];parent=[];transitions=[]
    labels=list(dict.fromkeys(r['label'] for r in rows))
    for label in labels:
        for stage in sorted({r['stage'] for r in rows if r['label']==label}):
            for split in ('TRAIN','DEV','ALL'):
                rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                expected={'TRAIN':16,'DEV':12,'ALL':28}[split]
                for metric in metrics:
                    vals=[num(r.get(metric)) for r in rr];valid=[v for v in vals if v is not None]
                    summaries.append(dict(label=label,stage=stage,split=split,metric=metric,n_registered=expected,n_valid=len(valid),
                        mean=float(np.mean(valid)) if valid else None,median=float(np.median(valid)) if valid else None,
                        min=min(valid) if valid else None,max=max(valid) if valid else None,sum=sum(valid),
                        status='PASS' if len(valid)==expected else 'NA_OR_PARTIAL'))
    comparisons=[('LONG','B_LONG_LAYOUT'),('B_POOL47','B_LONG_LAYOUT')]
    comparisons += [(a,a+'_BIND_'+d) for a in ('LONG','A_COUPLE') for d in ('015','020')]
    for base,candidate in comparisons:
        for r in [r for r in rows if r['label']==candidate]:
            b=ix[base,r['stage'],r['request_id']]
            for metric in metrics:
                x,y=num(b.get(metric)),num(r.get(metric));valid=x is not None and y is not None
                paired.append(dict(baseline=base,candidate=candidate,request_id=r['request_id'],parent_id=r['parent_id'],
                    noise_index=r['noise_index'],stage=r['stage'],split=r['split'],metric=metric,baseline_value=x,candidate_value=y,
                    delta=y-x if valid else None,status='PASS' if valid else 'NA'))
            if r['stage']=='POST':
                x,y=num(b['free_only_reached_bulk_area_fraction']),num(r['free_only_reached_bulk_area_fraction'])
                transitions.append(dict(baseline=base,candidate=candidate,request_id=r['request_id'],split=r['split'],
                    baseline_bulk_reach=x,candidate_bulk_reach=y,gain_ge30=x<.3<=y,loss_ge30=y<.3<=x,
                    baseline_FF=num(b['FF_illegal_affected_free_fraction']),candidate_FF=num(r['FF_illegal_affected_free_fraction']),
                    baseline_I=num(b['ConflictCov_C']),candidate_I=num(r['ConflictCov_C'])))
    groups=collections.defaultdict(list);fields=('baseline','candidate','parent_id','split','stage','metric')
    for r in paired:groups[tuple(r[k] for k in fields)].append(r)
    for key,rr in sorted(groups.items()):
        valid=[r for r in rr if r['status']=='PASS']
        parent.append(dict(zip(fields,key),n_registered=4,n_valid=len(valid),
            **{k:float(np.mean([r[k] for r in valid])) if valid else None for k in ('baseline_value','candidate_value','delta')},
            status='PASS' if len(valid)==4 else 'NA_OR_PARTIAL'))
    for name,data in [('per_output',rows),('summary',summaries),('paired_per_noise',paired),('paired_per_parent',parent),('connectivity_transitions',transitions)]:
        C.csvwrite(OUT/'analysis'/(name+'.csv'),data)
    keys=['FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction',
        'free_only_reached_bulk_area_fraction','bulk_reach_ge_0_30_descriptive','legal_seam_bulk_edge_count','deleted_free_faces',
        'actual_free_faces','free_total_surface_area','nonmanifold_edges','EXACT_COORD_components']
    keyrows=[r for r in summaries if r['metric'] in keys and r['stage']=='POST' and r['split']!='ALL']
    C.csvwrite(OUT/'analysis/key_metrics.csv',keyrows)
    C.write(OUT/'analysis/completion.json',dict(status='COMPLETE',outputs=len(rows),numeric_fields=len(metrics),
        paired_per_noise=len(paired),paired_per_parent=len(parent),new_model_calls=0,new_scans=0,
        sources={str(p):C.sha(p) for p in inputs},files={str(p):C.sha(p) for p in (OUT/'analysis').glob('*.csv')}))
    for label in labels:
        for split in ('TRAIN','DEV'):
            v={r['metric']:r['mean'] for r in keyrows if r['label']==label and r['split']==split}
            print(label,split,'FF/I/reach/count',*(round(v[k]*(16 if split=='TRAIN' else 12) if k=='bulk_reach_ge_0_30_descriptive' else 100*v[k],4) for k in ['FF_illegal_affected_free_fraction','ConflictCov_C','free_only_reached_bulk_area_fraction','bulk_reach_ge_0_30_descriptive']))

if __name__=='__main__':main()
