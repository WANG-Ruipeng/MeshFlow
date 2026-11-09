"""Read saved final results into report tables; zero scoring/model operations."""
import csv, json, hashlib
from pathlib import Path
from .common import OUT

ARMS=('FM_CONTROL','REPLAY_FM_CONTROL','TRAJECTORY_REPLAY','TRAJECTORY_REPLAY_CAPPED')
METRICS=('SeamCov','ConflictCov_C','ExposedRetain','D_sym2048','F_to_GT_MSE2048_L0',
         'GT_to_F_MSE2048_L0','fixed_query_MSE','fixed_query_p95_L','coverage_fraction',
         'CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction',
         'free_area_over_GT_free_area','free_exact_zero_area_fraction',
         'free_near_degenerate_fraction','free_minimum_angle_lt5_fraction')

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main(out=OUT):
    out=Path(out);dest=out/'report_tables';dest.mkdir(exist_ok=True)
    sources={}
    def read(p):
        p=out/p;data=p.read_bytes();sources[str(p)]=hashlib.sha256(data).hexdigest()
        return json.loads(data)
    for arm in ARMS:
        d=read(f'training/explore_{arm}_complete.json')
        assert d['status']=='PASS' and d['steps']==1000
    labels=['START']+[f'explore_{a}_{s}' for a in ARMS for s in (500,1000)]
    rows=[]
    for label in labels:
        d=read(f'evaluation/{label}/DEV/summary.json')
        assert d['status']=='COMPLETE' and d['completed']==16
        rows.append(dict(label=label,parents=8,outputs=16,**{m:d['metrics'][m] for m in METRICS},
            E_lower=d['metrics']['ExposedRetain_lower'],E_upper=d['metrics']['ExposedRetain_upper']))
    write_csv(dest/'DEV_all_models_metrics.csv',rows)
    header='| Model | S % | I % | E % | Dsym2048 | MSE512 | p95/L0 | CF % | FF % |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|'
    lines=[header]
    for r in rows:
        label=r['label'].replace('explore_','')
        values=[100*r[m] for m in ('SeamCov','ConflictCov_C','ExposedRetain')]
        values+=[r[m] for m in ('D_sym2048','fixed_query_MSE','fixed_query_p95_L')]
        values+=[100*r[m] for m in ('CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    (dest/'DEV_metrics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    comparisons=[]
    for arm in ('TRAJECTORY_REPLAY','TRAJECTORY_REPLAY_CAPPED'):
        for step in (500,1000):
            c=f'explore_{arm}_{step}';m=f'explore_REPLAY_FM_CONTROL_{step}'
            g=read(f'comparisons/DEV/{c}__vs__{m}/gate.json')
            for ref in ('START','matched'):
                block=g['vs_'+ref]
                for metric,s in block['stats'].items():
                    comparisons.append(dict(candidate=c,reference='START' if ref=='START' else m,
                        metric=metric,delta_percentage_points=s['delta']*100,
                        delta_lower_pp=s['lower']*100,delta_upper_pp=s['upper']*100,
                        favorable_parent_median_lower_pp=s['favorable_median_lower']*100,
                        favorable_parent_median_upper_pp=s['favorable_median_upper']*100,
                        improved_parents_lower=s['improved_parents_lower'],improved_parents_upper=s['improved_parents_upper'],
                        overall_gate=g['status'],numeric_gate=g['numeric_status'],shape_quality=g['shape_quality_status'],
                        path_A=g['path_A']['status'],path_B=g['path_B']['status'],
                        visual_gate=g['visual_review']['status']))
    write_csv(dest/'DEV_candidate_deltas_and_gates.csv',comparisons)
    lines=['| Candidate | Reference | S delta pp | I delta pp | E delta pp | Shape | A | B | Visual | Overall |',
           '|---|---|---:|---:|---:|---|---|---|---|---|']
    for c,m in dict.fromkeys((r['candidate'],r['reference']) for r in comparisons):
        group={r['metric']:r for r in comparisons if r['candidate']==c and r['reference']==m}
        r=group['SeamCov'];v=[group[k]['delta_percentage_points'] for k in ('SeamCov','ConflictCov_C','ExposedRetain')]
        lines.append('| '+c.replace('explore_','')+' | '+m.replace('explore_','')+' | '+' | '.join(f'{x:+.4f}' for x in v)+' | '+' | '.join(r[k] for k in ('shape_quality','path_A','path_B','visual_gate','overall_gate'))+' |')
    (dest/'DEV_gates.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (dest/'manifest.json').write_text(json.dumps(dict(status='COMPLETE',source_hashes=sources,
        interpretation='Rows are parent-equal means over the same frozen DEV16; all per-output/parent/noise sources remain under evaluation and comparisons. Delta I negative is favorable. Gates jointly require both references; repeated gate fields are not reference-specific independent gates.',
        model_calls=0,scorer_calls=0),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',metric_rows=len(rows),delta_rows=len(comparisons))))

if __name__=='__main__':main()
