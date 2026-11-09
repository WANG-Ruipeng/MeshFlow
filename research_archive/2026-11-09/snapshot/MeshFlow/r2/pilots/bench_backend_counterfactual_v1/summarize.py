"""Deterministic small tables, no geometry recomputation or model calls."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_backend_counterfactual_v1'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def run():
    allrows=rows(OUT/'scores/original_reference_scores.csv')+rows(OUT/'scores/new_per_output.csv')
    fields=['actual_free_faces','deleted_free_faces','free_total_surface_area','CF_illegal_affected_free_fraction',
            'FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction']
    result=[]
    for label in ['NATIVE_S6000','COVERAGE_FINAL','SOURCE_REFERENCE']:
        for stage in ['RAW','BASE_POST','GEO_RAW','SOURCE_GEO']:
            rr=[r for r in allrows if r['label']==label and r['stage']==stage]
            if not rr:continue
            r={'label':label,'stage':stage,'outputs':len(rr),'parents':len(set(x['parent_id'] for x in rr))}
            for field in fields:
                vals=[float(x[field]) for x in rr if x.get(field) not in ('',None)]
                r[field]=statistics.mean(vals) if len(vals)==len(rr) else None
            r['FF_illegal_pairs_sum']=sum(int(x['FF_illegal_pair_count']) for x in rr)
            r['CF_illegal_pairs_sum']=sum(int(x['CF_illegal_pair_count']) for x in rr)
            result.append(r)
    p=OUT/'summary.csv'
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
