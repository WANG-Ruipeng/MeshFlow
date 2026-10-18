"""Full 135-field paired accounting; stdlib only, no scoring/model calls."""
from pathlib import Path
import csv, hashlib, json, math, statistics as st

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_hinge_opposite_v1'
OLD=ROOT/'experiments/r2_seam_hinge_quality_v1/results/final/per_output.csv'
STAGES=['POST','POST_HINGE015','POST_HINGE025','HINGE_OPP015','HINGE_OPP025']
COMPARISONS=[('POST',s) for s in STAGES[1:]]+[('POST_HINGE015','HINGE_OPP015'),('POST_HINGE025','HINGE_OPP025')]
META={'request_id','condition_id','parent_id','noise_index','N','K','split','label','stage','mesh_path','domain','topology_definition_version'}

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def num(v):
    if str(v).lower() in ('true','false'):return float(str(v).lower()=='true')
    try:
        x=float(v);return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None
def quantile(a,p):
    a=sorted(a);x=(len(a)-1)*p;i=math.floor(x);j=math.ceil(x)
    return a[i]+(a[j]-a[i])*(x-i)

def main():
    policy_path=Path(__file__).with_name('metric_policy.json')
    policy=json.loads(policy_path.read_text(encoding='utf-8-sig'))
    fields=list(policy['directions']);assert len(fields)==135
    rows=[r for r in read(OLD) if r['label']=='PLAIN_S2000' and r['stage'] in STAGES[:3]]+read(OUT/'new_per_output.csv')
    lookup={(r['stage'],r['request_id']):r for r in rows};assert len(rows)==len(lookup)==140
    ids=sorted({r['request_id'] for r in rows});assert len(ids)==28
    for rid in ids:
        b=lookup['POST',rid]
        for s in STAGES[1:]:
            for k in ('condition_id','parent_id','noise_index','N','K','split','actual_free_faces','deleted_free_faces','free_face_denominator','C_query_weight_denominator'):
                assert lookup[s,rid][k]==b[k],(rid,s,k)
    write('combined_per_output.csv',sorted(rows,key=lambda r:(r['split'],r['parent_id'],r['noise_index'],STAGES.index(r['stage']))))
    deltas=[]
    for ref,candidate in COMPARISONS:
        for rid in ids:
            b,r=lookup[ref,rid],lookup[candidate,rid]
            for k in fields:
                x,y=num(b.get(k)),num(r.get(k));d=y-x if x is not None and y is not None else None;direction=policy['directions'][k]
                change='NA' if d is None else 'unchanged' if abs(d)<=1e-12 else ('increased' if d>0 else 'decreased') if direction=='context_only' else 'improved' if (d<0)==(direction=='lower_descriptive') else 'worsened'
                deltas.append(dict(reference=ref,candidate=candidate,request_id=rid,parent_id=r['parent_id'],noise_index=r['noise_index'],split=r['split'],metric=k,baseline=x,candidate_value=y,delta=d,direction=direction,change=change))
    write('paired_per_output.csv',deltas);write('all_regressions.csv',[r for r in deltas if r['change']=='worsened'])
    summaries=[];parents=[]
    for group in ['ALL','TRAIN','DEV']+sorted({r['parent_id'] for r in rows}):
        subset=[rid for rid in ids if group=='ALL' or lookup['POST',rid]['split']==group or lookup['POST',rid]['parent_id']==group]
        for k in fields:
            rr=dict(group=group,metric=k,n_registered=len(subset),direction=policy['directions'][k])
            for s in STAGES:
                valid=[num(lookup[s,rid].get(k)) for rid in subset];valid=[v for v in valid if v is not None];rr[s+'_n']=len(valid)
                for stat,fun in [('mean',st.mean),('median',st.median),('min',min),('max',max),('sum',sum),('p95',lambda v:quantile(v,.95))]:rr[s+'_'+stat]=fun(valid) if valid else None
            (summaries if group in ('ALL','TRAIN','DEV') else parents).append(rr)
    write('metric_summary.csv',summaries);write('paired_per_parent.csv',parents)
    counts=[]
    for ref,candidate in COMPARISONS:
        for group in ('ALL','TRAIN','DEV'):
            for k in fields:
                dd=[r for r in deltas if r['reference']==ref and r['candidate']==candidate and r['metric']==k and (group=='ALL' or r['split']==group)]
                valid=[r['delta'] for r in dd if r['delta'] is not None];bad=sorted([r for r in dd if r['change']=='worsened'],key=lambda r:abs(r['delta']),reverse=True)
                counts.append(dict(group=group,reference=ref,candidate=candidate,metric=k,delta_mean=st.mean(valid) if valid else None,n=len(valid),**{c:sum(r['change']==c for r in dd) for c in ('improved','unchanged','worsened','increased','decreased','NA')},worst_request=bad[0]['request_id'] if bad else '',worst_delta=bad[0]['delta'] if bad else 0))
    write('paired_summary.csv',counts)
    result=dict(status='COMPLETE',rows=len(rows),numeric_fields=len(fields),paired_fields=len(deltas),summary_rows=len(summaries),parent_rows=len(parents),missing_numeric_cells=sum(num(r.get(k)) is None for r in rows for k in fields),original_scorers_unchanged=True,source_hashes={str(p):sha(p) for p in (OLD,OUT/'new_per_output.csv',policy_path)},aggregation='macro per output, four noises per parent, ALL weighted 16 TRAIN:12 DEV; source parent is not a reconstruction target',model_calls=0)
    (OUT/'analysis_complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
