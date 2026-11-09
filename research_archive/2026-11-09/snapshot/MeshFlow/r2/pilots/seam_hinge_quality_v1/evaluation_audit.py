"""Independent arithmetic/identity validation of completed frozen result tables."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,json,math,time
from .evaluation import C,OUT,HIST,ARMS,requests

def readcsv(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def run():
    dest=OUT/'evaluation_audit';done=OUT/'results/final/completion.json';final=C.read(done)
    for p,h in final['files'].items():assert C.sha(p)==h
    fields=final['original_numeric_fields'];assert len(fields)==124
    rows=readcsv(OUT/'results/final/per_output.csv');assert len(rows)==504
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows};assert len(lookup)==504
    qq={q['request_id']:q for q in requests()};counts=collections.Counter((r['label'],r['stage']) for r in rows)
    assert len(counts)==18 and set(counts.values())=={28}
    for r in rows:
        q=qq[r['request_id']]
        for k in ('parent_id','condition_id','split'):assert r[k]==q[k]
        for k in ('N','K','noise_index'):assert int(r[k])==q[k]
        assert r['C_preserved_exact']=='True'
        n=int(float(r['actual_free_faces']));k=q['K']
        assert int(float(r['free_face_denominator']))==n
        assert int(float(r['CF_tested_pair_denominator']))==k*n
        assert int(float(r['FF_tested_pair_denominator']))==n*(n-1)//2
        assert int(float(r['deleted_free_faces']))==q['N']-k-n
        if r['stage']=='RAW':assert n==q['N']-k
    old=readcsv(HIST/'results/final/per_output.csv');oldlookup={(r['label'],r['stage'],r['request_id']):r for r in old}
    comparisons=0
    for label in ('PLAIN_S1000','PLAIN_S2000'):
        for stage in ('RAW','POST','BOUND015'):
            for rid in qq:
                a,b=oldlookup[label,stage,rid],lookup[label,stage,rid]
                for metric in fields:
                    x,y=a.get(metric,''),b.get(metric,'')
                    if x==y:pass
                    elif x in ('True','False') or y in ('True','False'):assert x==y
                    elif x=='' or y=='':assert x==y
                    else:assert math.isclose(float(x),float(y),rel_tol=1e-13,abs_tol=1e-13),(label,stage,rid,metric,x,y)
                    comparisons+=1
    pairs=readcsv(OUT/'results/final/paired_per_noise.csv');checked=0
    for r in pairs:
        a=lookup[r['baseline'],r['baseline_stage'],r['request_id']];b=lookup[r['candidate'],r['candidate_stage'],r['request_id']]
        metric=r['metric'];x=a[metric];y=b[metric]
        if r['status']=='PASS':
            cv=lambda v:1. if v=='True' else 0. if v=='False' else float(v)
            assert math.isclose(cv(y)-cv(x),float(r['delta']),rel_tol=1e-12,abs_tol=1e-12)
        else:assert not x or not y
        checked+=1
    sums=readcsv(OUT/'results/final/summary.csv');sumchecks=0
    for r in sums:
        rr=[x for x in rows if x['label']==r['label'] and x['stage']==r['stage'] and (r['split']=='ALL' or x['split']==r['split'])]
        cv=lambda v:1. if v=='True' else 0. if v=='False' else float(v)
        vv=[cv(x[r['metric']]) for x in rr if x[r['metric']]!='']
        assert len(rr)==int(r['n_registered']) and len(vv)==int(r['n_valid'])
        if vv:assert math.isclose(sum(vv)/len(vv),float(r['mean']),rel_tol=1e-12,abs_tol=1e-12)
        else:assert r['mean']==''
        sumchecks+=1
    result=dict(status='PASS',rows=len(rows),groups=len(counts),original_field_count=124,
        baseline_cached_scalar_checks=comparisons,paired_delta_checks=checked,summary_mean_checks=sumchecks,
        sources={str(p):C.sha(p) for p in (done,OUT/'results/final/per_output.csv',OUT/'results/final/paired_per_noise.csv',OUT/'results/final/summary.csv',HIST/'results/final/per_output.csv')},
        checked='Distinct28requests all18method/stagedomains; exactoldbaseline124fields; explicitdenominators; pairedarithmetic; fullgroupmeans',
        model_calls=0,new_quality_scans=0,visual_review='SEPARATE_ACTUAL_IMAGE_RECORDS_REQUIRED')
    C.write(dest/'completion.json',result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2),flush=True)
