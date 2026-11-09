"""Exact count audit only: no model, geometry, rendering, or original-scoring call.
Input is read-only MINI counts. All output files stay in this new directory.
"""
from __future__ import annotations
import argparse,collections,csv,datetime,hashlib,io,json,math
from fractions import Fraction
from pathlib import Path

SMALL='USER_SMALL'
EXP='USER_EXPANDED_67'
DEFAULT_MINI=Path('/mnt/e/MeshFlow/deliverables/R2_32_67_ANALYSIS_MINI_V1')
DEFAULT_OUT=Path('/mnt/e/MeshFlow/experiments/r2_source_interface_audit_v1/exact_count_audit')

def sha(b):return hashlib.sha256(b).hexdigest()
def mean(xs):return sum(xs)/len(xs) if xs else None
def exact_mean(xs):return sum(xs,Fraction(0))/len(xs) if xs else None
def sign(v):return 'NA'if v is None else 'IMPROVED'if v>0 else 'WORSE'if v<0 else 'TIED'
def fraction_fields(prefix,v):
    return {prefix+'_numerator':v.numerator if v is not None else None,prefix+'_denominator':v.denominator if v is not None else None,prefix+'_fraction':str(v)if v is not None else None,prefix+'_float_display':float(v)if v is not None else None}
def write_json(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main():
    a=argparse.ArgumentParser();a.add_argument('--mini',type=Path,default=DEFAULT_MINI);a.add_argument('--out',type=Path,default=DEFAULT_OUT);args=a.parse_args()
    mini=args.mini.resolve();out=args.out.resolve()
    if out==mini or mini in out.parents or out.name!='exact_count_audit':raise ValueError('Wrong write root')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():raise RuntimeError('Completed audit exists; no silent overwrite/retry')
    source_names=['AVMR_PMR_all8.csv','metric_dictionary.json','protocol_and_code/original/scoring.py','protocol_and_code/original/avmr_pmr.py','protocol_and_code/original/valid_graph.py','protocol_and_code/original/opposite_reach.py','_work/metrics_provenance.json']
    contents={n:(mini/n).read_bytes()for n in source_names}
    provenance=json.loads(contents['_work/metrics_provenance.json']);prior=provenance['outputs']
    sources=[]
    for n,data in contents.items():
        existing=prior.get(n,{}).get('sha256')
        if existing and existing!=sha(data):raise ValueError('MINI recorded hash differs: '+n)
        sources.append(dict(path=str(mini/n),relative_path=n,bytes=len(data),existing_sha256=existing,read_sha256=sha(data),existing_hash_source='_work/metrics_provenance.json::outputs'if existing else 'NO_SELF_HASH'))
    d=json.loads(contents['metric_dictionary.json']);settings={r['setting']:r for r in d['matching_settings']}
    allrows=list(csv.DictReader(io.StringIO(contents['AVMR_PMR_all8.csv'].decode('utf-8-sig'))))
    rows=[r for r in allrows if r['panel']=='NEW32' and r['stage']=='POST' and r['label']in(SMALL,EXP)]
    if len(rows)!=2048:raise ValueError('Expected 2 arms x128 requests x8 settings')
    lookup={}
    for r in rows:
        k=(r['request_id'],r['setting'],r['label'])
        if k in lookup:raise ValueError('Duplicate original row identity')
        lookup[k]=r
        for f in ('tau','threshold_model','scale','domain'):
            if r[f]!=settings[r['setting']][f]:raise ValueError('Original setting mismatch')
    parents={r['parent_id']for r in rows};conditions={r['condition_id']for r in rows};requests={r['request_id']for r in rows}
    if (len(parents),len(conditions),len(requests))!=(32,64,128):raise ValueError('Panel identity counts differ')
    per_request=[];request_values={}
    for rid in sorted(requests):
      for setting in settings:
        s,e=lookup[(rid,setting,SMALL)],lookup[(rid,setting,EXP)]
        for field in ('parent_id','condition_id','candidate','region_type','tier','boundary_vertex_count'):
            if s[field]!=e[field]:raise ValueError('Unpaired identity/denominator '+field)
        count=int(s['boundary_vertex_count']);ms=int(s['matched_count']);me=int(e['matched_count'])
        if not(0<=ms<=count and 0<=me<=count):raise ValueError('Invalid integer count')
        good=s['status']==e['status']=='PASS' and count>0
        for metric,raw_field in (('A_VMR','VMR'),('PMR','perfect')):
            exact_s=(Fraction(ms,count)if metric=='A_VMR'else Fraction(int(ms==count)))if good else None
            exact_e=(Fraction(me,count)if metric=='A_VMR'else Fraction(int(me==count)))if good else None
            fs=float(s[raw_field])if good else None;fe=float(e[raw_field])if good else None
            if good and (fs!=float(exact_s) or fe!=float(exact_e)):raise ValueError('Stored output float disagrees with original count ratio')
            de=exact_e-exact_s if good else None;df=fe-fs if good else None
            row=dict(panel='NEW32',stage='POST',comparison=EXP+'-'+SMALL,request_id=rid,parent_id=s['parent_id'],condition_id=s['condition_id'],candidate=s['candidate'],region_type=s['region_type'],tier=s['tier'],metric=metric,setting=setting,scale=s['scale'],tau=s['tau'],threshold_model=s['threshold_model'],domain=s['domain'],boundary_vertex_count=count,SMALL_matched_count=ms,EXP_matched_count=me,SMALL_status=s['status'],EXP_status=e['status'],SMALL_original_float_string=s[raw_field],EXP_original_float_string=e[raw_field],original_float_delta_reproduced=df,original_float_delta_hex=df.hex()if df is not None else None,exact_sign=sign(de),float_strict_sign=sign(df),status='PASS'if good else 'NA_FROM_ORIGINAL_STATUS_OR_EMPTY_BOUNDARY')
            row.update(fraction_fields('SMALL_exact',exact_s));row.update(fraction_fields('EXP_exact',exact_e));row.update(fraction_fields('delta_exact',de));per_request.append(row)
            request_values[(rid,setting,metric)]=(exact_s,exact_e,de,fs,fe,df)
    condition_groups=collections.defaultdict(list)
    for r in per_request:condition_groups[(r['parent_id'],r['condition_id'],r['setting'],r['metric'])].append(r)
    per_condition=[];condition_values={}
    for key,rr in sorted(condition_groups.items()):
        if len(rr)!=2 or {r['candidate']for r in rr}!={'0','1'}:raise ValueError('Expected fixed paired two-noise condition')
        rr=sorted(rr,key=lambda r:int(r['candidate']));vals=[request_values[(r['request_id'],r['setting'],r['metric'])]for r in rr]
        good=all(r['status']=='PASS'for r in rr)
        v=tuple((exact_mean([x[i]for x in vals])if i<3 else mean([x[i]for x in vals]))if good else None for i in range(6))
        r=dict(parent_id=key[0],condition_id=key[1],setting=key[2],metric=key[3],noise_count=2,status='PASS'if good else 'NA_CONTAINS_ORIGINAL_MISSING',original_float_delta_reproduced=v[5],exact_sign=sign(v[2]),float_strict_sign=sign(v[5]))
        for name,value in zip(('SMALL_exact','EXP_exact','delta_exact'),v[:3]):r.update(fraction_fields(name,value))
        per_condition.append(r);condition_values[key]=v
    parent_groups=collections.defaultdict(list)
    for r in per_condition:parent_groups[(r['parent_id'],r['setting'],r['metric'])].append(r)
    per_parent=[];parent_values={}
    for key,rr in sorted(parent_groups.items()):
        if len(rr)!=2:raise ValueError('Expected two registered conditions per parent')
        rr=sorted(rr,key=lambda r:r['condition_id']);vals=[condition_values[(r['parent_id'],r['condition_id'],r['setting'],r['metric'])]for r in rr]
        good=all(r['status']=='PASS'for r in rr)
        v=tuple((exact_mean([x[i]for x in vals])if i<3 else mean([x[i]for x in vals]))if good else None for i in range(6))
        r=dict(panel='NEW32',stage='POST',comparison=EXP+'-'+SMALL,parent_id=key[0],setting=key[1],metric=key[2],condition_count=2,request_count=4,condition_ids=json.dumps([r['condition_id']for r in rr]),status='PASS'if good else 'NA_CONTAINS_ORIGINAL_MISSING',SMALL_float_mean_reproduced=v[3],EXP_float_mean_reproduced=v[4],original_float_delta_reproduced=v[5],original_float_delta_hex=v[5].hex()if v[5]is not None else None,absolute_float_means_difference=v[4]-v[3]if good else None,exact_sign=sign(v[2]),float_strict_sign=sign(v[5]),sign_disagrees=sign(v[2])!=sign(v[5]))
        for name,value in zip(('SMALL_exact','EXP_exact','delta_exact'),v[:3]):r.update(fraction_fields(name,value))
        per_parent.append(r);parent_values[key]=v
    summary=[]
    for setting in settings:
      for metric in ('A_VMR','PMR'):
        rr=[r for r in per_parent if r['setting']==setting and r['metric']==metric];valid=[r for r in rr if r['status']=='PASS']
        ec=collections.Counter(r['exact_sign']for r in rr);fc=collections.Counter(r['float_strict_sign']for r in rr)
        dv=exact_mean([parent_values[(r['parent_id'],setting,metric)][2]for r in valid])
        r=dict(panel='NEW32',stage='POST',metric=metric,setting=setting,**{k:settings[setting][k]for k in('scale','tau','threshold_model','domain')},parents_registered=len(rr),parents_valid=len(valid),exact_improved=ec['IMPROVED'],exact_tied=ec['TIED'],exact_worse=ec['WORSE'],exact_NA=ec['NA'],float_strict_improved=fc['IMPROVED'],float_strict_tied=fc['TIED'],float_strict_worse=fc['WORSE'],float_strict_NA=fc['NA'],sign_disagreement_parents=sum(x['sign_disagrees']for x in rr),original_float_parent_delta_mean_reproduced=mean([x['original_float_delta_reproduced']for x in valid]))
        r.update(fraction_fields('parent_equal_delta_exact',dv));summary.append(r)
    differences=[r for r in per_parent if r['sign_disagrees']]
    for name,rr in [('per_request.csv',per_request),('per_condition.csv',per_condition),('per_parent.csv',per_parent),('summary.csv',summary),('float_sign_disagreements.csv',differences)]:write_csv(out/name,rr)
    expected={'L_tau_0.01':(10,11,11),'L_tau_0.02':(16,8,8)}
    check={}
    for setting,triple in expected.items():
        r=next(r for r in summary if r['setting']==setting and r['metric']=='A_VMR');actual=tuple(r[k]for k in('exact_improved','exact_tied','exact_worse'))
        check[setting]=dict(claim=list(triple),actual=list(actual),status='CONFIRMED'if actual==triple else 'DIFFERS_FROM_CLAIM')
    info=dict(schema='r2_matching_exact_count_audit_v1',status='COMPLETE',aggregation='Fraction(matched_count,boundary_vertex_count) or Fraction(perfect); paired EXP-SMALL; mean two noises within each C; mean two C within parent. Both metrics higher matching diagnostic only.',float_path='Reproduced from original VMR/perfect CSV strings, per-request subtraction then pairwise mean at C and parent. For each group n=2, Python sum/2 follows original np.mean(float64) arithmetic; original old tables are neither changed nor reloaded.',fraction_comparison='Exact numerator sign, no arbitrary epsilon, no output rounding',null_policy='Original non-PASS or empty boundary is NA, never filled with zero; no such case in selected rows',selected_source_rows=len(rows),request_metric_rows=len(per_request),condition_metric_rows=len(per_condition),parent_metric_rows=len(per_parent),summary_rows=len(summary),claim_checks=check,sign_disagreements=differences,all_parent_rows_derived=True,old_assets_written=0,model_calls=0,geometry_calls=0,render_calls=0)
    write_json(out/'summary.json',info)
    for x in sources:
        x['after_read_sha256']=sha(Path(x['path']).read_bytes());x['unchanged']=x['after_read_sha256']==x['read_sha256']
        if not x['unchanged']:raise ValueError('Source mutated during count audit')
    write_json(out/'source_hashes.json',dict(sources=sources,all_unchanged=True))
    files=[p for p in out.iterdir()if p.is_file()and p.name!='completion.json']
    receipt=dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))for p in sorted(files)],counts=info['claim_checks'],model_calls=0,geometry_calls=0,render_calls=0)
    write_json(out/'completion.json',receipt)
    print(json.dumps(dict(status='COMPLETE',claims=check,sign_disagreements=len(differences),summary=[{k:r[k]for k in('metric','setting','exact_improved','exact_tied','exact_worse','float_strict_improved','float_strict_tied','float_strict_worse')}for r in summary]),indent=2))

if __name__=='__main__':main()
