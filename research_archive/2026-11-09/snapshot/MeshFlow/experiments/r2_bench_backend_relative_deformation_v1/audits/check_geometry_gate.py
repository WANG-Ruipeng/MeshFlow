"""Recompute the frozen parent-level alert gate directly from per-output CSV."""
import collections,csv,json,math,statistics,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=csvread(OUT/'scores/per_output.csv');fields=['FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C','FF_illegal_pair_count']
grouped=collections.defaultdict(list)
for r in rows:
 if r['stage']=='CLEAN_FINAL_POST' and r['label'] in ('COVERAGE_FINAL','SHAPE_D0','RELDEF_D0'):
  grouped[(r['label'],r['eval_group'],r['parent_id'])].append(r)
parents={}
for k,v in grouped.items():
 assert len(v)==2 and len({r['condition_id'] for r in v})==1
 parents[k]={f:statistics.mean(float(r[f]) for r in v) for f in fields}
registered={(r['parent_id'],r['reference_label']):r for r in csvread(OUT/'analysis/geometry_gate_per_parent.csv')}
gates=[];alerts=collections.defaultdict(list)
for (label,group,parent),r in parents.items():
 if label!='RELDEF_D0':continue
 for ref in ('COVERAGE_FINAL','SHAPE_D0'):
  a=parents[(ref,group,parent)];d={s+'_delta_pp':100*(r[f]-a[f]) for s,f in zip(('FF','CF','I'),fields)}
  alert=d['FF_delta_pp']>=10 or d['CF_delta_pp']>=5 or d['I_delta_pp']>=5
  saved=registered[(parent,ref)]
  assert saved['eval_group']==group and saved['alert']==str(alert)
  for k,val in d.items():assert math.isclose(val,float(saved[k]),rel_tol=1e-10,abs_tol=1e-12)
  gates.append(dict(parent_id=parent,eval_group=group,reference_label=ref,alert=alert,**d))
  if alert and group in ('CURRENT_TRAIN','DEV'):alerts[ref].append(parent)
complete=read(OUT/'analysis/COMPLETE.json')
assert {k:len(alerts[k]) for k in ('COVERAGE_FINAL','SHAPE_D0')}==complete['alerts_by_reference']
assert complete['geometry_alert_old_parents']==len(alerts['COVERAGE_FINAL'])==4
assert complete['geometry_gate_pass']==(len(alerts['COVERAGE_FINAL'])<2)==False
macro=[]
for label in ('COVERAGE_FINAL','SHAPE_D0','RELDEF_D0'):
 for group in sorted({k[1] for k in parents}):
  pp=[r for (l,g,p),r in parents.items() if l==label and g==group]
  macro.append(dict(label=label,eval_group=group,parent_count=len(pp),outputs=2*len(pp),
    **{f:statistics.mean(r[f] for r in pp) for f in fields}))
result=dict(status='PASS',method='Independent original output -> one condition/noise mean -> parent-equal group mean; no production aggregator.',
 alerts_by_reference={k:len(alerts[k]) for k in ('COVERAGE_FINAL','SHAPE_D0')},alert_parents=dict(alerts),gate_pass=False,
 gates=gates,macro=macro,analysis_hashes={str(p):sha(p) for p in (OUT/'analysis/geometry_gate_per_parent.csv',OUT/'analysis/COMPLETE.json')},
 model_calls=0,scoring_calls=0)
dest=OUT/'audits/geometry_gate_independent.json'
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(status='PASS',alerts_by_reference=result['alerts_by_reference'],DEV=[r for r in macro if r['eval_group']=='DEV'])))
