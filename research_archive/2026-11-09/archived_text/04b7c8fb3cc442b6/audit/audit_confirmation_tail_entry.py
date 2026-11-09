import csv,json,hashlib
from pathlib import Path
run=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
rows={}
for label in ('C20_40_MIX','R2_MIX_H_ALL'):
 p=run/'confirmation/training'/label/'attempts/initial/exposure.csv'
 with p.open(encoding='utf-8-sig',newline='') as f:
  values=[r for r in csv.DictReader(f) if r.get('step')=='4001']
 if len(values)!=8:raise ValueError('First tail batch not complete')
 rows[label]=values
a,b=list(rows.values())
assert [{k:v for k,v in r.items() if k!='recipe'} for r in a]==[{k:v for k,v in r.items() if k!='recipe'} for r in b]
assert sum(float(r['ratio'])==.2 for r in a)==4 and sum(float(r['ratio'])==.4 for r in a)==4
data={'status':'PASS','scope':'Actual first common tail batch step4001; final audit will check all2000 steps','rows':rows,'compared_fields':'All exposure.csv columns except recipe label','new_model_calls':0}
out=run/'audit/confirmation_tail_entry_actual.json'
payload=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
if out.exists() and out.read_bytes()!=payload:raise ValueError('Frozen evidence differs')
out.write_bytes(payload)
print(json.dumps({'status':'PASS','step':4001,'paired_samples':8,'sha256':hashlib.sha256(payload).hexdigest()}))
