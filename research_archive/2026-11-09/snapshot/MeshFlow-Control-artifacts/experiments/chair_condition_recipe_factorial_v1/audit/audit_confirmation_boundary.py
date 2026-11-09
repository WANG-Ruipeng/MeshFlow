import argparse,csv,hashlib,json,datetime
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--boundary',type=int,choices=[2000,4000],required=True);a=ap.parse_args()
run=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
picked={}
for label in ('C20_40_MIX','R2_MIX_H_ALL'):
 p=run/'confirmation/training'/label/'attempts/initial/training_log.csv'
 with p.open(encoding='utf-8-sig',newline='') as f:
  rows=[r for r in csv.DictReader(f) if r.get('step') in (str(a.boundary),str(a.boundary+1))]
 if len(rows)!=2:raise RuntimeError('Boundary not complete yet: '+label)
 for r in rows:
  step=int(r['step']); counts=(int(r['ratio20_samples']),int(r['ratio40_samples']))
  expected=(4,4) if label=='R2_MIX_H_ALL' else ((8,0) if step<=2000 else (0,8) if step<=4000 else (4,4))
  assert counts==expected,(label,step,counts)
  assert float(r['lambda_value'])==.25
  assert int(float(r['generator_adamw_step']))==step
 picked[label]={'path':str(p),'scope':'Two actual completed rows from append-only live log; final full audit still required','rows':rows}
receipt={'status':'PASS','boundary':a.boundary,'actual_rows':picked,'new_model_calls':0,'new_OT_calls':0}
p=run/'audit'/('confirmation_boundary_'+str(a.boundary)+'.json')
b=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode()
if p.exists() and p.read_bytes()!=b:raise RuntimeError('Refuse replacing boundary evidence')
p.write_bytes(b)
print(json.dumps({'status':'PASS','boundary':a.boundary,'sha256':hashlib.sha256(b).hexdigest(),'path':str(p)}))
