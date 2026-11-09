from pathlib import Path
import json,csv,sys,importlib.util
R=Path('/mnt/e/MeshFlow') if Path('/mnt/e/MeshFlow').exists() else Path('E:/MeshFlow')
O=R/'native_t1/runs/chair_domain_control_v1'
print('PYTHON',sys.executable)
for m in ('numpy','scipy','matplotlib','pandas','trimesh','numba','psutil'):
 print(m,importlib.util.find_spec(m))
p=json.loads((O/'panels.json').read_text())
for tab in ('A','B','teacher'):
 r=next(x for x in p[tab] if x.get('status')=='READY')
 print('RECORD',tab,json.dumps(r))
try:
 import numpy as np
 for tab in ('A','B'):
  r=next(x for x in p[tab] if x.get('status')=='READY')
  for k in ('input_path','ref_npz','target_path','control_points_path'):
   if k in r:
    loc=Path(r[k]);loc=loc if loc.is_absolute() else O/loc
    with np.load(loc,allow_pickle=False) as z: print('ARRAYS',tab,k,{k:(z[k].shape,str(z[k].dtype)) for k in z.files})
except ImportError: pass
for name in ('protocol.json','checkpoint_manifest.json','execution_accounting.json'):
 obj=json.loads((O/name).read_text());print(name,json.dumps(obj)[:7500])
for path in sorted(O.glob('*.csv')):
 with path.open() as f:
  reader=csv.DictReader(f);rows=list(reader)
 print('CSV',path.name,len(rows))
