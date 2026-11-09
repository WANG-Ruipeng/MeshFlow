import json
from pathlib import Path
r=Path('native_t1/runs/jedge5_blackbox_20261002_113600')
a=json.loads((r/'geometry/endpoint_results.json').read_text())
print('contexts',type(a['contexts']).__name__)
for q in a['contexts']:
 if q.get('parent')==0 and q.get('patch') in (0,4):print(q)
print('noise',a['noise'])
b=json.loads((r/'probes/combined_results.json').read_text());print('probe keys',list(b));print('probe nonrows',{k:v for k,v in b.items() if not isinstance(v,list)})
for k,v in b.items():
 if isinstance(v,list):print(k,'len',len(v),'first',v[:2])
