import json
from pathlib import Path
r=Path('native_t1/runs/jedge5_solver_pilot_20261002_121000')
a=json.loads((r/'evaluation/results.json').read_text())
print('keys',list(a['rows'][0]['scalars']))
for q in a['rows']:
 if q['patch']==4:print(q['seed'],q['method'],{k:v for k,v in q['scalars'].items() if any(s in k for s in ['coverage','minimum_angle','degenerate','nan','inf','affected'])})
for n in range(4):
 q=json.loads((r/f'raw/case{n}/results.json').read_text());print('time',n,q['lock_wait_seconds'],q['seconds'],q['counts'])
