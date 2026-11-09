"""Recorded non-scientific correction and control-flow smoke test."""
import difflib,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
P=Path('/mnt/e/MeshFlow');code=P/'r2/pilots/condition_width_response_v1/run.py'
out=P/'experiments/r2_condition_width_response_v1/engineering/entry_path_fix';out.mkdir(parents=True,exist_ok=True)
old=code.read_text()
needle='C.OUT/"execution/cpu_phase_dispatch_"+str(__import__("time").time_ns())+".json"'
new=old.replace(needle,'C.OUT/"execution"/("cpu_phase_dispatch_"+str(__import__("time").time_ns())+".json")')
assert new!=old
(out/'run.before.py').write_text(old)
code.write_text(new)
(out/'code.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='run.before.py',tofile='run.py')))
from r2.pilots.condition_width_response_v1 import run
calls=[];paths=[]
snapshot={'cpu_logical':24,'memory':'MemAvailable: 20000000 kB','loadavg':[0,0,0]}
with patch.object(sys,'argv',['run','--stage','all']),patch.object(run,'call',side_effect=lambda m,*a:calls.append([m,*a])),patch.object(run.C,'resources',return_value=snapshot),patch.object(run.C,'write',side_effect=lambda p,v:paths.append(str(p))):
 run.main()
assert len(paths)==1 and '/execution/cpu_phase_dispatch_' in paths[0]
assert len(calls)==10 and ['gpu','--stage','sample-probe'] in calls and ['evaluate','--stage','aggregate'] in calls
value=dict(status='PASS',scope='Mocked subprocess dispatch only; real model/scoring/rendering already completed separately',code_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),calls=calls,dispatch_path=paths[0],new_model_calls=0,scientific_definition_changed=False,reason='Static inspection found Path plus string in all-stage CPU dispatch filename; parenthesize the filename before joining.')
(out/'check.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
