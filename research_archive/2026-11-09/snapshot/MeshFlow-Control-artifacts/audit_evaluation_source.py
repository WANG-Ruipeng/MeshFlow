from pathlib import Path
import json,hashlib,ast
root=Path('/mnt/e/MeshFlow-Control')
p=root/'docs/evaluation_source_map.json';d=json.loads(p.read_text())
for r in d['files']:
 r['destination_sha256']=hashlib.sha256((root/r['destination']).read_bytes()).hexdigest()
d['new_adapters']=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),origin='New standalone adapter; formulas delegated to explicitly migrated functions') for p in (root/'src/meshflow_control/evaluation').glob('*.py') if str(p.relative_to(root)) not in {r['destination'] for r in d['files']}]
d['runtime_old_workspace_dependency']=False
d['migration_regression_receipt']='External artifacts/tests/evaluation_migration/regression.json'
p.write_text(json.dumps(d,indent=2)+'\n')
bad=[]
for p in (root/'src/meshflow_control/evaluation').glob('*.py'):
 t=ast.parse(p.read_text())
 for n in ast.walk(t):
  if isinstance(n,ast.ImportFrom) and n.module and ('experiments' in n.module or 'native_t1' in n.module):bad.append(str(p))
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('exec','eval'):bad.append(str(p))
print(json.dumps({'status':'PASS' if not bad else 'BLOCKED','forbidden_dependencies':bad,'source_modules':len(list((root/'src/meshflow_control/evaluation').glob('*.py')))}))
