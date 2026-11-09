import json,hashlib
from pathlib import Path
repo=Path('/mnt/e/MeshFlow-Control')
run=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
paths=sorted(set(p for f in ('tools','tests') for p in (repo/f).glob('*recipe_confirmation*.py')))
paths += [repo/'src/meshflow_control/data/recipe_confirmation.py',repo/'src/meshflow_control/training/recipe_confirmation.py']
data={'schema':'recipe_confirmation_additional_sources_v1','stage':'independent_confirmation','sources':{str(p.relative_to(repo)):sha(p) for p in paths},'scientific_previous_sources_modified':False,'note':'Separate extension registration; original frozen sources and training authorization remain unchanged.'}
out=run/'registrations/confirmation_additional_source_hashes.json'
raw=json.dumps(data,indent=2,sort_keys=True)+'\n'
if out.exists() and out.read_text()!=raw: raise RuntimeError('Refuse changing frozen additional sources')
out.write_text(raw,encoding='utf8')
print(json.dumps({'status':'PASS','count':len(paths),'path':str(out),'sha256':sha(out)}))
