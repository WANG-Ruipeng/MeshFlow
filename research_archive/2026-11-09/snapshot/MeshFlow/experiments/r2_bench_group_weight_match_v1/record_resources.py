"""Preparation-only environment snapshot; no model calls."""
import json
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resources
p=Path(__file__).resolve().parent/'resources_preparation.json'
assert not p.exists()
r=resources()
r['dispatch_plan']='One authorized W_MATCHED1500 only after CPU/registration gates; independent CPU coefficient/evaluation setup parallel with bounded threads. No baseline retraining.'
r['scientific_calls']=0
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:r[k] for k in ('gpu','CPU_cores','CPU_percent','RAM_available','load')},indent=2))
