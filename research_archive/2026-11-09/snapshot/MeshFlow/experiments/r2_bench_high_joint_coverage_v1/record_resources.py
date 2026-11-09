"""CPU-only environment snapshot before preparing this authorized experiment."""
import json
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resources
p=Path(__file__).resolve().parent/'resources_preparation.json'
assert not p.exists()
r=resources()
r['dispatch_plan']='One formal WIDE training task only after frozen gates; independent CPU data and four-RAW GEO_ONLY may overlap with bounded threads.'
r['scientific_calls']=0
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:r[k] for k in ('gpu','CPU_cores','CPU_percent','RAM_available','load')},indent=2))
