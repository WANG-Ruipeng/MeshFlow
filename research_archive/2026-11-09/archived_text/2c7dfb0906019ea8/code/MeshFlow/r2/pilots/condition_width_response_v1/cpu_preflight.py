import ast,json
from pathlib import Path
from r2.pilots.condition_width_response_v1 import common as C
from r2.pilots.condition_width_response_v1 import gpu
C.package()
from meshflow_control.source_winding import validate_source
r=C.read(C.OUT/"registration.json")
for p in C.CODE.glob("*.py"):ast.parse(p.read_text(encoding="utf8"))
rows=[]
for q in r["requests"]:
    b=gpu.old().read_input(q)
    s=C.arrays(q["condition_source"])
    validate_source(s,b["C"])
    rows.append(dict(request_id=q["request_id"],input_pass=True,source_pass=True))
if len(rows)!=48:raise RuntimeError("48 requests required")
C.write(C.OUT/"execution/cpu_preflight.json",dict(status="PASS",checks=rows,registered_requests=48,models_loaded=0,model_forwards=0))
print(json.dumps(dict(status="PASS",requests=48,source_geometry_validated=48)))