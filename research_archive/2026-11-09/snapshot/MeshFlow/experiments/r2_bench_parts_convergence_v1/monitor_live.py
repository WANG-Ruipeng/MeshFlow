"""Read live metadata from WSL, avoiding Windows file-sharing delete locks."""
import json
from pathlib import Path
from r2.pilots.bench_parts_convergence_v1.controller import read

root=Path(__file__).resolve().parent
rows=[]
for arm in ('COMMON','PLAIN','SEAM'):
    folder=root/'training'/arm
    if not folder.exists():continue
    try:
        s=read(folder/'status.json')
        rows.append(dict(Arm=arm,Status=s.get('status'),Step=s.get('step'),Target=s.get('target')))
    except FileNotFoundError:
        rows.append(dict(Arm=arm,Status='METADATA_TEMPORARILY_UNAVAILABLE'))
dispatch=read(root/'dispatch/status.json')
print(json.dumps(dict(phase=dispatch.get('phase'),dispatch_status=dispatch.get('status'),training=rows)))
