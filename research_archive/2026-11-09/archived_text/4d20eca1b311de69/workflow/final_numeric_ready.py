"""Wait for the fixed 184 CPU outputs, then collect existing evidence once."""
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from r2.pilots.bench_full_prior_motif_guidance_v1 import prepare as P, evaluate as E, analyze as A

folder=P.OUT/'workflow/final_numeric_ready'
folder.mkdir(exist_ok=True)
claim=folder/'claim.json'
with claim.open('x') as f:
    json.dump({'utc':datetime.now(timezone.utc).isoformat(),'tasks':184,'new_scientific_calls':0},f)
while len(list((P.OUT/'cpu').glob('*/complete.json')))<184:
    time.sleep(30)
print('All 184 CPU receipts exist; collecting exact frozen rows.',flush=True)
E.collect('final')
result={name:fn('final') for name,fn in (('metrics',A.metrics),('guidance',A.guidance),('calls',A.calls))}
summary={name:{k:v for k,v in data.items() if k not in ('files','protected','historical_incidents')} for name,data in result.items()}
P.write_immutable(folder/'completion.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),results=summary,new_scientific_calls=0))
print(json.dumps({k:v.get('status','COMPLETE') for k,v in result.items()}),flush=True)
