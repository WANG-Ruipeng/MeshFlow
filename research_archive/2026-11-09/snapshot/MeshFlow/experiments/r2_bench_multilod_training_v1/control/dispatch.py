"""Resource-only dispatch records; never launches a scientific computation."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from r2.pilots.bench_multilod_training_v1.common import OUT,ARMS,REGISTRATION,read,sha,write
from r2.pilots.bench_four_routes_v1.resources import snapshot

p=argparse.ArgumentParser()
p.add_argument('--arm',choices=ARMS,required=True)
p.add_argument('--observed-peak-mib',type=float)
p.add_argument('--note',default='')
a=p.parse_args()
r=snapshot()
gpu=r['gpu']['stdout'].strip().split(',')
total,used,free,util=map(float,gpu[-4:])
headroom=max(2048.,.1*total)
if r['ram']['available']<12*1024**3:raise RuntimeError('Insufficient host RAM')
minimum=headroom if a.observed_peak_mib is None else headroom+1.2*a.observed_peak_mib
slots=1 if a.observed_peak_mib is None else math.floor(max(0,free-headroom)/(1.2*a.observed_peak_mib))
if free<minimum or slots<1:raise RuntimeError('Resource wait; no GO written')
if a.arm==ARMS[1] and a.observed_peak_mib is None:raise RuntimeError('Second worker requires representative measurement')
record=dict(authorized=True,arm=a.arm,training_registration_sha256=sha(REGISTRATION),
    retained_headroom_mib=headroom,minimum_free_mib=minimum,
    observed_comparable_worker_peak_mib=a.observed_peak_mib,estimated_incremental_slots=slots,
    resource=r,note=a.note,measurement_limit='WDDM process VRAM unavailable; full-card increment plus allocator and context margin used conservatively. Not a hardware guarantee.',
    scientific_budget_change=False)
write(OUT/'dispatch'/(a.arm+'_go.json'),record)
print(json.dumps(dict(arm=a.arm,free_mib=free,utilization=util,headroom_mib=headroom,slots=slots)))
