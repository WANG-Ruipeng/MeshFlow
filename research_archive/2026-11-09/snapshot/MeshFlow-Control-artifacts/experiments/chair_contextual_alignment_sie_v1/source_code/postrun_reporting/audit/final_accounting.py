"""Final ledger summary; no torch, model call, or scientific recomputation."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,time,sys
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
calls=[json.loads(s) for s in (R/"audit/budget/calls.jsonl").read_text().splitlines() if s.strip()]
counts=defaultdict(Counter);arm=defaultdict(Counter);seconds=defaultdict(float);errors=[]
for x in calls:
 event=x["event"];kind=x.get("kind","NA");counts[kind][event]+=1
 if kind=="optimizer_update":arm[x.get("context",{}).get("arm","NA")][event]+=1
 if event=="RETURN":seconds[kind]+=x.get("seconds",0)
 if event=="ERROR":errors.append(x)
occ=[json.loads(s) for s in (R/"audit/budget/gpu_occupancy.jsonl").read_text().splitlines() if s.strip()]
starts={x["id"]:x for x in occ if x["event"]=="START"}
ends={x["id"]:x for x in occ if x["event"]=="END"}
durations=defaultdict(float)
for x in occ:
 if x["event"]=="END":durations[x["phase"]]+=x["seconds"]
owner_ids_unique=len(starts)==sum(x["event"]=="START" for x in occ) and len(ends)==sum(x["event"]=="END" for x in occ)
pending=[x for k,x in starts.items() if k not in ends]
mon=defaultdict(list)
for line in (R/"orchestration/gpu_monitor.jsonl").read_text().splitlines():
 x=json.loads(line)
 try:
  fields=x["gpu"].splitlines()[0].split(",")
  total,used,free,util=map(float,fields[-4:])
  mon[x["phase"]].append((total,used,free,util))
 except (KeyError,ValueError,IndexError):pass
peaks={k:dict(samples=len(v),total_MiB=v[0][0],peak_total_GPU_used_MiB=max(x[1] for x in v),minimum_free_MiB=min(x[2] for x in v),mean_compute_utilization_percent=sum(x[3] for x in v)/len(v),scope="5-second nvidia-smi device-wide samples; includes desktop usage, not exact instantaneous process peak") for k,v in mon.items()}
binding=read(R/"audit/execution_binding.json")
bound=[]
for p,h in binding["files"].items():
 a=digest(p);bound.append(dict(path=p,expected_sha256=h,actual_sha256=a,unchanged=a==h))
report=dict(schema="contextual_alignment_final_accounting_v1",time_unix=time.time(),controller=read(R/"orchestration/status.json"),calls={k:dict(v) for k,v in counts.items()},updates_by_arm={k:dict(v) for k,v in arm.items()},call_elapsed_seconds=dict(seconds),call_time_note="Nested head/combined backward times overlap; do not sum as independent GPU time.",errors=errors,gpu_ownership_seconds=sum(durations.values()),gpu_phase_seconds=durations,gpu_pending_owners=pending,gpu_owner_ids_unique=owner_ids_unique,gpu_end_without_start=[k for k in ends if k not in starts],gpu_monitor=peaks,execution_binding=bound,generator_calls_by_this_script=0,optimizer_updates_by_this_script=0)
if "--write" in sys.argv:(R/"audit/final_accounting.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
print(json.dumps(report,indent=2))
