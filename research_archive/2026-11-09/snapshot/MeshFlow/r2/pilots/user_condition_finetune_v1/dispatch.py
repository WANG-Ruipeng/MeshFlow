"""Record a measured resource decision without launching a job."""
import argparse,json,time,math
from control import OUT,resources,write,event
p=argparse.ArgumentParser();p.add_argument("--stage",required=True);p.add_argument("--reason",required=True)
p.add_argument("--workers",type=int,choices=(1,2),default=1);p.add_argument("--observed-peak-mib",type=float)
a=p.parse_args();r=resources("dispatch_"+a.stage)
fields=r["gpu"]["stdout"].strip().splitlines()[-1].split(",")
total,used,free,util=[float(s.strip().split()[0]) for s in fields[1:]]
headroom=max(2048.,.1*total)
if free<headroom:raise RuntimeError("Wait: insufficient current headroom")
slots=None if a.observed_peak_mib is None else math.floor(max(0,free-headroom)/(1.2*a.observed_peak_mib))
if a.workers==2 and (slots is None or slots<1):raise RuntimeError("Cannot add second worker from this observation")
d=dict(stage=a.stage,created_unix=time.time(),authorized=True,worker_count_after_dispatch=a.workers,
 gpu_total_mib=total,gpu_used_mib=used,gpu_free_mib=free,gpu_utilization=util,headroom_mib=headroom,
 observed_worker_process_peak_mib=a.observed_peak_mib,estimated_incremental_slots=slots,
 reason=a.reason,resource_snapshot=r,settings_changed_to_fit=False,extra_probe=False)
path=OUT/"dispatch"/(a.stage+"_"+str(time.time_ns())+".json");write(path,d);event("DISPATCH_DECISION",path=str(path),stage=a.stage,workers=a.workers)
print("DISPATCH_RECEIPT="+str(path),flush=True)
