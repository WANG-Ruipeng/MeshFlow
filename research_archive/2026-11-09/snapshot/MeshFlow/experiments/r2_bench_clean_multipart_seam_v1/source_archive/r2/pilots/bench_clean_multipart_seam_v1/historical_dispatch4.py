"""One recorded CPU-only concurrency transition after an explicit parent drain.

No science settings or post_cpu source changes; completed cells are verified
and reused. This entry refuses to start while the old parent/workers live or
while any incomplete cell exists.
"""
from __future__ import annotations
import time
import psutil
from . import common as C
from . import post_cpu as P


def main():
    plan=C.read(C.OUT/"historical_drain_to_four_plan.json")
    observed=[]
    oldids=[plan["controller_pid"],*[r["pid"] for r in plan["child_processes"]]]
    for pid in oldids:
        if psutil.pid_exists(pid):
            proc=psutil.Process(pid)
            if proc.status()!=psutil.STATUS_ZOMBIE:
                observed.append(dict(pid=pid,command=proc.cmdline(),status=proc.status()))
    if observed:raise RuntimeError("Old CPU pipeline has not fully drained: "+repr(observed))
    partial=[str(p.parent) for p in (C.OUT/"historical").glob("*/*/started.json")
             if not (p.parent/"complete.json").exists()]
    if partial:raise RuntimeError("Incomplete cells after drain preserved: "+repr(partial))
    complete=list((C.OUT/"historical").glob("*/*/complete.json"))
    resources=C.resources()
    if resources["RAM_available"]<4*(1024**3) or resources["CPU_cores"]<8:
        raise RuntimeError("Insufficient measured CPU/RAM margin for4")
    stage=C.OUT/"historical_dispatch4_started.json"
    C.write_new(stage,dict(status="RUNNING",old_plan_sha256=C.sha(C.OUT/"historical_drain_to_four_plan.json"),
        old_controller_and_workers_exited=True,partial_cells=0,completed_cells_reused=len(complete),
        max_observed_prior_cell_RSS_KiB=max(C.read(p)["peak_RSS_KiB"] for p in complete),
        workers_requested=4,threads_per_worker=1,resources=resources,
        time_unix=time.time(),code_sha256=C.sha(__file__),
        scientific_change=False,extra_model_calls=0,extra_CPU_scientific_cells=0,
        timing="Four-worker phase timing separate from previous two-worker phase; concurrent throughput not isolated benchmark"))
    tick=time.perf_counter()
    result=P.run(workers=4)
    C.write_new(C.OUT/"historical_dispatch4_finished.json",
        dict(status="COMPLETE",time_unix=time.time(),seconds=time.perf_counter()-tick,
            completion_sha256=C.sha(C.OUT/"historical_completion.json"),resources_after=C.resources()))
    print("HISTORICAL144_COMPLETE",result["requests"],flush=True)


if __name__=="__main__":main()
