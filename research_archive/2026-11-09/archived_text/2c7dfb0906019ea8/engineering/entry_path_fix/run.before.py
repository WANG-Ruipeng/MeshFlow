"""Recoverable stage entry; scientific cells are never silently replayed."""
import argparse, json, subprocess, sys
from pathlib import Path
from . import common as C
def call(module,*args):
    p=subprocess.run([sys.executable,"-B","-m","r2.pilots.condition_width_response_v1."+module,*args],cwd=C.ROOT)
    if p.returncode:raise SystemExit(p.returncode)
def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("all","prepare","generate","postprocess","score","render","analyze","aggregate"),default="all")
    a=p.parse_args();stage=a.stage
    if stage in ("all","prepare"):
        call("prepare");call("cpu_preflight");call("analyze","--freeze");call("gpu","--stage","freeze-code")
    if stage in ("all","generate"):call("gpu","--stage","sample-probe")
    if stage in ("all","postprocess"):call("gpu","--stage","postprocess")
    if stage=="all":
        from concurrent.futures import ThreadPoolExecutor
        snapshot=C.resources();C.write(C.OUT/"execution/cpu_phase_dispatch_"+str(__import__("time").time_ns())+".json",snapshot)
        mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snapshot.get("memory") or "").splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
        parallel=(snapshot["cpu_logical"] or 1)>=6 and mem.get("MemAvailable",0)>8*1024**3 and (snapshot.get("loadavg") or [999])[0]<4
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(call,"evaluate","--stage","score","--max-workers","2"),pool.submit(call,"rendering")]
                for job in jobs:job.result()
        else:
            call("evaluate","--stage","score","--max-workers","1");call("rendering")
    elif stage=="score":call("evaluate","--stage","score","--max-workers","2")
    elif stage=="render":call("rendering")
    if stage in ("all","analyze"):call("analyze")
    if stage in ("all","aggregate"):
        if not (C.OUT/"evaluation/gallery/output_review_receipt.json").exists():
            print(json.dumps(dict(status="NUMERICAL_COMPLETE_REVIEW_PENDING",next="Actual AI_PROXY review of all 32 image pages; never infer review from rendering")))
            return
        call("evaluate","--stage","aggregate")
    print(json.dumps(dict(status="STAGE_COMPLETE",stage=stage)))
if __name__=="__main__":main()
