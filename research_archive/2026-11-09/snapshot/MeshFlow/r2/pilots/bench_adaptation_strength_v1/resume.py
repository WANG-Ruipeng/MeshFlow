"""Recoverable bounded dispatch entry. Default is read-only status."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from .common import ROOT, OUT, ARMS, read, sha, write_new, resources


def alive(job):
    import psutil
    try:
        p = psutil.Process(job["pid"])
        return p.is_running() and abs(p.create_time() - job["process_created"]) < 1
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def status():
    labels = read(OUT / "registration.json")["evaluation_labels"]
    training = {}
    for arm in ARMS:
        p = OUT / "training" / arm / "status.json"
        r = read(p) if p.exists() else {}
        training[arm] = {k:r.get(k) for k in ("status","local_step","global_step","elapsed_seconds","error")}
        if r.get("last"):
            training[arm]["last"] = r["last"]
    jobs = [read(p) for p in sorted((OUT / "dispatch").glob("job_*.json"))]
    return dict(training=training, jobs=[dict(stage=j["stage"],identity=j["identity"],pid=j["pid"],alive=alive(j),log=j["log"]) for j in jobs],
                RAW={lab:len(list((OUT/"evaluation"/lab).glob("*/raw/receipt.json"))) for lab in labels},
                POST={lab:len(list((OUT/"evaluation"/lab).glob("*/post/receipt.json"))) for lab in labels},
                report=(OUT/"REPORT.md").exists(),complete=(OUT/"completion.json").exists())


def execute(stage, identity, resume_training=False, observed_peak_mib=4500.):
    scope = read(OUT/"registration.json")
    jobsdir = OUT/"dispatch";jobsdir.mkdir(parents=True,exist_ok=True)
    oldjobs = [read(p) for p in jobsdir.glob("job_*.json")]
    if any(alive(j) and (j["stage"],j["identity"]) == (stage,identity) for j in oldjobs):
        raise RuntimeError("An active worker already owns this stage")
    if stage == "train":
        if identity not in ARMS:
            raise ValueError(identity)
        gate=read(OUT/"training_gate.json")
        if gate["status"]!="PASS" or gate["registration_sha256"]!=sha(OUT/"training_registration.json"):
            raise ValueError("CPU gate / registration mismatch")
        sp=OUT/"training"/identity/"status.json"
        s=read(sp) if sp.exists() else {}
        if s.get("status")=="COMPLETE":
            return s
        if s and (not resume_training or s.get("status") not in ("SAFE_SEGMENT","INTERRUPTED_SAFE")):
            raise RuntimeError("Existing attempt cannot be replayed implicitly")
        if resume_training and not s:
            raise RuntimeError("No safely saved state")
        argv=["training","--arm",identity,"--execute"]+(["--resume"] if resume_training else [])
        go=OUT/"training_dispatch_go.json"
        if not go.exists():
            write_new(go,dict(authorized=True,training_registration_sha256=sha(OUT/"training_registration.json"),
                              arms=list(ARMS),total_optimizer_updates=4500,automatic_retry=False))
    elif stage in ("raw","post"):
        if identity not in scope["evaluation_labels"]:
            raise ValueError(identity)
        argv=["sample" if stage=="raw" else "post_d0","--label",identity]
    else:
        raise ValueError(stage)
    snapshot=resources()
    gpu=snapshot["gpu"]
    if gpu["returncode"]:
        raise RuntimeError("GPU status unavailable")
    fields=gpu["stdout"].strip().splitlines()[0].split(",")
    total,free,util=(float(fields[i].strip().split()[0]) for i in (1,3,4))
    headroom=max(2048.,.1*total)
    slots=int(max(0.,free-headroom)//(1.2*observed_peak_mib))
    if slots<1 or snapshot["RAM_available"]<8*(1<<30):
        raise RuntimeError("RESOURCE_WAIT: no safe GPU/RAM slot; no task started")
    stamp=str(time.time_ns())
    log=jobsdir/f"{stage}_{identity}_{stamp}.log"
    command=[sys.executable,"-B","-u","-m",__package__+"."+argv[0],*argv[1:]]
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1",
             OMP_NUM_THREADS="2" if stage=="train" else "1",
             MKL_NUM_THREADS="2" if stage=="train" else "1",OPENBLAS_NUM_THREADS="1",
             PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/"r2/src"))
    with log.open("x",encoding="utf-8") as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    import psutil
    record=dict(stage=stage,identity=identity,pid=p.pid,process_created=psutil.Process(p.pid).create_time(),
                argv=command,log=str(log),resources=snapshot,estimated_process_peak_mib=observed_peak_mib,
                headroom_mib=headroom,incremental_slots_estimate=slots,observed_gpu_utilization=util,
                automatic_retry=False,unrelated_processes_untouched=True,
                actual_owned_GPU_workers=1+sum(alive(j) for j in oldjobs))
    write_new(jobsdir/f"job_{stage}_{identity}_{stamp}.json",record)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("train","raw","post"));p.add_argument("--identity")
    p.add_argument("--resume-training",action="store_true")
    p.add_argument("--observed-peak-mib",type=float,default=4500.)
    a=p.parse_args()
    print(json.dumps(execute(a.stage,a.identity,a.resume_training,a.observed_peak_mib) if a.stage else status(),
                     ensure_ascii=False,indent=2),flush=True)
