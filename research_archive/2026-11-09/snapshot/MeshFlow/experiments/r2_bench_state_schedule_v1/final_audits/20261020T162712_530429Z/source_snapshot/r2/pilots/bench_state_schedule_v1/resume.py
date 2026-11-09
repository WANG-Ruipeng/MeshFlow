"""Recoverable bounded stage entry. No automatic rollback or extra candidates."""
import argparse
import os
import subprocess
import sys
import time
from .common import OUT, ARMS, read, resources, write_new


def run(args):
    package = "r2.pilots.bench_state_schedule_v1."
    if args.stage == "status":
        result = {}
        for arm in ARMS:
            p = OUT/"training"/arm/"status.json"
            result[arm] = read(p) if p.exists() else {"status":"NOT_STARTED"}
        print(result)
        return
    snap = resources()
    write_new(OUT/"resume_resources"/f"{time.time_ns()}_{args.stage}.json", snap)
    extra = []
    if args.stage == "train":
        if args.arm not in ARMS:
            raise ValueError("--arm required")
        p = OUT/"training"/args.arm/"status.json"
        if p.exists():
            status = read(p)
            if status["status"] == "COMPLETE":
                print("Training complete, no rerun.")
                return
            if status["status"] != "INTERRUPTED_SAFE":
                raise RuntimeError("Live/failed/uncertain work is never replayed by this entry")
            extra = ["--resume"]
        module, argv = "training", ["--arm",args.arm,*extra]
    elif args.stage in ("raw","d0"):
        if args.arm not in ARMS:
            raise ValueError("--arm required")
        module, argv = ("sample" if args.stage=="raw" else "post_d0"), ["--label",args.arm+"_S1500"]
    elif args.stage in ("score-raw","score-post"):
        module, argv = "evaluation", [args.stage.split("-")[1],"--workers",str(args.workers)]
        if args.arm:
            argv += ["--labels",args.arm+"_S1500"]
    elif args.stage == "gallery":
        module, argv = "gallery", ["run","--workers",str(args.workers)]
    else:
        module, argv = args.stage, []
    cmd = [sys.executable,"-B","-m",package+module,*argv]
    print("Running bounded stage:",cmd,flush=True)
    subprocess.run(cmd,check=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=("status","train","raw","d0","score-raw","score-post",
        "collect","gallery","training_summary"))
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    if not 1<=a.workers<=16:
        p.error("CPU worker count outside this fixed bound")
    run(a)
