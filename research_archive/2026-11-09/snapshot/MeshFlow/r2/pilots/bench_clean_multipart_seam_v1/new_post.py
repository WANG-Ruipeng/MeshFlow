"""Same frozen CLEAN_FINAL cascade on the two registered new-model layouts."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import argparse, concurrent.futures, multiprocessing, resource, sys, time
from . import post_cpu as P
from .eval_prepare import LABELS
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.chair_back_transfer_v1.evaluate import worker_count

ROOT=P.ROOT


def work(job):
    label,q,regsha=job;rid=q["request_id"]
    folder=ROOT/"evaluation"/label/rid
    baseline=folder/"post/post.npz";receipt=C.read(folder/"post/receipt.json")
    if receipt["post_sha256"]!=C.sha(baseline) or receipt["D0_forwards"]!=1 or receipt["GEO_calls"]!=1:
        raise ValueError("New BASE producer differs")
    raw=folder/"raw/raw.npz";pre=folder/"post/pre_geo.npz"
    for name in ("inference_npz","metadata_npz"):
        if not C.resolve(q[name]).exists():raise FileNotFoundError(q[name])
    cache=ROOT/"base_pair_caches"/label/rid;pairpath=cache/"pairs.json"
    done=cache/"complete.json"
    if done.exists():
        old=C.read(done)
        if old["mesh_sha256"]!=C.sha(baseline):raise ValueError("Paircache identity differs")
        P.E.verify(old["files"])
    else:
        if cache.exists():raise RuntimeError("Partial pair scan retained")
        tick=time.perf_counter()
        C.write(cache/"started.json",dict(mesh_sha256=C.sha(baseline),utc=C.utc(),model_calls=0))
        mesh=C.arrays(baseline);K=int(q["K"])
        # One original finite-pair detector scan, reused by the full BASE score.
        pairs=P.H.scan(mesh["output"],K)
        C.write(pairpath,pairs)
        C.write(done,dict(status="COMPLETE",mesh_sha256=C.sha(baseline),
            original_full_pair_scans=1,model_calls=0,seconds=time.perf_counter()-tick,
            files={str(p):C.sha(p) for p in cache.iterdir() if p.is_file()}))
    return P.process_one(q,label,baseline,pre,raw,pairpath,ROOT/"cpu"/label/rid,True,regsha)


def run(labels,workers=2):
    if any(label not in LABELS for label in labels):raise ValueError("Unknown layout")
    manifest=C.read(ROOT/"eval_manifest.json");qs=manifest["requests"]
    if len(qs)!=48:raise ValueError("Original48 panel required")
    reg=ROOT/"new_cpu_registration.json"
    value=dict(status="FROZEN",version=P.VERSION,labels=list(LABELS),
        requests=qs,eval_manifest_sha256=C.sha(ROOT/"eval_manifest.json"),
        post_source_sha256=C.sha(P.__file__),dispatch_source_sha256=C.sha(__file__),
        chain="original BASE -> HINGE_OPP015 -> ALPHA -> HOLE",no_model_calls=True)
    C.write(reg,value)
    dest=ROOT/("new_cpu_completion_"+"__".join(labels)+".json")
    if dest.exists():
        r=C.read(dest);P.E.verify(r["files"]);return r
    js=[(label,q,C.sha(reg)) for label in labels for q in qs]
    snapshot=C.resources()
    C.write(ROOT/("new_cpu_resources_before_"+str(time.time_ns())+".json"),
        dict(labels=labels,workers_requested=workers,threads_per_worker=1,
             resources=snapshot,scope="Before first authorized CPU cell"))
    tick=time.perf_counter();first=work(js[0])
    snap=C.resources();count=worker_count(snap,first,workers)
    C.write(ROOT/("new_cpu_dispatch_"+str(time.time_ns())+".json"),
        dict(labels=labels,workers=count,threads_per_worker=1,resources_before=snapshot,
             resources_after_first=snap,first_actual_peak_RSS_KiB=first["peak_RSS_KiB"]))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context("spawn")) as pool:
        pending=iter(js[1:]);active={}
        while True:
            while len(active)<count:
                job=next(pending,None)
                if job is None:break
                active[pool.submit(work,job)]=job
            if not active:break
            ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:receipts.append(f.result());del active[f]
    if C.sha(P.__file__)!=value["post_source_sha256"]:raise ValueError("Frozen cascade changed")
    result=dict(status="COMPLETE",requests=len(js),labels=labels,workers=count,
        seconds=time.perf_counter()-tick,model_calls=0,D0_calls=0,
        BASE_pair_scans=len(js),BASE_scores=len(js),final_scores=len(js),
        files={str(ROOT/"cpu"/label/q["request_id"]/"complete.json"):
               C.sha(ROOT/"cpu"/label/q["request_id"]/"complete.json") for label,q,_ in js})
    C.write(dest,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--labels",nargs="+",choices=LABELS,default=list(LABELS))
    p.add_argument("--workers",type=int,choices=(1,2,3,4),default=2)
    a=p.parse_args();print(run(a.labels,a.workers),flush=True)
