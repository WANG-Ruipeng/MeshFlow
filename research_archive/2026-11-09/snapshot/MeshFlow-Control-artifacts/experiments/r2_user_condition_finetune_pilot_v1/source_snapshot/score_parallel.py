"""Parallel CPU dispatch over frozen evaluation.score_one; unchanged aggregation.
Each worker owns one unique label/request score path. Parent owns progress.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="2"
os.environ["CUDA_VISIBLE_DEVICES"]=""
FROZEN_EVALUATION_SHA="171c16b24d5079d85c9a60971a77e2c1b521ee9a67fca1f458d820e1f25cd23b"

def evaluation():
    name="_user_pilot_frozen_evaluation"
    if name in sys.modules:return sys.modules[name]
    path=Path(__file__).with_name("evaluation.py")
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN_EVALUATION_SHA:raise ValueError("Frozen evaluator changed")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def worker(args):
    out,q,receipt,query,registration_sha=args
    return evaluation().score_one(out,q,receipt,query,registration_sha)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True);parser.add_argument("--workers",type=int,choices=(1,2),default=1)
    args=parser.parse_args(argv);e=evaluation();out=e.root(args.out)
    reg=e.load_registration(out);receipts=e.collect_outputs(out);rb={(r["label"],r["request_id"]):r for r in receipts}
    queries={q["condition_id"]:q for q in reg["queries"]}
    for query in queries.values():e.load_queries(query)
    tasks=[(str(out),q,rb[label,q["request_id"]],queries[q["condition_id"]],e.digest(out/"evaluation/registration.json"))
           for label in e.LABELS for q in reg["requests"]]
    assert len(tasks)==84 and len({(t[2]["label"],t[1]["request_id"]) for t in tasks})==84
    started=time.perf_counter();completed=[];errors=[]
    if args.workers==1:
        for task in tasks:
            row=worker(task);completed.append({"label":row["label"],"request_id":row["request_id"],"status":row["status"]})
            e.write(out/"evaluation/parallel_score_progress.json",{"completed":len(completed),"target":84,"workers":1,"seconds":time.perf_counter()-started},freeze=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map={pool.submit(worker,t):(t[2]["label"],t[1]["request_id"]) for t in tasks}
            for future in as_completed(future_map):
                label,rid=future_map[future]
                try:
                    row=future.result();completed.append({"label":label,"request_id":rid,"status":row["status"]})
                except Exception as exc:
                    errors.append({"label":label,"request_id":rid,"exception":repr(exc)})
                e.write(out/"evaluation/parallel_score_progress.json",{"completed":len(completed),"errors":errors,"target":84,"workers":2,"seconds":time.perf_counter()-started},freeze=False)
                print(json.dumps({"stage":"parallel_score","completed":len(completed),"errors":len(errors),"target":84}),flush=True)
    dispatch_seconds=time.perf_counter()-started
    if errors:
        e.write(out/"evaluation/parallel_score_receipt.json",{"status":"BLOCKED","rows":completed,"errors":errors,
             "workers":args.workers,"worker_threads":2,"dispatch_wall_seconds":dispatch_seconds,"source_sha256":e.digest(__file__)},freeze=False)
        raise RuntimeError("CPU worker errors preserved; no implicit rescoring")
    # Existing score_one rows are identity/hash-checked cache reads. Only original
    # aggregate, pairs and free-diversity code then runs, preserving definitions.
    result=e.score_all(out)
    receipt={"status":"COMPLETE" if result["scored"]==84 else "PARTIAL_BLOCKED","registered":84,"scored":result["scored"],
      "workers":args.workers,"worker_threads":2,"dispatch_wall_seconds":dispatch_seconds,"total_wall_seconds":time.perf_counter()-started,
      "timing_scope":"Shared-host concurrent CPU wall time, not exclusive compute benchmark",
      "evaluation_sha256":FROZEN_EVALUATION_SHA,"source_sha256":e.digest(__file__),"rows":sorted(completed,key=lambda r:(r["label"],r["request_id"])),
      "model_calls":0,"GPU":0,"raw_modified":False,"original_score_definition_changed":False}
    e.write(out/"evaluation/parallel_score_receipt.json",receipt,freeze=False)
    print(json.dumps({k:v for k,v in receipt.items() if k!="rows"}),flush=True)

if __name__=="__main__":main()
