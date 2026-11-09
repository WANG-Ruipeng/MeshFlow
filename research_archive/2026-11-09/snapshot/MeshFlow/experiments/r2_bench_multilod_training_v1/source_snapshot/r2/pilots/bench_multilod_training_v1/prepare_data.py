"""Resumable, exactly 39 versions x 2 augmentations CPU OT qualification."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import time
import numpy as np
from . import common as C
from . import data as D


def same_array(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def ot_gate():
    import torch
    from r2.pilots.bench_natural_n_recipe_v1 import runtime as R
    D.require(not torch.cuda.is_initialized(), "CPU gate cannot run after CUDA initialization")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    manifest=C.read(D.MANIFEST_PATH)
    schedules={arm:C.read(D.schedule_path(arm)) for arm in C.ARMS}
    by_case={r["case_id"]:r for r in manifest["train"]}
    first={}
    for arm in C.ARMS:
        for entry in schedules[arm]["steps"]:
            for row in entry["samples"]:
                first.setdefault((row["case_id"],row["augmentation"]),(arm,entry["stage_step"],row))
    D.require(len(first)==78 and len(by_case)==39, "Expected exactly 39 version identities x CLEAN/X_WIDTH")
    receipts=[];new_calls=0
    source_hashes={str(D.MANIFEST_PATH):C.sha(D.MANIFEST_PATH),str(R.__file__):C.sha(R.__file__),
                   str(D.__file__):C.sha(D.__file__)}
    for (case_id,augmentation),(arm,step,row) in sorted(first.items()):
        identity=dict(case_id=case_id,augmentation=augmentation,arm=arm,stage_step=step,
                      selected_row_sha256=D.canonical_hash(row),case_sha256=row["case_sha256"],
                      runtime_sha256=source_hashes[str(R.__file__)])
        key=hashlib.sha256((case_id+"|"+augmentation).encode()).hexdigest()[:20]
        folder=C.OUT/"data/cpu_ot_checks"/key
        receipt_path=folder/"receipt.json";attempt_path=folder/"attempt.json"
        if receipt_path.exists():
            receipt=C.read(receipt_path)
            D.require(receipt["identity"]==identity and receipt["status"]=="PASS", "Completed CPU OT receipt identity differs")
            receipts.append(receipt);continue
        D.require(not attempt_path.exists(), "Unfinished CPU OT attempt exists; preserve it and record recovery before recomputation: "+key)
        case=D.load_case(by_case[case_id])
        original_C=case["C"].copy()
        C.write_new(attempt_path,dict(identity=identity,status="STARTED",production_OT_calls_planned=1,model_forwards=0))
        start=time.perf_counter()
        sample=R.sample_for_training(case,row,"PLAIN")
        proof=R.verify_sample(sample,"NATIVE")
        elapsed=time.perf_counter()-start;new_calls+=1
        n,k=int(sample["N"]),int(sample["K"])
        D.require((n,k)==(row["N"],row["K"]),"Runtime selected N/K differs")
        scaled=case["full_target"].copy();scaled[...,0]*=np.float32(row["alpha"])
        ids,cp=sample["source_face_ids"],sample["source_corner_permutations"]
        D.require(same_array(scaled[ids[:,None],cp],sample["x1"]),"OT source face/corner transport differs")
        known=np.asarray(sample["known_mask"],bool);fixed=np.asarray(sample["anchor_ids"])>=0
        D.require(np.array_equal(np.sort(ids),np.arange(n)) and set(ids[known])==set(case["known_ids"]),"C/free identities changed by OT")
        D.require(same_array(sample["x0"][known],sample["x1"][known]) and
                  same_array(sample["xt"][known],sample["x1"][known]),"C not held exactly")
        D.require(not sample["u"][fixed].any() and int(sample["free_coordinate_count"])==9*(n-k),"FM mask/denominator differs")
        D.require(same_array(sample["u"],sample["x1"]-sample["x0"]),"Velocity target differs")
        D.require(same_array(case["C"],original_C),"Source case was mutated")
        D.require(int(sample["production_OT_calls"])==1 and not torch.cuda.is_initialized(),"One CPU OT/no CUDA contract differs")
        for key_array in ("x0","xt","x1","u"):
            D.require(np.isfinite(sample[key_array]).all(),"Nonfinite prepared sample")
        receipt=dict(identity=identity,status="PASS",N=n,K=k,alpha=row["alpha"],
            production_OT_calls=1,actual_CPU_OT_calls=1,model_forwards=0,optimizer_updates=0,CUDA_initialized=False,
            exact_C=True,exact_selected_Y_face_corner_transport=True,source_case_unchanged=True,
            free_coordinate_count=proof["free_coordinate_count"],source_owner_is_not_FM_layout=True,
            wall_seconds=elapsed)
        C.write_new(receipt_path,receipt);receipts.append(receipt)
        print(json.dumps(dict(done=len(receipts),total=78,case_id=case_id,augmentation=augmentation,N=n,wall_seconds=round(elapsed,4))),flush=True)
    D.require(sum(r["actual_CPU_OT_calls"] for r in receipts)==78,"CPU OT count differs")
    result=dict(status="PASS",schema="BENCH_MULTILOD_ACTUAL_CPU_OT_GATE_V1",conditions=39,
        actual_CPU_OT_calls=78,production_OT_calls=78,prepared_CLEAN=39,prepared_X_WIDTH=39,
        model_forwards=0,model_backwards=0,optimizer_updates=0,CUDA_initialized=False,
        source_hashes=source_hashes,manifest_sha256=C.sha(D.MANIFEST_PATH),
        case_versions=39,exact_C_and_selected_Y_transport=True,actual_mask_denominators=True,
        inherited_runtime_version=R.VERSION,wall_seconds=sum(r["wall_seconds"] for r in receipts),
        receipts=[dict(case_id=r["identity"]["case_id"],augmentation=r["identity"]["augmentation"],N=r["N"],K=r["K"],
                       free_coordinate_count=r["free_coordinate_count"],actual_CPU_OT_calls=1) for r in receipts])
    C.write_new(C.OUT/"data/runtime_cpu_gate.json",result)
    print(json.dumps(dict(status="PASS",new_CPU_OT_calls_this_invocation=new_calls,total_completed_CPU_OT_calls=78)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--metadata-only",action="store_true");args=p.parse_args()
    gate=D.prepare()
    print(json.dumps(dict(status=gate["status"],case_versions=gate["case_versions"],variant_exposure=gate["variant_exposure"])),flush=True)
    if not args.metadata_only: ot_gate()


if __name__=="__main__":main()
