"""Read-only final artifact integrity check. No model forward, training or scoring."""
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
from meshflow_control.mesh_io import read_ply
from meshflow_control.artifacts import file_sha256,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--artifacts",required=True);a=p.parse_args();root=Path(a.artifacts)
    manifest=json.loads((root/"outputs/all_completed.json").read_text());rows=manifest["rows"]
    failures=[];per_output=[];seen=set()
    for r in rows:
        key=(r["label"],r["request_id"])
        if key in seen:failures.append(dict(key=key,error="Duplicate output identity"))
        seen.add(key)
        try:
            if file_sha256(r["raw_path"])!=r["raw_sha256"]:raise ValueError("RAW file hash changed")
            if file_sha256(r["input_path"])!=r["input_sha256"]:raise ValueError("Input hash changed")
            with np.load(r["input_path"],allow_pickle=False) as q:C=q["C"].copy().reshape(-1,3,3);z=q["z"].copy()
            with np.load(r["raw_path"],allow_pickle=False) as q:
                raw=q["raw"].copy().reshape(-1,3,3);path=q["trajectory"].copy();actual_z=q["z"].copy()
                mapping=q["source_corner_to_output_vertex"].copy();faces=q["known_face_indices"].copy()
            ply,indices=read_ply(Path(r["raw_path"]).with_suffix(".ply"))
            N,K=r["N"],r["K"]
            checks=dict(N=len(raw)==N,K=len(C)==K,finite=bool(np.isfinite(raw).all() and np.isfinite(path).all()),
                FP32=raw.dtype==np.float32 and path.dtype==np.float32,
                exact_C=raw[:K].tobytes()==C.tobytes(),exact_PLY=ply.tobytes()==raw.tobytes(),
                all51=path.shape==(51,N,9) and all(v[:K].tobytes()==C.tobytes() for v in path),
                final_state=path[-1].tobytes()==raw.tobytes(),Gaussian=actual_z.shape==z.shape and actual_z.tobytes()==z.tobytes(),
                original_free_start=path[0,K:].tobytes()==z[0,K:].tobytes(),
                corner_mapping=np.array_equal(mapping,np.arange(3*K)),face_mapping=np.array_equal(faces,np.arange(K)),
                PLY_corner_mapping=np.array_equal(indices[:K].reshape(-1),np.arange(3*K)))
            if not all(checks.values()):raise ValueError("Output contract failures: "+str(checks))
            per_output.append(dict(label=r["label"],request_id=r["request_id"],checks=checks,status="PASS"))
        except Exception as e:
            failures.append(dict(key=key,error=repr(e)));per_output.append(dict(label=r["label"],request_id=r["request_id"],status="FAIL",error=repr(e)))
    counts=collections.Counter();by_arm=collections.Counter();returns=collections.Counter();errors=[];complete=0
    for line in (root/"audit/budget/calls.jsonl").read_text().splitlines():
        r=json.loads(line)
        if r["event"]=="RESERVE":
            counts[r["kind"]]+=1
            if r["kind"]=="optimizer_update":by_arm[r["context"]["arm"]]+=1
        elif r["event"]=="RETURN":returns[r["kind"]]+=1
        elif r["event"]=="ERROR":errors.append(r)
        elif r["event"]=="ROLLOUT_COMPLETE":complete+=1
    expected=dict(train_forward=24000,train_backward=24000,optimizer_update=3000,preflight_forward=47,preflight_backward=11,rollout_attempt=278,sampling_forward=13900)
    formal_complete=all(counts[k]==v for k,v in expected.items()) and dict(by_arm)==dict(A=1000,B=1000,C=1000)
    occupancy=[json.loads(s) for s in (root/"audit/budget/gpu_occupancy.jsonl").read_text().splitlines()]
    starts={r["id"]:r for r in occupancy if r["event"]=="START"};ends={r["id"]:r for r in occupancy if r["event"]=="END"}
    gpu_seconds=sum(r["seconds"] for r in ends.values())
    checkpoints=[]
    for arm in "ABC":
        for step in (500,1000):
            fp=root/"checkpoints"/(arm+"_step"+str(step).zfill(4)+".pt")
            checkpoints.append(dict(arm=arm,step=step,path=str(fp),sha256=file_sha256(fp),bytes=fp.stat().st_size))
    result=dict(status="PASS" if len(rows)==288 and not failures and formal_complete and not errors and set(starts)==set(ends) and gpu_seconds<=21600 else "REVIEW_REQUIRED",
        output_count=len(rows),expected_output_count=288,output_counts=dict(collections.Counter(r["label"] for r in rows)),
        generation_counts=dict(collections.Counter(r["generation_status"] for r in rows)),failures=failures,
        per_output=per_output,calls_reserved=dict(counts),calls_returned=dict(returns),expected_calls=expected,
        note="Official-reference sampling reserves and ROLLOUT_COMPLETE records establish its 100 actual forwards; its wrapper does not emit RETURN per step.",
        optimizer_updates_by_arm=dict(by_arm),model_errors=errors,rollout_completions=complete,gpu_owner_seconds=gpu_seconds,
        gpu_owners_closed=set(starts)==set(ends),checkpoints=checkpoints)
    atomic_json(root/"audit/delivery_integrity.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_output","checkpoints")},indent=2))
if __name__=="__main__":main()
