"""Run a frozen panel; completed artifacts are reused, incomplete calls never silently retry."""
import argparse,json,os
from pathlib import Path
from meshflow_control.runtime import gpu_worker
from meshflow_control.accounting import Budget
from meshflow_control.artifacts import file_sha256,atomic_json
from meshflow_control.checkpoints import load_start,load_checkpoint
from meshflow_control.sampling import sample
from meshflow_control.mesh_io import write_raw
import numpy as np
import torch
torch.set_num_threads(4)

def main():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","label","requests","out","ledger"):p.add_argument("--"+name,required=True)
    p.add_argument("--cache-only",action="store_true",help="Export trusted existing START records on CPU only; no model load")
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.label not in ("START","A500","B500","C500","A1000","B1000","C1000"):raise ValueError("Unregistered label")
    request_path=Path(a.requests);source=json.loads(request_path.read_text());rows=source["rows"]
    if a.label.endswith("500"):rows=[r for r in rows if r["panel"]=="P1"]
    source_checkpoint_hash=file_sha256(a.checkpoint);completed=[];pending=[]
    def finish(row,dst,status):
        row.update(raw_path=str(dst/"raw.npz"),raw_sha256=file_sha256(dst/"raw.npz"),result_path=str(dst/"run.json"),
            checkpoint_sha256=source_checkpoint_hash,generation_status=status)
        atomic_json(dst/"receipt.json",row);completed.append(row)
    for request in rows:
        row=dict(request,label=a.label,source_manifest_sha256=file_sha256(request_path));dst=out/a.label/row["request_id"];receipt=dst/"receipt.json"
        if file_sha256(row["input_path"])!=row["input_sha256"]:raise RuntimeError("Input changed")
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if any(old.get(k)!=v for k,v in row.items()):raise RuntimeError("Saved complete request identity differs")
            if old["checkpoint_sha256"]!=source_checkpoint_hash or old["input_sha256"]!=row["input_sha256"]:
                raise RuntimeError("Resume identity differs")
            if file_sha256(old["raw_path"])!=old["raw_sha256"]:raise RuntimeError("Saved RAW changed")
            completed.append(old);continue
        if (dst/"started.json").exists():
            raise RuntimeError("Unreceipted attempted request requires explicit engineering recovery: "+str(dst))
        if a.label=="START" and row.get("historical_start"):
            old=row["historical_start"]
            if old.get("status")!="VERIFIED_CACHE" or old.get("checkpoint_sha256")!=source_checkpoint_hash:raise RuntimeError("Historical checkpoint identity differs")
            if file_sha256(old["raw_path"])!=old["raw_sha256"]:raise RuntimeError("Historical RAW changed")
            with np.load(old["raw_path"],allow_pickle=False) as d:
                raw=d["output"] if "output" in d else d["raw"];trajectory=d["path"] if "path" in d else d["trajectory"]
                original_z=d["z"] if "z" in d else d["initial_gaussian"]
                if raw.dtype!=np.float32 or trajectory.dtype!=np.float32 or original_z.dtype!=np.float32:raise RuntimeError("Historical precision differs")
            with np.load(row["input_path"],allow_pickle=False) as d:C=d["C"];z=d["z"]
            if original_z.reshape(z.shape).tobytes()!=z.tobytes():raise RuntimeError("Historical actual Gaussian differs")
            raw=np.asarray(raw,np.float32).reshape(row["N"],9);trajectory=np.asarray(trajectory,np.float32).reshape(51,row["N"],9)
            if raw.tobytes()!=trajectory[-1].tobytes() or trajectory[0,row["K"]:].tobytes()!=z[0,row["K"]:].tobytes():raise RuntimeError("Historical trajectory endpoints differ")
            audit=dict(cache_reused=True,actual_model_forwards=0,current_generation_seconds=None,
                historical_raw_sha256=old["raw_sha256"],checkpoint_sha256=source_checkpoint_hash,
                known_all51_bitwise=all(x[:row["K"]].tobytes()==C.tobytes() for x in trajectory),
                N=row["N"],K=row["K"],request_id=row["request_id"])
            if not audit["known_all51_bitwise"]:raise RuntimeError("Cached C path mismatch")
            write_raw(dst,raw,C,z,trajectory,audit);finish(row,dst,"VERIFIED_CACHE")
        else:pending.append((row,dst))
    if pending and not a.cache_only:
        with gpu_worker(a.ledger,"sampling_"+a.label):
            model,_,payload=load_start(a.checkpoint,device="cuda",trainable=False) if a.label=="START" else load_checkpoint(a.checkpoint,device="cuda",trainable=False)
            if a.label!="START":
                arm=a.label[0];step=int(a.label[1:])
                if payload["pilot_step"]!=step or model.readout_mode!={"A":"none","B":"mixed","C":"clean"}[arm]:
                    raise RuntimeError("Checkpoint and requested evaluation label differ")
            budget=Budget(a.ledger)
            for row,dst in pending:
                with np.load(row["input_path"],allow_pickle=False) as d:C=d["C"].copy();z=d["z"].copy()
                dst.mkdir(parents=True,exist_ok=True)
                atomic_json(dst/"started.json",dict(checkpoint_sha256=source_checkpoint_hash,input_sha256=row["input_sha256"],budget=budget.snapshot()))
                sample(model,C,row["N"],seed=row["noise_seed"],z=z,budget=budget,out=dst,
                    context={"label":a.label,"request_id":row["request_id"]})
                finish(row,dst,"GENERATED")
                print(json.dumps({"label":a.label,"completed":len(completed),"expected":len(rows),"request_id":row["request_id"]}),flush=True)
    manifest=dict(schema="direct_context_completed_outputs_v1",source_manifest_sha256=file_sha256(request_path),
        checkpoint_sha256=source_checkpoint_hash,label=a.label,rows=sorted(completed,key=lambda r:r["request_id"]))
    if a.cache_only and torch.cuda.is_initialized():raise RuntimeError("CPU cache export initialized CUDA")
    atomic_json(out/(a.label+("_cached_partial.json" if a.cache_only else "_completed.json")),manifest)
    print(json.dumps({"status":"CACHED_PARTIAL" if a.cache_only else "COMPLETE","label":a.label,"outputs":len(completed)}))
if __name__=="__main__":main()
