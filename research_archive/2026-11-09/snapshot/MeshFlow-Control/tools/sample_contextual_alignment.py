"""Only registered context-alignment endpoint Euler50; independent generator exports."""
from meshflow_control import runtime
from meshflow_control.training.context_budget import Budget
from meshflow_control.artifacts import file_sha256,atomic_json
from meshflow_control.data.io import freeze_json
from meshflow_control.checkpoints import load_start
from meshflow_control.sampling import sample
from pathlib import Path
import argparse,json,time,sys
import numpy as np
import torch

LABELS=("START","A_FM_1000","B_FREE_ALIGN_1000","C_KNOWN_ALIGN_1000")
def run(root,label,checkpoint):
    root=Path(root).resolve();checkpoint=Path(checkpoint).resolve()
    if label not in LABELS:raise ValueError("Unregistered label")
    request_path=root/"data/evaluation/requests.json";doc=json.loads(request_path.read_text());rows=doc["rows"]
    sha=file_sha256(checkpoint);source_sha=file_sha256(request_path);out=root/"outputs";out.mkdir(exist_ok=True)
    complete=[];pending=[];existing={}
    if label=="START":
        cache=json.loads((root/"data/evaluation/start_cached_outputs.json").read_text())
        for r in cache["rows"]:
            for p,h in (("raw_path","raw_sha256"),("result_path","result_sha256")):
                if h in r and file_sha256(r[p])!=r[h]:raise ValueError("Cached output changed")
            if r["label"]!="START":raise ValueError("Invalid cache")
            existing[r["request_id"]]=r
    for request in rows:
        if file_sha256(request["input_path"])!=request["input_sha256"]:raise ValueError("Frozen input changed")
        if request["request_id"] in existing:
            complete.append(existing[request["request_id"]]);continue
        row=dict(request,label=label,checkpoint_sha256=sha,source_manifest_sha256=source_sha)
        dst=out/label/request["request_id"];receipt=dst/"receipt.json"
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if any(saved.get(k)!=v for k,v in row.items()):raise ValueError("Completed request binding differs")
            if file_sha256(saved["raw_path"])!=saved["raw_sha256"] or file_sha256(saved["result_path"])!=saved["result_sha256"]:
                raise ValueError("Completed output changed")
            complete.append(saved);continue
        if (dst/"started.json").exists():raise RuntimeError("Unreceipted attempt requires explicit recovery")
        pending.append((row,dst))
    if pending:
        torch.set_num_threads(4)
        with runtime.gpu_worker(root/"audit/budget","sampling_"+label):
            budget=Budget(root/"audit/budget")
            if label=="START":model,_,audit=load_start(checkpoint,device="cuda",trainable=False)
            else:
                from meshflow_control.training.context_checkpoint import load_generator
                model,audit=load_generator(checkpoint,device="cuda",expected_sha256=sha)
                if audit["arm"]!=label[0] or audit["pilot_step"]!=1000:raise ValueError("Export label mismatch")
                if "meshflow_control.training.context_alignment" in sys.modules or "meshflow_control.teachers.siglip_targets" in sys.modules:raise RuntimeError("Pure inference imported head or teacher")
            atomic_json(root/"audit"/("sampling_load_"+label+".json"),audit)
            if model.readout_mode!="none":raise RuntimeError("Unexpected inference branch")
            for row,dst in pending:
                with np.load(row["input_path"],allow_pickle=False) as d:C=d["C"].copy();z=d["z"].copy()
                dst.mkdir(parents=True,exist_ok=True)
                atomic_json(dst/"started.json",dict(checkpoint_sha256=sha,input_sha256=row["input_sha256"],budget=budget.snapshot()))
                _,_,receipt=sample(model,C,int(row["N"]),seed=int(row.get("noise_seed",row.get("seed"))),z=z,
                    budget=budget,out=dst,context=dict(label=label,request_id=row["request_id"]))
                row.update(raw_path=str(dst/"raw.npz"),raw_sha256=file_sha256(dst/"raw.npz"),
                    result_path=str(dst/"run.json"),result_sha256=file_sha256(dst/"run.json"),generation_status="GENERATED")
                atomic_json(dst/"receipt.json",row);complete.append(row)
                print(json.dumps(dict(label=label,completed=len(complete),expected=len(rows),request_id=row["request_id"])),flush=True)
            del model
    if len(complete)!=48:raise RuntimeError("Expected complete label 48 outputs")
    dest=out/(label+"_completed.json")
    freeze_json(dest,dict(schema="contextual_alignment_completed_outputs_v1",label=label,
        checkpoint_sha256=sha,source_manifest_sha256=source_sha,rows=sorted(complete,key=lambda r:r["request_id"])))
    return dict(path=str(dest),sha256=file_sha256(dest),outputs=len(complete))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--label",required=True);p.add_argument("--checkpoint",required=True)
    a=p.parse_args();print(json.dumps(run(a.out,a.label,a.checkpoint)),flush=True)
