"""One-time official reference process. External audit only, never a package dependency."""
import argparse,json,time
from pathlib import Path
import numpy as np
from meshflow_control import runtime
from meshflow_control.accounting import Budget
import torch
torch.set_num_threads(4)
from native_t1.chair_checkpoint import load_chair_start
from native_t1.model import native_execution
from native_t1.chair_sampling import clamped_sample
from native_t1.artifacts import state_sha256
p=argparse.ArgumentParser();p.add_argument("--requests",required=True);p.add_argument("--start",required=True);p.add_argument("--out",required=True);p.add_argument("--ledger",required=True)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path(a.requests).read_text())["rows"]
selected=[]
for row in rows:
    if row["panel"]=="P1" and row["cohort"]=="OLD" and row["parent_id"] not in [x["parent_id"] for x in selected]:selected.append(row)
    if len(selected)==2:break
if len(selected)!=2:raise RuntimeError("Two fixed real inputs required")
budget=Budget(a.ledger)
with runtime.gpu_worker(a.ledger,"official_reference"):
    model,audit=load_chair_start(a.start,device="cuda");before=state_sha256(model);results=[]
    for i,row in enumerate(selected):
        dst=out/("ref_"+str(i)+".npz")
        if dst.exists():raise FileExistsError("Do not silently replay reference request")
        with np.load(row["input_path"],allow_pickle=False) as d:C=d["C"].astype(np.float32).reshape(-1,3,3);z=d["z"].copy()
        N,K=row["N"],row["K"];state=torch.from_numpy(z.copy()).cuda();cond=torch.from_numpy(C.reshape(K,9)).cuda();state[:,:K]=cond[None]
        t=torch.tensor([.35+.35*i],device="cuda",dtype=torch.float32);y=torch.tensor([N],device="cuda",dtype=torch.int64)
        known=torch.zeros((1,N),device="cuda",dtype=torch.bool);known[:,:K]=True;valid=torch.ones_like(known);outputs={}
        for bf16 in (False,True):
            with torch.no_grad(),native_execution(model,coordinates=state,autocast=bf16):
                value=budget.call("preflight_forward",lambda:model(state,t,y,valid,known),context={"phase":"official_reference","input":i,"bf16":bf16})
            outputs["bf16_output" if bf16 else "fp32_output"]=value.float().cpu().numpy()
        budget.reserve("rollout_attempt",context={"phase":"official_reference","input":i})
        def pre(module,args):budget.reserve("sampling_forward",context={"phase":"official_reference","input":i})
        h=model.register_forward_pre_hook(pre)
        try:raw,path,sample_audit=clamped_sample(model,torch.from_numpy(z.copy()).cuda(),cond)
        finally:h.remove()
        np.savez_compressed(dst,x=state.cpu().numpy(),t=t.cpu().numpy(),y=y.cpu().numpy(),valid=valid.cpu().numpy(),
            known=known.cpu().numpy(),C=C,z=z,raw=raw,trajectory=path,**outputs)
        budget.event("ROLLOUT_COMPLETE",context={"phase":"official_reference","input":i},nfe=50)
        results.append(dict(input=i,request_id=row["request_id"],source=row,reference_path=str(dst),sampling=sample_audit))
        print("OFFICIAL_REFERENCE_COMPLETE",i,flush=True)
    if state_sha256(model)!=before:raise RuntimeError("Official START changed")
    (out/"official_reference.json").write_text(json.dumps(dict(status="PASS",inputs=results,start=audit,budget=budget.snapshot()),indent=2),encoding="utf-8")
