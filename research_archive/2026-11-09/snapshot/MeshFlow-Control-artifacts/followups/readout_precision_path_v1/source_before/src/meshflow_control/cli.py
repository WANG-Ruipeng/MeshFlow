"""Lazy public entry points: help/config do not import torch or access assets."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(prog="meshflow-control",description="Native triangle-conditioned MeshFlow research")
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("check-config");c.add_argument("--config",required=True)
    c=sub.add_parser("inspect");c.add_argument("--checkpoint",required=True)
    c=sub.add_parser("prepare");c.add_argument("--source-root",required=True);c.add_argument("--out",required=True)
    c=sub.add_parser("train");c.add_argument("--config",required=True);c.add_argument("--data-manifest",required=True)
    c.add_argument("--stream-root",required=True);c.add_argument("--start-checkpoint",required=True);c.add_argument("--out",required=True)
    c.add_argument("--ledger",required=True);c.add_argument("--resume");c.add_argument("--stop-step",type=int,choices=[500,1000],default=1000)
    c=sub.add_parser("sample");c.add_argument("--checkpoint",required=True);c.add_argument("--start",action="store_true")
    c.add_argument("--condition",required=True);c.add_argument("--condition-key",default="C");c.add_argument("--num-faces",type=int,required=True)
    c.add_argument("--seed",type=int,required=True);c.add_argument("--candidates",type=int,default=1);c.add_argument("--out",required=True)
    c.add_argument("--ledger",required=True)
    for name in ("evaluate","render"):
        c=sub.add_parser(name);c.add_argument("--manifest",required=True);c.add_argument("--out",required=True)
        if name=="evaluate":c.add_argument("--workers",type=int,default=2)
    a=p.parse_args(argv)
    if a.command=="check-config":
        from .config import load_config
        print(json.dumps(load_config(a.config),indent=2));return
    if a.command=="inspect":
        import torch
        cp=torch.load(a.checkpoint,map_location="cpu",weights_only=True,mmap=True)
        fields=("schema_version","schema","readout","pilot_step","conditional_total_step","model_config","source_start_sha256")
        print(json.dumps({k:cp[k] for k in fields if k in cp},indent=2))
        if torch.cuda.is_initialized():raise RuntimeError("Inspect initialized CUDA")
        return
    if a.command=="prepare":
        from .data.prepare import import_legacy_data
        source=Path(a.source_root); result=import_legacy_data(source/"native_t1/runs/control_mvp48_v1",source/"native_t1/runs/chair_domain_control_v1",a.out,source_repo=source)
    elif a.command=="train":
        from .config import load_config
        from .runtime import gpu_worker
        from .accounting import Budget
        from .training.trainer import train
        config=load_config(a.config)
        with gpu_worker(a.ledger,"train_"+config["arm"]):
            result=train(config=config,data_manifest=a.data_manifest,stream_root=a.stream_root,
                start_checkpoint=a.start_checkpoint,out=a.out,budget=Budget(a.ledger),
                stop_step=a.stop_step,resume=a.resume,device="cuda")
    elif a.command=="sample":
        from .runtime import gpu_worker
        from .accounting import Budget
        from .checkpoints import load_start,load_checkpoint
        from .mesh_io import load_condition
        from .sampling import sample
        if not 1<=a.candidates<=320:raise ValueError("Invalid explicit candidate budget")
        with gpu_worker(a.ledger,"sample"):
            model,_,_=load_start(a.checkpoint,device="cuda",trainable=False) if a.start else load_checkpoint(a.checkpoint,device="cuda")
            C=load_condition(a.condition,a.condition_key);budget=Budget(a.ledger);result=[]
            for j in range(a.candidates):
                out=Path(a.out) if a.candidates==1 else Path(a.out)/("candidate_"+str(j))
                if (out/"run.json").exists():raise FileExistsError(out)
                _,_,audit=sample(model,C,a.num_faces,seed=a.seed+j,budget=budget,out=out,context={"candidate":j})
                result.append(audit)
    elif a.command=="evaluate":
        from .evaluation.evaluate import evaluate
        result=evaluate(a.manifest,a.out,workers=a.workers)
    elif a.command=="render":
        from .evaluation.render import render
        result=render(a.manifest,a.out)
    print(json.dumps(result,indent=2,default=str))
