"""One trainer for all three arms; an external controller owns the GPU lock.

No work occurs on import. Eight free-only FM microbatches form each update;
only whole completed updates may be checkpointed or safely resumed.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path
import signal
import time
import traceback
import torch
from ..data.dataset import collate, model_inputs
from ..data.stream import HybridStream
from ..data.io import read_json, rows, atomic_json, append_event, digest, json_hash
from .losses import masked_fm_loss

MODES={"A":"none","B":"mixed","C":"clean"}
LOG_COLUMNS=("arm","step","conditional_total_step","FM","free_coordinate_denominator",
    "shared_batch_sha256","input_batch_sha256","label_batch_sha256","grad_norm_before_clip",
    "old_gradient_norm","new_gradient_norm","new_actual_update_norm","unused_parameters",
    "old_adamw_step","new_adamw_step","input_seconds","forward_seconds","backward_seconds",
    "optimizer_seconds","update_seconds","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes")
EXPOSURE_COLUMNS=("arm","step","sample_index","uid","task_id","N","K","ratio","alpha","t",
    "free_coordinates","shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key","ot_cache_status")

def validate_recovery(events,checkpoint_step):
    """Reject lost/uncovered updates or half-executed batches; never silently replay."""
    kinds=("INPUT_BEGIN","INPUT_DONE","UPDATE_BEGIN","UPDATE_DONE")
    found={kind:[] for kind in kinds}
    for e in events:
        if e.get("event") in found:
            step=int(e["step"])
            if step>checkpoint_step:
                raise RuntimeError("Uncheckpointed training attempt at step "+str(step))
            found[e["event"]].append(step)
    expected=list(range(1,checkpoint_step+1))
    if any(sorted(v)!=expected for v in found.values()):
        raise RuntimeError("Checkpoint cursor is not covered by unique complete input/update events")

def validate_optimizer(model,optimizer,step):
    if not isinstance(optimizer,torch.optim.AdamW) or model.training:
        raise ValueError("Trainable eval-mode model with restored AdamW required")
    named=dict(model.named_parameters())
    if not named or not all(p.requires_grad and p.dtype==torch.float32 for p in named.values()):
        raise ValueError("All original and added parameters must remain trainable FP32")
    owned=[p for g in optimizer.param_groups for p in g["params"]]
    if len(owned)!=len(named) or len({id(p) for p in owned})!=len(named) or {id(p) for p in owned}!={id(p) for p in named.values()}:
        raise ValueError("Optimizer must own every parameter exactly once")
    for group in optimizer.param_groups:
        if (group["lr"],tuple(group["betas"]),group["weight_decay"])!=(1e-5,(.9,.95),0.):
            raise ValueError("Frozen AdamW hyperparameters differ")
    for name,p in named.items():
        s=optimizer.state[p];expected=step+(0 if name.startswith("readouts.") else 5000)
        if set(s)!={"step","exp_avg","exp_avg_sq"} or float(s["step"])!=expected:
            raise ValueError("AdamW cursor differs: "+name)
        if any(s[k].shape!=p.shape or s[k].dtype!=torch.float32 for k in ("exp_avg","exp_avg_sq")):
            raise ValueError("AdamW moments lost parameter identity: "+name)
    return named

def _sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)

def _call(budget,kind,fn,context,device):
    _sync(device);tick=time.perf_counter()
    def complete():
        r=fn();_sync(device);return r
    result=budget.call(kind,complete,context=context)
    return result,time.perf_counter()-tick

def _norm(tensors):
    values=[t.detach().float().norm() for t in tensors if t is not None]
    return float(torch.stack(values).norm()) if values else 0.

def _scalar(value):
    return value.detach().cpu().item() if isinstance(value,torch.Tensor) else value

def update(model,optimizer,samples,budget,arm,step,*,observe=False):
    """One genuine authorized update. Synthetic CPU tests never call this function."""
    from ..models.native import native_execution
    if arm not in MODES or model.readout_mode!=MODES[arm]:
        raise ValueError("Arm and model readout source differ")
    if len(samples)!=8 or len({s["object_id"] for s in samples})!=8 or sum(s["alpha"]==1. for s in samples)!=4:
        raise ValueError("Require eight distinct parents and four alpha=1 samples")
    named=validate_optimizer(model,optimizer,step-1)
    device=next(iter(named.values())).device
    if device.type!="cuda":raise ValueError("Formal trainer requires explicit owned CUDA worker")
    denominator=sum(s["free_coordinate_count"] for s in samples)
    if denominator!=9*sum(int((s["valid_mask"] & ~s["known_mask"]).sum()) for s in samples):
        raise ValueError("Effective free-coordinate normalization differs")
    optimizer.zero_grad(set_to_none=True)
    tick=time.perf_counter()
    result=dict(FM=0.,free_coordinate_denominator=denominator,forward_seconds=0.,backward_seconds=0.,
                optimizer_seconds=0.,sample_losses=[],readout_observations=[])
    previous_observer=model.readout_observer
    try:
        for micro,sample in enumerate(samples):
            batch=collate([sample],device)
            context=dict(arm=arm,step=step,conditional_total_step=5000+step,microbatch=micro,
                sample_index=sample["sample_index"],input_bytehash=sample["input_bytehash"],
                label_bytehash=sample["label_bytehash"])
            observations=[]
            model.readout_observer=observations.append if observe else None
            with native_execution(model,coordinates=(batch["xt"],batch["t"]),autocast=True):
                velocity,seconds=_call(budget,"train_forward",lambda:model(*model_inputs(batch)).float(),context,device)
                result["forward_seconds"]+=seconds
                loss=masked_fm_loss(velocity,batch["u"],batch["valid_mask"],batch["known_mask"],denominator=denominator)
                if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite FM loss")
                contribution=float(loss.detach())
                _,seconds=_call(budget,"train_backward",loss.backward,context,device)
                result["backward_seconds"]+=seconds
            result["FM"]+=contribution
            result["sample_losses"].append(dict(sample_index=sample["sample_index"],FM_contribution=contribution,
                free_coordinates=sample["free_coordinate_count"],batch_denominator=denominator))
            for r in observations:
                result["readout_observations"].append(dict(microbatch=micro,sample_index=sample["sample_index"],
                    **{k:_scalar(v) for k,v in r.items()}))
            del velocity,loss,batch
    finally:
        model.readout_observer=previous_observer
    unused=[name for name,p in named.items() if p.grad is None]
    if unused:raise RuntimeError("Unused trainable parameters: "+",".join(unused))
    old=[p for n,p in named.items() if not n.startswith("readouts.")]
    new={n:p for n,p in named.items() if n.startswith("readouts.")}
    result["unused_parameters"]=len(unused)
    result["old_gradient_norm"]=_norm([p.grad for p in old])
    result["new_gradient_norm"]=_norm([p.grad for p in new.values()])
    # Only the small new branch is copied for actual parameter-delta diagnostics.
    previous={n:p.detach().clone() for n,p in new.items()}
    result["readout_parameter_gradients"]={n:float(p.grad.detach().norm()) for n,p in new.items()}
    result["grad_norm_before_clip"]=float(torch.nn.utils.clip_grad_norm_(list(named.values()),1.,error_if_nonfinite=True))
    _,result["optimizer_seconds"]=_call(budget,"optimizer_update",optimizer.step,
        dict(arm=arm,step=step,conditional_total_step=5000+step),device)
    result["readout_parameter_updates"]={n:float((p.detach()-previous[n]).norm()) for n,p in new.items()}
    result["new_actual_update_norm"]=_norm([p.detach()-previous[n] for n,p in new.items()])
    if not all(bool(torch.isfinite(p).all()) for p in named.values()):
        raise FloatingPointError("Nonfinite parameters after optimizer update")
    validate_optimizer(model,optimizer,step)
    result.update(old_adamw_step=5000+step,new_adamw_step=step if new else None,
        update_seconds=time.perf_counter()-tick,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result

def train(config,data_manifest,stream_root,start_checkpoint,out,budget,stop_step=1000,resume=None,device="cuda"):
    """Run to the predeclared 500 or 1000 boundary, under the caller's GPU lock."""
    from ..config import validate_config,config_hash
    from ..checkpoints import load_start,load_checkpoint,save_checkpoint,restore_rng
    config=validate_config(config);arm=config.get("arm")
    if arm not in MODES or config["readout_mode"]!=MODES[arm] or stop_step not in (500,1000):
        raise ValueError("Registered A/B/C and fixed stop500/1000 required")
    data_manifest=Path(data_manifest).resolve()
    if data_manifest.name!="train_manifest.json":raise ValueError("Explicit imported training manifest required")
    out=Path(out).resolve();folder=out/"training"/arm;folder.mkdir(parents=True,exist_ok=True)
    events_path=folder/"events.jsonl";status_path=folder/"status.json"
    if status_path.exists() and read_json(status_path).get("status")=="BLOCKED":
        raise RuntimeError("Prior numerical/incomplete training failure is BLOCKED; no silent retry")
    stream=HybridStream(data_manifest.parent,stream_root,arm)
    paired_path=Path(stream_root)/"paired_inputs.json"
    if not paired_path.exists():raise ValueError("Freeze shared CPU matching/input identities before GPU training")
    paired=read_json(paired_path)
    if paired["plan_sha256"]!=stream.plan_sha256 or paired["data_manifest_sha256"]!=digest(data_manifest):
        raise ValueError("Precomputed paired identities do not match the input stream")
    expected_inputs={r["sample_index"]:r for r in paired["records"]}
    if resume:
        model,optimizer,payload=load_checkpoint(resume,device=device,trainable=True)
        if payload["training_config_hash"]!=config_hash(config) or payload["data_manifest_hash"]!=digest(data_manifest):
            raise ValueError("Resume configuration/data identity differs")
        if payload["readout"]["mode"]!=MODES[arm]:raise ValueError("Resume model arm differs")
        step=payload["pilot_step"];stream.load_state_dict(payload["stream_state"])
        restore_rng(payload["rng"])
    else:
        model,optimizer,payload=load_start(start_checkpoint,mode=MODES[arm],device=device,trainable=True)
        step=0
        atomic_json(folder/"start_import_audit.json",payload)
    if stream.batch_index!=step or step>stop_step:raise ValueError("Training cursor/phase differs")
    validate_recovery(rows(events_path),step)
    validate_optimizer(model,optimizer,step)
    if step==stop_step:
        return dict(status="COMPLETE",arm=arm,step=step,actual_updates_this_call=0,
            checkpoint={"path":str(resume),"sha256":digest(resume)} if resume else None)
    torch.cuda.reset_peak_memory_stats()
    start_step=step;begin=time.perf_counter()
    state=dict(status="TRAINING",arm=arm,step=step,stop_step=stop_step,conditional_total_step=5000+step,
        data_manifest_sha256=digest(data_manifest),plan_sha256=stream.plan_sha256,
        training_config_hash=config_hash(config),budget_before=budget.snapshot())
    stop={"signal":None}
    def interrupt(signum,frame):stop["signal"]=int(signum)
    handlers={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(current):
        append_event(events_path,dict(event="CHECKPOINT_BEGIN",arm=arm,step=current))
        p=out/"checkpoints"/(arm+"_step"+str(current).zfill(4)+".pt")
        receipt=save_checkpoint(p,model,optimizer,step=current,stream_state=stream.state_dict(),
            config=config,data_manifest_hash=digest(data_manifest),
            extra={"arm":arm,"plan_sha256":stream.plan_sha256,"budget":budget.snapshot()})
        append_event(events_path,dict(event="CHECKPOINT_DONE",arm=arm,step=current,checkpoint=receipt))
        atomic_json(folder/"latest.json",receipt)
        return receipt
    try:
        atomic_json(status_path,state)
        with (folder/"training_log.csv").open("a",newline="",encoding="utf-8") as log, (folder/"exposure.csv").open("a",newline="",encoding="utf-8") as exposure:
            writer=csv.DictWriter(log,fieldnames=LOG_COLUMNS);swriter=csv.DictWriter(exposure,fieldnames=EXPOSURE_COLUMNS)
            if log.tell()==0:writer.writeheader()
            if exposure.tell()==0:swriter.writeheader()
            for current in range(step+1,stop_step+1):
                if stop["signal"] is not None:
                    state.update(status="INTERRUPTED_SAFE",signal=stop["signal"],checkpoint=save(step))
                    break
                append_event(events_path,dict(event="INPUT_BEGIN",arm=arm,step=current))
                tick=time.perf_counter();samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick
                for s in samples:
                    expected=expected_inputs[s["sample_index"]]
                    for key in ("shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key","free_coordinate_count"):
                        if s[key]!=expected[key]:raise ValueError("Actual input/label differs from frozen shared identity: "+key)
                audit=stream.last_audit
                append_event(events_path,dict(event="INPUT_DONE",arm=arm,step=current,**audit))
                for s in samples:
                    swriter.writerow(dict(arm=arm,step=current,sample_index=s["sample_index"],
                        uid=s["object_id"],task_id=s["task_id"],N=int(s["y"]),K=s["K"],ratio=s["ratio"],
                        alpha=s["alpha"],t=float(s["t"]),free_coordinates=s["free_coordinate_count"],
                        **{k:s[k] for k in ("shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key","ot_cache_status")}))
                exposure.flush();os.fsync(exposure.fileno())
                append_event(events_path,dict(event="UPDATE_BEGIN",arm=arm,step=current,shared_batch_sha256=audit["shared_batch_sha256"]))
                r=update(model,optimizer,samples,budget,arm,current,observe=current in (1,2,500,1000) or current%50==0)
                step=current
                append_event(events_path,dict(event="UPDATE_DONE",arm=arm,step=step,
                    old_adamw_step=r["old_adamw_step"],new_adamw_step=r["new_adamw_step"],
                    train_forward=8,train_backward=8,optimizer_updates=1,
                    shared_batch_sha256=audit["shared_batch_sha256"]))
                append_event(folder/"diagnostics.jsonl",dict(event="TRAIN_DIAGNOSTICS",arm=arm,step=step,
                    sample_losses=r["sample_losses"],readout_observations=r["readout_observations"],
                    readout_parameter_gradients=r["readout_parameter_gradients"],
                    readout_parameter_updates=r["readout_parameter_updates"]))
                line=dict(arm=arm,step=step,conditional_total_step=5000+step,input_seconds=input_seconds,
                    **{k:audit[k] for k in ("shared_batch_sha256","input_batch_sha256","label_batch_sha256")},
                    **{k:r[k] for k in LOG_COLUMNS if k in r})
                writer.writerow(line);log.flush();os.fsync(log.fileno())
                state.update(step=step,conditional_total_step=5000+step,actual_updates_this_call=step-start_step,
                    last_FM=r["FM"],last_new_gradient_norm=r["new_gradient_norm"],
                    last_new_actual_update_norm=r["new_actual_update_norm"],budget=budget.snapshot())
                if step in (500,1000):state["checkpoint"]=save(step)
                if step%10==0 or step in (1,2,500,1000):atomic_json(status_path,state)
                if step%25==0 or step in (1,2):print(arm,step,"FM",r["FM"],flush=True)
            else:
                state["status"]="COMPLETE"
        if stop["signal"] is not None and state["status"]=="TRAINING":
            state.update(status="INTERRUPTED_SAFE",signal=stop["signal"],checkpoint=save(step))
    except BaseException as exc:
        state.update(status="BLOCKED",error=repr(exc),traceback=traceback.format_exc(),
            step=step,stream_cursor=stream.batch_index,budget=budget.snapshot())
        append_event(events_path,dict(event="TRAIN_FAILED",arm=arm,step=step,error=repr(exc)))
        atomic_json(status_path,state)
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        state.update(wall_seconds=time.perf_counter()-begin,actual_updates_this_call=step-start_step,
            budget_after=budget.snapshot())
        atomic_json(status_path,state)
    return state
