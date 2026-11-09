"""Root-dispatched fixed D endpoint curves. Never trains or repeats a rollout.

Seven isolated workers total 60 BF16/Euler50 generations and 140 teacher
forwards. Raw3000 is reused from A/C and is deliberately not an allowed label.
"""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse, json, time, traceback
import numpy as np
import torch
from native_t1.artifacts import file_sha256, state_sha256, array_hash
from .common import read_json, write_json, registered_seeds, load_model, load_cases, noise_bank
from .sampling import sample, new_counts
from .replay import read_stage_gate, load_replay_checkpoint, ENDPOINTS

LABELS=("G0", "R_B_raw250", "R_J_raw250", "R_B_EMA250", "R_J_EMA250", "R_B_EMA500", "R_J_EMA500")


def label_spec(label):
    if label not in LABELS:raise ValueError("Only seven registered D labels; raw3000 must reuse A/C")
    if label=="G0":return dict(label=label,arm=None,kind="raw",additional_step=0,cumulative_updates=2500,full_cases=4)
    arm=label[:3];ending=label[4:];kind="EMA" if ending.startswith("EMA") else "raw";step=int(ending[len(kind):])
    return dict(label=label,arm=arm,kind=kind,additional_step=step,cumulative_updates=2500+step,full_cases=20 if step==500 else 4)


def plan_D(label,seeds):
    spec=label_spec(label)
    if len(seeds)!=2 or len(set(seeds))!=2:raise ValueError("Exactly two registered Gaussian seeds required")
    patches=range(5) if spec["full_cases"]==20 else (0,4)
    chosen=seeds if spec["full_cases"]==20 else seeds[:1]
    return [dict(model=label,parent=parent,patch=patch,seed=int(seed),K=12,N=112,precision="bf16",nfe=50,
                 phase="D",checkpoint_kind=spec["kind"],cumulative_updates=spec["cumulative_updates"])
            for parent in (0,1) for patch in patches for seed in chosen]


def require_replay(out,label):
    spec=label_spec(label);gate=read_stage_gate(out)
    if spec["arm"] is None:return dict(status="PASS",ABC_gate=gate,checkpoint=None,replay="NOT_APPLICABLE_COMMON_G0")
    path=Path(out)/"stage_d"/spec["arm"]/"replay.json"
    if not path.is_file():raise RuntimeError("D endpoint evaluation blocked: completed replay is missing")
    r=read_json(path);arm=spec["arm"]
    if (r.get("status")!="PASS_EXACT_REPLAY" or r.get("arm")!=arm or r.get("completed_updates")!=500
            or r.get("EMA_evaluation_allowed") is not True or r.get("raw500_bitwise_reference_equal") is not True
            or r.get("raw500_state_sha256")!=ENDPOINTS[arm][2]):
        raise RuntimeError("D endpoint evaluation blocked: raw replay is not exact")
    for key,value in (("optimizer_updates",500),("EMA_updates",500),("forward_returns",4000),("backward_returns",4000),("OT_returns",4000)):
        if r.get("counts",{}).get(key)!=value:raise RuntimeError("Replay completed-update accounting differs")
    record=r["checkpoints"][str(spec["additional_step"])][spec["kind"]]
    if (record.get("arm")!=arm or record.get("kind")!=spec["kind"]
            or record.get("completed_updates")!=spec["additional_step"]
            or record.get("cumulative_updates")!=spec["cumulative_updates"]):raise RuntimeError("D checkpoint metadata differs")
    checkpoint=Path(record["path"])
    if not checkpoint.is_file() or file_sha256(checkpoint)!=record["sha256"]:raise RuntimeError("D checkpoint file changed")
    return dict(status="PASS",ABC_gate=gate,replay_path=str(path.resolve()),replay_sha256=file_sha256(path),checkpoint=record)


def _save_npz(path,**arrays):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    tick=time.perf_counter()
    with path.open("xb") as stream:np.savez_compressed(stream,**arrays)
    return dict(path=str(path),sha256=file_sha256(path),bytes=path.stat().st_size,archive_write_seconds=time.perf_counter()-tick)


def run_D(out,label):
    """Exactly one registered label; caller must schedule resources and dependencies."""
    from .teacher import load_teacher_bank, teacher_curve
    spec=label_spec(label);out=Path(out).resolve();proof=require_replay(out,label)
    seeds=list(registered_seeds(out));plan=plan_D(label,seeds)
    folder=out/"D_evaluation"/label;folder.mkdir(parents=True,exist_ok=False);path=folder/"results.json"
    counts=new_counts();teacher_counts={};tick=time.perf_counter();model=None
    r=dict(status="RUNNING",stage="D_evaluation",label=label,spec=spec,seeds=seeds,gate=proof,
           formal_expected=len(plan),formal_completed=0,teacher_expected=20,rows=[],counts=counts,
           teacher_counts=teacher_counts,extra_full_replays=0,optimizer_updates=0,OT_calls=0,
           timing=dict(load_seconds=0.,sampling_seconds=0.,archive_write_seconds=0.,teacher_seconds=0.))
    write_json(path,r)
    try:
        cases=load_cases();noises=noise_bank(out);samples,bank=load_teacher_bank(out)
        if set(noises)!=set(seeds):raise RuntimeError("D noise archive differs from registered seeds")
        r["teacher_bank"]=bank["artifact"]
        start=time.perf_counter()
        if label=="G0":model,identity=load_model("G0")
        else:
            cp=proof["checkpoint"]
            model,identity=load_replay_checkpoint(cp["path"],expected_arm=spec["arm"],expected_kind=spec["kind"],expected_sha256=cp["sha256"])
            if identity["state_sha256"]!=cp["state_sha256"]:raise RuntimeError("D strict restored state differs")
        r["timing"]["load_seconds"]=time.perf_counter()-start;r["model_identity"]=identity
        before=state_sha256(model);r["state_before"]=before;device=next(model.parameters()).device
        for item in plan:
            if counts["full_rollout_attempts"]>=spec["full_cases"]:raise RuntimeError("D label full-generation cap exhausted")
            r["active_request"]=item;write_json(path,r)
            gaussian=noises[item["seed"]].detach().clone().to(device)
            case=cases[item["parent"],item["patch"]];C=torch.from_numpy(case["C"].copy()).to(device)
            raw,states,audit=sample(model,gaussian,C,nfe=50,precision="bf16",counts=counts)
            key=f"{label}_P{item['parent']}_patch{item['patch']}_seed{item['seed']}_bf16_Euler50"
            saved=_save_npz(folder/(key+".npz"),raw=raw,states=states,state_indices=np.arange(51,dtype=np.int64),
                            initial_gaussian=gaussian.cpu().numpy(),C=C.cpu().numpy())
            saved["path"]=str(Path(saved["path"]).relative_to(out))
            r["rows"].append(dict(item,key=key,audit=audit,noise_sha256=array_hash(gaussian),**saved))
            r["formal_completed"]+=1;r["timing"]["sampling_seconds"]+=audit["seconds"]
            r["timing"]["archive_write_seconds"]+=saved["archive_write_seconds"];write_json(path,r)
        if counts["complete_rollouts"]!=spec["full_cases"] or counts["model_forward_returns"]!=50*spec["full_cases"]:
            raise RuntimeError("D label generation accounting differs")
        r["active_request"]="fixed_teacher_curve20";write_json(path,r);start=time.perf_counter()
        teacher=teacher_curve(model,samples,model_name=label,out=folder,ledger=teacher_counts,cases=cases)
        r["timing"]["teacher_seconds"]=time.perf_counter()-start
        teacher_path=folder/"stage_d/teacher"/(label+"_bf16.json")
        r["teacher"]=dict(path=str(teacher_path.relative_to(out)),sha256=file_sha256(teacher_path),
                           status=teacher["status"],completed_states=teacher["completed_states"],counts=teacher["counts"])
        if (teacher["status"]!="PASS" or teacher_counts.get("teacher_forward_returns")!=20
                or teacher_counts.get("velocity_leaf_autograd_returns",0) or teacher_counts.get("parameter_autograd_returns",0)):
            raise RuntimeError("D teacher fixed budget/zero-gradient contract differs")
        after=state_sha256(model)
        if before!=after or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError("D evaluation mutated frozen model")
        r.update(status="PASS",state_after=after,active_request=None)
    except BaseException as exc:
        r.update(status="STOPPED",error=repr(exc),traceback=traceback.format_exc())
        if hasattr(exc,"partial_path"):
            r["failed_partial_states"]=_save_npz(folder/"failed_partial_states.npz",states=exc.partial_path,
                                                 state_indices=np.asarray(exc.partial_state_indices,dtype=np.int64))
        raise
    finally:
        r["timing"]["wall_seconds"]=time.perf_counter()-tick
        r["total_counts"]=dict(model_forward_attempts=counts["model_forward_attempts"]+teacher_counts.get("teacher_forward_attempts",0),
                               model_forward_returns=counts["model_forward_returns"]+teacher_counts.get("teacher_forward_returns",0),
                               complete_rollouts=counts["complete_rollouts"],teacher_states=teacher_counts.get("teacher_forward_returns",0),
                               model_backward_calls=0,optimizer_updates=0,OT_calls=0,autograd_calls=0,extra_full_replays=0)
        r["count_note"]="total_counts combines disjoint sampler and teacher_counts; do not add total_counts a second time. Raw3000 has no worker and must reuse A/C."
        write_json(path,r)
        if model is not None:
            del model
            import gc;gc.collect();torch.cuda.empty_cache()
    return r


def plan_fixture():
    """Pure scalar planning checks; does not load raw/teacher assets or a model."""
    seeds=[10401,10402];plans={name:plan_D(name,seeds) for name in LABELS};checks=0
    for name,rows in plans.items():
        assert len(rows)==label_spec(name)["full_cases"];checks+=1
        assert len({(v["parent"],v["patch"],v["seed"]) for v in rows})==len(rows);checks+=1
    assert sum(len(x) for x in plans.values())==60;checks+=1
    assert sum(sum(r["nfe"] for r in x) for x in plans.values())==3000;checks+=1
    assert len(plans)*20==140;checks+=1
    assert all(x["patch"] in (0,4) and x["seed"]==10401 for name,rows in plans.items() if label_spec(name)["full_cases"]==4 for x in rows);checks+=1
    try:label_spec("R_J_raw500")
    except ValueError:checks+=1
    else:raise AssertionError("raw500 must not generate twice")
    return dict(status="PASS",checks=checks,labels=len(plans),planned_full_generations=60,planned_sampling_forwards=3000,
                planned_teacher_forwards=140,planned_total_forwards=3140,model_calls=0,OT_calls=0,autograd_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);parser.add_argument("--label",choices=LABELS,required=True)
    args=parser.parse_args(argv);result=run_D(args.out,args.label)
    print(json.dumps(dict(label=args.label,status=result["status"],counts=result["total_counts"])),flush=True)

if __name__=="__main__":main()
