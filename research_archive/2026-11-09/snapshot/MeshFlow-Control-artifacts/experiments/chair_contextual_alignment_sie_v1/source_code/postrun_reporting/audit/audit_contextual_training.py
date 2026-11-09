"""Read-only contextual-alignment training audit. No model, OT or sampling calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

START_FILE = "d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80"
START_STATE = "166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972"
HEAD_COUNT = 3934720
TRAIN_KINDS = ("train_forward","train_backward","aligner_forward","aligner_backward","optimizer_update","gradient_observation_backward")
LIMITS = dict(train_forward=24000,train_backward=24000,aligner_forward=8000,aligner_backward=8000,
 optimizer_update=3000,gradient_observation_backward=12,preflight_forward=24,preflight_backward=16,
 teacher_forward=4096,teacher_image=4096,rollout_attempt=200,sampling_forward=10000)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def jhash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def lines(path,live=False):
    p=Path(path)
    if not p.exists():return []
    raw=p.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not live:raise AssertionError("Incomplete JSONL tail: "+str(p))
        raw=raw[:raw.rfind(b"\n")+1]
    return [json.loads(v) for v in raw.decode("utf-8-sig").splitlines() if v.strip()]


def csvrows(path,live=False):
    p=Path(path)
    if not p.exists():return []
    raw=p.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not live:raise AssertionError("Incomplete CSV tail: "+str(p))
        raw=raw[:raw.rfind(b"\n")+1]
    return list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))


def require(condition,message):
    if not condition:raise AssertionError(message)


def close(a,b,tolerance=2e-7):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=tolerance*max(1.,abs(float(a)),abs(float(b))),
            "Numerical log reconciliation differs: "+str((a,b)))


def source_audit(run,repo):
    records=read(run/"audit/code_frozen_before_training.json")
    bad=[]
    for r in records:
        p=repo/r["Path"]
        if not p.is_file() or digest(p)!=r["Sha256"].lower():bad.append(r["Path"])
    require(not bad,"Frozen source mismatch: "+repr(bad))
    return dict(status="PASS",files=len(records),source_manifest_sha256=digest(run/"audit/code_frozen_before_training.json"),
                missing_or_mismatched=bad)


def ledger_audit(run,live):
    entries=lines(run/"audit/budget/calls.jsonl",live)
    counts=Counter();returns=Counter();errors=[]
    waiting=Counter();completed=Counter()
    metadata=Counter()
    for row in entries:
        event=row["event"]
        if event=="RESERVE":
            kind=row["kind"];require(kind in LIMITS,"Unknown budget category "+kind)
            counts[kind]+=1
            require(row["ordinal"]==counts[kind],"Reservation ordinals are not contiguous")
            if kind != "rollout_attempt":
                waiting[(kind,canonical(row.get("context",{})))]+=1
        elif event in ("RETURN","ERROR"):
            key=(row["kind"],canonical(row.get("context",{})))
            require(waiting[key]>0,"Return/error has no matching reserved call")
            waiting[key]-=1
            if event=="RETURN":
                returns[row["kind"]]+=1;completed[key]+=1
            else:errors.append(row)
        else:
            metadata[event]+=1
    for kind,count in counts.items():require(count<=LIMITS[kind],"Budget exceeded: "+kind)
    unresolved={str(k):n for k,n in waiting.items() if n}
    if not live:require(not unresolved,"Unresolved actual model/optimizer calls")
    require(not errors,"Actual call errors exist; explicit engineering accounting review needed")
    return dict(status="PASS_PREFIX" if live else "PASS",reservations=dict(counts),returns=dict(returns),
                errors=errors,unresolved_snapshot_calls=unresolved,noncall_events=dict(metadata),
                reservation_only_units=["rollout_attempt"],teacher_image_semantics="Per-image reservations and returns within batched teacher forward; not additional model forwards"),completed


def start_audit(path,arm):
    a=read(path)
    require(a["source_start_sha256"]==START_FILE and a["source_model_state_sha256"]==START_STATE,"Wrong START provenance")
    require(a["old_tensor_count"]==194 and a["old_parameter_count"]==130401155,"Wrong generator capacity")
    require(a["old_tensors_equal"] and a["optimizer_restored"] and a["original_optimizer_step"]==5000,"Original optimizer not restored")
    mapping=a["parameter_name_mapping"]
    require(len(mapping)==194 and len({v["saved_optimizer_id"] for v in mapping})==194,"Original optimizer mapping not bijective")
    require(all(v["old_name"]==v["new_name"] for v in mapping),"Original parameter names changed")
    require(a["readout_mode"]=="none" and a["new_parameter_count"]==0,"Unexpected readout")
    require(a["aligner_parameter_count"]==(0 if arm=="A" else HEAD_COUNT),"Wrong head capacity")
    require(a["alignment_arm"]==arm,"START receipt arm differs")
    return dict(status="PASS",generator_parameters=130401155,generator_tensors=194,original_step=5000,
                aligner_initial_sha256=a["aligner_initial_sha256"],source_receipt_sha256=digest(path))


def preflight_audit(run,ledger):
    p=read(run/"preflight/preflight.json")
    require(p["status"]=="PASS","Preflight not PASS")
    require(len(p["rows"])==2 and p["sample_indices"]==[0,2],"Preflight fixed input identities differ")
    require(p["optimizer_updates"]==0 and p["old_optimizer_step"]==5000,"Preflight changed optimizer")
    require(p["generator_parameters_unchanged"] and p["aligner_parameters_unchanged"] and p["identical_B_C_head"],"Preflight parameter identity differs")
    delta={k:p["budget_after"]["counts"][k]-p["budget_before"]["counts"][k] for k in LIMITS}
    require(delta["preflight_forward"]==6 and delta["preflight_backward"]==10 and delta["optimizer_update"]==0,"Wrong preflight actual call delta")
    for row in p["rows"]:
        require(row["B_C_message_identical"] and row["capture_released"] and row["original_FM_formula_equal"],"Capture/FM check failed")
        f,k=row["domains"]["free"],row["domains"]["known"]
        require(f["message_sha256"]==k["message_sha256"],"B/C captured different messages")
        for d in (f,k):
            require(d["generator_prediction_bitwise_equal"] and d["FM_bitwise_equal"] and d["direct_unselected_gradient_zero"],"Preflight branch changed prediction or token domain")
            require(d["auxiliary_late_all_unused"] and d["auxiliary_early_gradient_norm"]>0 and d["head_gradient_norm"]>0,"Alignment gradient path missing")
            require(d["head_prediction_dtype"]=="torch.float32" and d["captured_dtype"]=="torch.bfloat16","Precision changed")
    return dict(status="PASS",actual_generator_forward=6,actual_zero_update_backward=10,optimizer_updates=0,
                teacher_before=p["budget_before"]["counts"],head_capacity=p["head_parameters"],
                sample_indices=p["sample_indices"],early_aux_gradient_norms=[r["domains"][d]["auxiliary_early_gradient_norm"] for r in p["rows"] for d in ("free","known")])


def arm_audit(run,arm,expected,plan,ledger_entries,live):
    folder=run/"training"/arm
    if not (folder/"start_import_audit.json").exists():
        require(live,"Missing arm "+arm);return dict(status="PENDING"),[],[]
    initial=start_audit(folder/"start_import_audit.json",arm)
    logs=csvrows(folder/"training_log.csv",live)
    exposure=csvrows(folder/"exposure.csv",live)
    diagnostics=lines(folder/"diagnostics.jsonl",live)
    events=lines(folder/"events.jsonl",live)
    require(not any(r["event"]=="TRAIN_FAILED" for r in events),"Failed training attempt "+arm)
    done=sorted(int(r["step"]) for r in events if r["event"]=="UPDATE_DONE")
    steps=min(len(logs),len(diagnostics),len(done),10) if live else 1000
    if steps==0:return dict(status="PENDING",initial=initial),[],[]
    require(done[:steps]==list(range(1,steps+1)),"Update events are not contiguous")
    if not live:
        require(len(logs)==len(diagnostics)==len(done)==1000 and len(exposure)==8000,"Final training/exposure row counts differ")
        require(read(folder/"status.json")["status"]=="COMPLETE","Arm not complete")
    selected=exposure[:8*steps]
    require(len(selected)==8*steps,"Missing committed exposure rows")
    diag={int(d["step"]):d for d in diagnostics[:steps]}
    bystep={i+1:selected[i*8:(i+1)*8] for i in range(steps)}
    for i,s in enumerate(selected):
        require(int(s["sample_index"])==i and int(s["step"])==i//8+1,"Exposure order differs")
        p=plan[i];e=expected[i]
        for key in ("uid","task_id"):
            require(s[key]==p[key],"Exposure parent/condition differs")
        for key in ("N","K"):require(int(s[key])==p[key],"Exposure N/K differs")
        require(float(s["alpha"])==p["alpha"],"Exposure augmentation differs")
        for key in ("input_bytehash","label_bytehash","shared_input_sha256","ot_cache_key"):
            require(s[key]==e[key],"Frozen actual input/label/OT identity differs")
        require(int(s["free_coordinates"])==e["free_coordinate_count"]==9*(int(s["N"])-int(s["K"])),"FM free scalar count differs")
        require(s["ot_cache_status"]=="HIT","Formal training recomputed OT")
    completed_calls=Counter()
    for row in ledger_entries:
        ctx=row.get("context",{})
        if row.get("event")=="RETURN" and row.get("kind") in TRAIN_KINDS and ctx.get("arm")==arm and int(ctx.get("step",0))<=steps:
            completed_calls[(row["kind"],int(ctx["step"]))]+=1
    observations=[]
    for index,l in enumerate(logs[:steps],1):
        require(int(l["step"])==index and l["arm"]==arm,"Training log order differs")
        samples=bystep[index];d=diag[index]
        require(len({s["uid"] for s in samples})==8 and sum(float(s["alpha"])==1. for s in samples)==4,"Parent/alpha batch distribution differs")
        denom=sum(int(s["free_coordinates"]) for s in samples)
        require(int(l["free_coordinate_denominator"])==denom and int(l["alignment_denominator"])==4,"Loss denominator differs")
        require(int(l["aligned_samples"])==(0 if arm=="A" else 4),"Alignment sample count differs")
        require(int(l["old_adamw_step"])==5000+index and int(l["conditional_total_step"])==5000+index,"Generator optimizer cursor differs")
        require(l["aligner_adamw_step"]==("" if arm=="A" else str(index)),"Aligner optimizer cursor differs")
        require(int(l["unused_generator_parameters"])==int(l["unused_aligner_parameters"])==0,"Unused parameters")
        for hashfield,source in (("shared_batch_sha256","shared_input_sha256"),("input_batch_sha256","input_bytehash"),("label_batch_sha256","label_bytehash")):
            require(l[hashfield]==jhash([s[source] for s in samples]),"Batch identity aggregate differs")
        for kind,n in (("train_forward",8),("train_backward",8),("optimizer_update",1),
                       ("aligner_forward",0 if arm=="A" else 4),("aligner_backward",0 if arm=="A" else 4)):
            require(completed_calls[(kind,index)]==n,"Actual per-update call count differs: "+str((arm,index,kind)))
        subs=d["sample_losses"];require(len(subs)==8,"Missing micro losses")
        close(sum(s["FM_contribution"] for s in subs),l["FM"])
        raw_sum=0.
        for s,e in zip(subs,samples):
            require(s["sample_index"]==int(e["sample_index"]) and s["batch_free_denominator"]==denom and s["alignment_denominator"]==4,"Micro denominator/identity differs")
            eligible=arm!="A" and float(e["alpha"])==1.
            require((s["raw_align"] is not None)==eligible,"Augmented sample received alignment")
            if eligible:
                raw_sum+=s["raw_align"]
                close(s["weighted_align"],.025*s["raw_align"])
                require(s["selected_token_count"]==(int(e["N"])-int(e["K"]) if arm=="B" else int(e["K"])),"Alignment selected wrong token domain")
            else:require(s["weighted_align"]==0.,"Unexpected auxiliary loss")
        close(l["raw_align"],raw_sum/4.)
        close(l["weighted_align"],.1*raw_sum/4.)
        close(l["total_loss"],float(l["FM"])+float(l["weighted_align"]))
        require(math.isfinite(float(l["generator_gradient_norm"])),"Nonfinite generator gradient")
        if arm!="A":
            require(float(l["aligner_gradient_norm"])>0. and float(l["aligner_actual_update_norm"])>0.,"No actual head learning")
        if index in (1,100,500,1000):
            require(float(l["generator_actual_update_norm"])>0.,"Missing genuine parameter update observation")
        else:require(l["generator_actual_update_norm"]=="","Unregistered generator parameter observation")
        obs=d["gradient_observations"]
        expected_obs=int(arm!="A" and index in (1,100,1000))
        require(len(obs)==expected_obs and completed_calls[("gradient_observation_backward",index)]==2*expected_obs,"Sparse gradient observation count differs")
        observations += [dict(arm=arm,step=index,**o) for o in obs]
        for event in ("INPUT_BEGIN","INPUT_DONE","UPDATE_BEGIN","UPDATE_DONE"):
            require(sum(r["event"]==event and int(r.get("step",0))==index for r in events)==1,"Missing/duplicated durable update events")
    parents=Counter(s["uid"] for s in selected)
    if not live:
        require(len(parents)==32 and set(parents.values())=={250},"Final parent exposures differ")
        saved=[r["step"] for r in events if r["event"]=="CHECKPOINT_DONE"]
        require(500 in saved and 1000 in saved,"Missing fixed checkpoints")
    return dict(status="PASS_PREFIX" if live else "PASS",steps=steps,exposures=8*steps,
        initial=initial,parent_exposures=dict(parents),actual_calls={k:sum(v for (kind,_),v in completed_calls.items() if kind==k) for k in TRAIN_KINDS},
        FM_first=float(logs[0]["FM"]),FM_last=float(logs[steps-1]["FM"]),
        alignment_last=float(logs[steps-1]["raw_align"]),peak_allocated_bytes=max(int(x["cuda_peak_allocated_bytes"]) for x in logs[:steps]),
        peak_reserved_bytes=max(int(x["cuda_peak_reserved_bytes"]) for x in logs[:steps]),
        gradient_observations=observations),selected,logs[:steps]


def state_hash(state):
    h=hashlib.sha256()
    import torch
    for name,tensor in sorted(state.items()):
        tensor=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkpoint_audit(run,arm):
    import torch
    require(not torch.cuda.is_initialized(),"Audit must not initialize CUDA")
    result=[]
    initial=read(run/"training"/arm/"start_import_audit.json")
    original_names=["generator."+v["new_name"] for v in initial["parameter_name_mapping"]]
    events=lines(run/"training"/arm/"events.jsonl")
    for step in (500,1000):
        saved=[v["checkpoint"] for v in events if v["event"]=="CHECKPOINT_DONE" and v["step"]==step]
        require(len(saved)==1,"Fixed checkpoint save event not unique")
        path=Path(saved[0]["path"]);sha=digest(path)
        require(sha==saved[0]["sha256"],"Saved checkpoint file changed")
        p=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        require(p["schema"]=="meshflow_contextual_alignment_training_v1" and p["update_in_progress"] is False,"Wrong training schema")
        require(p["arm"]==arm and p["config"]["arm"]==arm and p["pilot_step"]==step and p["conditional_total_step"]==5000+step,"Checkpoint arm/step differs")
        require(p["stream_state"]["batch_index"]==step and p["stream_state"]["sample_index"]==8*step,"Saved stream cursor differs")
        generator=p["generator_state"]
        require(len(generator)==194 and sum(t.numel() for t in generator.values())==130401155,"Generator capacity differs")
        require(all(t.dtype==torch.float32 for t in generator.values()),"Generator parameter dtype differs")
        actual=state_hash(generator);require(actual==p["generator_state_sha256"],"Generator tensor state hash differs")
        groups=p["optimizer"]["param_groups"];names=p["optimizer_parameter_names"]
        require(names[0]==original_names,"Original named optimizer mapping changed")
        require(len(groups)==(1 if arm=="A" else 2),"Wrong optimizer groups")
        head=p["aligner_state"]
        require((head is None)==(arm=="A"),"Wrong auxiliary state ownership")
        if head is not None:
            require(sum(t.numel() for t in head.values())==HEAD_COUNT and state_hash(head)==p["aligner_state_sha256"],"Head state count/hash differs")
            require(names[1]==["aligner."+n for n in head],"Head optimizer order differs")
        all_states=p["optimizer"]["state"]
        count=0
        for i,(group,nms) in enumerate(zip(groups,names)):
            require((group["lr"],tuple(group["betas"]),group["weight_decay"])==(1e-5 if i==0 else 1e-4,(.9,.95),0.),"Optimizer recipe differs")
            for ident,name in zip(group["params"],nms):
                parameter=generator[name[len("generator."):]] if i==0 else head[name[len("aligner."):]]
                st=all_states[ident]
                require(set(st)=={"step","exp_avg","exp_avg_sq"} and float(st["step"])==(5000+step if i==0 else step),"Actual AdamW step differs")
                require(all(st[k].dtype==torch.float32 and st[k].shape==parameter.shape for k in ("exp_avg","exp_avg_sq")),"Actual optimizer moment identity/dtype differs")
                count+=1
        result.append(dict(step=step,path=str(path),sha256=sha,bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
            generator_state_sha256=actual,generator_tensors=194,
            generator_parameters=130401155,optimizer_state_tensors=count,old_adamw_step=5000+step,
            aligner_adamw_step=None if head is None else step,aligner_parameters=0 if head is None else HEAD_COUNT))
        del p,generator,head,all_states,st
    path=run/"inference"/f"{arm}_generator_step1000.pt"
    receipt=read(run/"inference"/f"{arm}_export_receipt.json")
    require(digest(path)==receipt["sha256"],"Generator export file differs")
    p=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    require(p["schema"]=="meshflow_contextual_alignment_generator_v1" and "aligner_state" not in p and "optimizer" not in p,"Inference contains training branch")
    require(p["arm"]==arm and p["pilot_step"]==1000 and p["conditional_total_step"]==6000,"Inference endpoint identity differs")
    require(p["generator_state_sha256"]==result[-1]["generator_state_sha256"]==state_hash(p["generator_state"]),"Export not same final generator")
    require(p["source_training_checkpoint_sha256"]==result[-1]["sha256"],"Export provenance differs")
    capture=read(run/"audit"/f"{arm}_export_capture_check.json")
    require(capture["status"]=="PASS" and capture["capture_on_off_bitwise"] and capture["no_remaining_hooks"] and capture["generator_file_sha256"]==receipt["sha256"],"Export capture check missing or failed")
    require(not torch.cuda.is_initialized(),"Audit initialized CUDA")
    return dict(status="PASS",training=result,export=dict(path=str(path),sha256=receipt["sha256"],bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
          schema=p["schema"],same_generator_state=True,capture_on_off_bitwise=True))


def run_audit(run,repo,live):
    run,repo=Path(run),Path(repo)
    began=time.perf_counter()
    output=dict(status="RUNNING",scope="first 10 completed steps per available arm" if live else "complete training and exports",
                model_forward=0,model_backward=0,optimizer_update=0,CPU_only=True,checks={})
    target=run/"audit"/("training_first_updates.json" if live else "training_endpoint_independent.json")
    try:
        output["checks"]["frozen_sources"]=source_audit(run,repo)
        ledger,completed=ledger_audit(run,live)
        output["checks"]["call_ledger"]=ledger
        output["checks"]["preflight"]=preflight_audit(run,ledger)
        entries=lines(run/"audit/budget/calls.jsonl",live)
        expected={r["sample_index"]:r for r in read(run/"training_stream/paired_inputs.json")["records"]}
        plan=read(run/"training_stream/train_plan.json")["records"]
        observed={};observedlogs={};arms={}
        for arm in "ABC":
            arms[arm],observed[arm],observedlogs[arm]=arm_audit(run,arm,expected,plan,entries,live)
        output["checks"]["arms"]=arms
        headhash=[v["initial"]["aligner_initial_sha256"] for k,v in arms.items() if k in "BC" and "initial" in v]
        if len(headhash)==2:require(headhash[0]==headhash[1],"B/C initial head tensors differ")
        paired=min(len(v) for v in observed.values())
        keys=("sample_index","uid","task_id","N","K","ratio","alpha","t","alignment_eligible","free_coordinates",
              "shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key")
        for i in range(paired):
            require(all({k:observed[a][i][k] for k in keys}=={k:observed["A"][i][k] for k in keys} for a in "BC"),"Cross-arm actual per-sample identity differs")
        paired_batches=min(len(v) for v in observedlogs.values())
        for i in range(paired_batches):
            for k in ("shared_batch_sha256","input_batch_sha256","label_batch_sha256"):
                require(len({observedlogs[a][i][k] for a in "ABC"})==1,"Cross-arm batch hash differs")
        output["checks"]["cross_arm"]=dict(status="PASS" if paired else "PENDING",paired_sample_positions=paired,
            actual_exposure_rows=sum(len(v) for v in observed.values()),paired_batches=paired_batches,
            B_C_initial_head_equal=len(headhash)==2 and headhash[0]==headhash[1])
        if not live:
            require(paired==8000 and paired_batches==1000,"Incomplete final pairing")
            for k,n in dict(train_forward=24000,train_backward=24000,aligner_forward=8000,
                           aligner_backward=8000,optimizer_update=3000,gradient_observation_backward=12).items():
                require(ledger["returns"].get(k,0)==ledger["reservations"].get(k,0)==n,"Final actual call total differs: "+k)
            output["checks"]["checkpoints"]={a:checkpoint_audit(run,a) for a in "ABC"}
            with (run/"audit/training_parent_exposure.csv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=["arm","parent_id","exposures"]);w.writeheader()
                for a,info in arms.items():
                    for uid,n in sorted(info["parent_exposures"].items()):w.writerow(dict(arm=a,parent_id=uid,exposures=n))
        output["status"]="PASS_PREFIX_PENDING_ENDPOINTS" if live else "PASS"
    except Exception as exc:
        output.update(status="BLOCKED",error=repr(exc),traceback=traceback.format_exc())
    output.update(wall_seconds=time.perf_counter()-began,audit_source=str(Path(__file__).resolve()),
                  audit_source_sha256=digest(__file__),created_unix=time.time())
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        previous=target.with_name(target.stem+"."+str(time.time_ns())+".previous.json")
        target.rename(previous)
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=output["status"],path=str(target),sha256=digest(target),
        arms={a:r.get("steps",0) for a,r in output["checks"].get("arms",{}).items()},error=output.get("error"))))
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--repo",required=True)
    p.add_argument("--live",action="store_true")
    a=p.parse_args()
    result=run_audit(a.run,a.repo,a.live)
    raise SystemExit(0 if result["status"].startswith("PASS") else 1)
