"""Read-only CPU accounting and plots. Final output requires both COMPLETE arms.

Run with --check during training to validate committed prefixes without writing
results. This control script is not imported by the frozen training code.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
ARMS = ("PLAIN1500", "MULTILOD1500")
GROUPS = ("OLD", "FOCUS", "HIGH")
SEEDS = ("noise_seed", "epsilon_seed", "time_seed", "permutation_seed")
LIMITS = dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
              head_forward=0, head_backward=0, head_optimizer_update=0, probe_forward=0)
DEST = OUT / "training_summary"
WARNING = "Different N/targets change the supervised problem. Training loss reduction is not evidence of better generated meshes; use the fixed paired generation evaluation."


def require(ok, message):
    if not ok: raise ValueError(message)


def resolve(value):
    text=str(value).replace("\\", "/")
    for prefix in ("/mnt/e/MeshFlow/", "E:/MeshFlow/"):
        if text.startswith(prefix): return ROOT/text[len(prefix):]
    p=Path(text);return p if p.is_absolute() else ROOT/p


def read(path):
    return json.loads(resolve(path).read_text("utf-8-sig"))


def sha(path):
    h=hashlib.sha256()
    with resolve(path).open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()


def jsonl(path, partial=False):
    path=resolve(path)
    if not path.exists():return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        require(partial,"Incomplete JSONL tail: "+str(path))
        raw=raw.rsplit(b"\n",1)[0]+b"\n" if b"\n" in raw else b""
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def close(a,b,tol=1e-10):
    return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def call_audit(rows,n,final):
    expected={**LIMITS,"train_forward":8*n,"train_backward":8*n,"optimizer_update":n}
    pending={};begins=collections.Counter();done=collections.Counter();seconds=collections.Counter()
    identities=collections.defaultdict(list);timestamps=[];pids=set();attempts=collections.defaultdict(list)
    for row in rows:
        c=row.get("context",{});step=c.get("stage_step",0)
        if not final and step>n:continue
        kind=row["kind"];require(kind in LIMITS,"Unexpected call kind: "+kind)
        require(1<=step<=n,"Call outside registered completed prefix")
        require(c.get("global_step")==25500+step and c.get("stage")=="NATIVE","Call cursor differs")
        key=(row["attempt_id"],row["call_id"])
        event=row["event"]
        if event=="CALL_BEGIN":
            require(key not in pending,"Duplicate open call")
            pending[key]=row;begins[kind]+=1
        elif event=="CALL_DONE":
            require(key in pending and pending[key]["kind"]==kind,"Unmatched completed call")
            del pending[key];done[kind]+=1;seconds[kind]+=float(row["seconds"])
            identities[kind].append((step,c.get("microbatch")))
        else:raise ValueError("Non-successful call ledger event: "+event)
        timestamps.append(float(row["time_unix"]));pids.add(row["pid"])
        attempts[row["attempt_id"]].append(float(row["time_unix"]))
    require(not pending,"Pending calls within completed prefix")
    for kind,count in expected.items():
        require(begins[kind]==done[kind]==count,"Budget mismatch: "+kind)
        wanted=[(step,micro) for step in range(1,n+1) for micro in (range(8) if kind in ("train_forward","train_backward") else (None,))] if count else []
        require(identities[kind]==wanted,"Call identity/order mismatch: "+kind)
    return dict(attempted=dict(begins),successful=dict(done),authorized_or_prefix_counts=expected,
                pending=0,failed=0,instrumented_call_seconds=dict(seconds),pids=sorted(pids),
                first_call_unix=min(timestamps) if timestamps else None,last_call_unix=max(timestamps) if timestamps else None,
                call_wall_span_seconds=max(timestamps)-min(timestamps) if timestamps else 0.,
                attempt_call_spans={k:max(v)-min(v) for k,v in attempts.items()})


def arm_audit(arm,reg,final=False):
    folder=OUT/"training"/arm;state_path=folder/"status.json"
    state=read(state_path) if state_path.exists() else dict(status="NOT_STARTED",stage_step=0)
    n=int(state.get("stage_step",0))
    require(0<=n<=1500,"Unexpected step")
    if final:
        require(state["status"]=="COMPLETE" and n==1500,"Both arms must be COMPLETE at1500")
        require((state["global_step"],state["native_total_step"],state["seam_total_step"])==(27000,15000,0),"Endpoint cursor differs")
        require(state["registration_sha256"]==sha(OUT/"training_registration.json"),"State registration differs")
        require((folder/"attempts"/state["attempt"]/"final_status.json").exists(),"Final resource/status flush not finished")
    schedule=read(reg["schedules"][arm]["path"])
    coefficients=read(reg["coefficient_schedules"][arm]["path"])
    require(sha(reg["schedules"][arm]["path"])==reg["schedules"][arm]["sha256"],"Schedule hash differs")
    require(sha(reg["coefficient_schedules"][arm]["path"])==reg["coefficient_schedules"][arm]["sha256"],"Coefficient hash differs")
    log_rows=jsonl(folder/"training_log.jsonl",not final)
    exposures=jsonl(folder/"exposure.jsonl",not final)
    logs=[r for r in log_rows if r["stage_step"]<=n]
    used=[r for r in exposures if r["stage_step"]<=n]
    require([r["stage_step"] for r in logs]==list(range(1,n+1)),"Training log gaps/duplicates")
    require([(r["stage_step"],r["microbatch"]) for r in used]==[(i,j) for i in range(1,n+1) for j in range(8)],"Exposure identity/order differs")
    if final:require(len(log_rows)==1500 and len(exposures)==12000,"Extra training/exposure rows")
    for log,entry,coef in zip(logs,schedule["steps"],coefficients["steps"]):
        require(log["group_allocation"]==coef,"Runtime group allocation differs from registered coefficients")
        group_sse=collections.Counter();group_counts=collections.Counter();loss=[];fm=[]
        start=(entry["stage_step"]-1)*8
        for micro,(actual,frozen) in enumerate(zip(used[start:start+8],entry["samples"])):
            require(all(actual.get(k)==v for k,v in frozen.items()),"Actual exposure differs from frozen source/version/seed row")
            count=9*(actual["N"]-actual["K"])
            require(actual["all_free_scalar_count"]==count==coef["micro_free_scalar_counts"][micro],"Actual free scalar count differs")
            require(close(actual["loss_gain"],coef["micro_gains"][micro]),"Actual gain differs")
            require(actual["reference_group_mse_weights"]==coef["reference_group_mse_weights"],"Actual reference a_g differs")
            require(actual["actual_batch_denominator_D_W"]==coef["D_W"],"Actual loss denominator differs")
            group_sse[actual["replay_group"]]+=actual["all_free_velocity_sse"]
            group_counts[actual["replay_group"]]+=count
            loss.append(actual["loss_contribution"]);fm.append(actual["unweighted_loss_contribution"])
        require(close(math.fsum(loss),log["objective"]) and close(math.fsum(fm),log["FM"]),"Logged objective does not sum actual micro losses")
        for g in GROUPS:
            require(close(group_sse[g],log["group_velocity_sse"][g]) and group_counts[g]==coef["actual_group_scalar_counts"][g],"Actual group SSE/count mismatch")
        require(log["free_coordinate_denominator"]==coef["D_W"],"Formal update denominator differs")
    calls=call_audit(jsonl(folder/"calls.jsonl",not final),n,final)
    if final:
        for key in ("checkpoint","inference_export"):
            receipt=state[key];path=resolve(receipt["path"])
            require(path.exists() and path.stat().st_size>0 and len(receipt["sha256"])==64,"Formal endpoint artifact missing")
        require(not state["calls"]["failed"] and not state["calls"]["pending_call_ids"],"Endpoint contains pending/failed calls")
        for kind,limit in LIMITS.items():
            require(state["calls"]["attempted"].get(kind,0)==state["calls"]["successful"].get(kind,0)==limit,"Endpoint budget differs")
    return dict(arm=arm,state=state,n=n,logs=logs,exposures=used,calls=calls,schedule=schedule,coefficients=coefficients)


def paired_audit(arms):
    a,b=(arms[x] for x in ARMS);n=min(a["n"],b["n"])
    fields=("parent_id","condition_id","augmentation","replay_group","alpha","scale_xyz","logical_instance_id",*SEEDS)
    for left,right in zip(a["exposures"][:8*n],b["exposures"][:8*n]):
        require(all(left[k]==right[k] for k in fields),"Actual paired scientific streams differ")
        require(left["reference_group_mse_weights"]==right["reference_group_mse_weights"],"Paired group a_g differs")
    return dict(status="PASS",paired_completed_updates=n,paired_completed_instances=8*n,
                same_parent_condition_augmentation_seed=True,same_reference_group_mse_weights=True)


def window_rows(arm,logs,width):
    rows=[]
    for start in range(0,len(logs),width):
        block=logs[start:start+width]
        if not block:continue
        row=dict(arm=arm,window_steps=width,first_step=block[0]["stage_step"],last_step=block[-1]["stage_step"],
                 updates=len(block),objective_mean=statistics.fmean(r["objective"] for r in block),
                 unweighted_FM_mean=statistics.fmean(r["FM"] for r in block),
                 all_free_velocity_sse=math.fsum(r["all_free_velocity_sse"] for r in block),
                 all_free_scalar_count=sum(r["all_free_scalar_count"] for r in block),
                 update_seconds=math.fsum(r["update_seconds"] for r in block),input_seconds=math.fsum(r["input_seconds"] for r in block))
        row["pooled_free_velocity_MSE"]=row["all_free_velocity_sse"]/row["all_free_scalar_count"]
        for g in GROUPS:
            sse=math.fsum(r["group_velocity_sse"][g] for r in block)
            count=sum(r["group_allocation"]["actual_group_scalar_counts"][g] for r in block)
            row[g+"_velocity_sse"],row[g+"_scalar_count"],row[g+"_pooled_velocity_MSE"]=sse,count,sse/count if count else None
        rows.append(row)
    return rows


def exposure_tables(arms):
    parent=collections.defaultdict(lambda:collections.defaultdict(list));nd=collections.Counter();groups=[]
    for arm,info in arms.items():
        for r in info["exposures"]:
            parent[r["parent_id"]][arm].append(r)
            nd[arm,r["replay_group"],r["variant"],r["N"],r["K"]]+=1
        for g in GROUPS:
            rr=[r for r in info["exposures"] if r["replay_group"]==g]
            groups.append(dict(arm=arm,group=g,instances=len(rr),parents=len({r["parent_id"] for r in rr}),
                               LOW=sum(r["variant"]=="LOW" for r in rr),NATURAL=sum(r["variant"]=="NATURAL" for r in rr),
                               free_scalars=sum(r["all_free_scalar_count"] for r in rr)))
    paired=[]
    for pid,byarm in sorted(parent.items()):
        row=dict(parent_id=pid,replay_group=byarm[ARMS[0]][0]["replay_group"],condition_id=byarm[ARMS[0]][0]["condition_id"])
        for arm in ARMS:
            rr=byarm[arm]
            row.update({arm+"__"+k:v for k,v in dict(instances=len(rr),LOW=sum(r["variant"]=="LOW" for r in rr),
                NATURAL=sum(r["variant"]=="NATURAL" for r in rr),CLEAN=sum(r["augmentation"]=="CLEAN" for r in rr),
                X_WIDTH=sum(r["augmentation"]=="X_WIDTH" for r in rr),N_min=min(r["N"] for r in rr),N_max=max(r["N"] for r in rr),
                N_mean=statistics.fmean(r["N"] for r in rr),free_scalars=sum(r["all_free_scalar_count"] for r in rr),
                N_distribution_json=json.dumps(dict(sorted(collections.Counter(r["N"] for r in rr).items())))).items()})
        row["instances_delta"]=row[ARMS[1]+"__instances"]-row[ARMS[0]+"__instances"]
        require(row["instances_delta"]==0,"Actual per-parent exposure changed")
        paired.append(row)
    distribution=[dict(arm=a,replay_group=g,variant=v,N=n,K=k,instances=c) for (a,g,v,n,k),c in sorted(nd.items())]
    return paired,groups,distribution


def resource_tables(arm,info,training_pids):
    folder=OUT/"training"/arm;observations=[];attempts=[];process_peaks=[];device_peaks=[]
    for path in sorted((folder/"attempts").glob("*/resources.json")):
        data=read(path);summ=data["summary"]
        attempts.append(dict(attempt=path.parent.name,resource_summary=summ,final_status=read(path.parent/"final_status.json")))
        if summ.get("process_peak_mib") is not None:process_peaks.append(summ["process_peak_mib"])
        if summ.get("total_device_used_peak_mib") is not None:device_peaks.append(summ["total_device_used_peak_mib"])
        for sample in data["samples"]:
            row=dict(arm=arm,attempt=path.parent.name,time_unix=sample["time_unix"],pid=sample["pid"],process_used_mib=None)
            for line in sample.get("gpu",{}).get("stdout","").splitlines():
                ff=[x.strip() for x in line.split(",")]
                if len(ff)<5:continue
                try:row.update(gpu_name=ff[0],device_total_mib=float(ff[1]),device_used_mib=float(ff[2]),device_free_mib=float(ff[3]),device_utilization_percent=float(ff[4]))
                except ValueError:pass
            visible=set()
            for line in sample.get("processes",{}).get("stdout","").splitlines():
                ff=[x.strip() for x in line.split(",")]
                if len(ff)<3:continue
                try:pid=int(ff[0])
                except ValueError:continue
                if pid in training_pids:visible.add(pid)
                if pid==sample["pid"]:
                    try:row["process_used_mib"]=float(ff[-1])
                    except ValueError:pass
            row["visible_registered_training_processes"]=len(visible)
            observations.append(row)
    timing=dict(arm=arm,formal_updates=info["n"],
        formal_update_seconds=sum(r["update_seconds"] for r in info["logs"]),
        input_seconds=sum(r["input_seconds"] for r in info["logs"]),
        call_wall_span_seconds=info["calls"]["call_wall_span_seconds"],
        all_attempt_elapsed_seconds=sum(a["final_status"]["elapsed_seconds"] for a in attempts),
        measured_process_peak_mib=max(process_peaks) if process_peaks else None,
        process_peak_status="OBSERVED" if process_peaks else "NA_NVIDIA_SMI_NOT_EXPOSED",
        whole_device_peak_mib=max(device_peaks) if device_peaks else None,
        max_update_allocated_bytes=max(r["peak_allocated_bytes"] for r in info["logs"]),
        max_update_reserved_bytes=max(r["peak_reserved_bytes"] for r in info["logs"]),
        resource_observations=len(observations),attempts=len(attempts),
        attribution="Whole-device memory includes other processes. Allocator is not process VRAM. Observed concurrency affects wall time.")
    return observations,timing


def csv_bytes(rows):
    buf=io.StringIO(newline="");keys=list(dict.fromkeys(k for r in rows for k in r))
    writer=csv.DictWriter(buf,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def save(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==raw,"Existing summary artifact differs: "+str(path))
    else:
        with path.open("xb") as f:f.write(raw)


def moving(logs,width,value):
    return [statistics.fmean(value(r) for r in logs[max(0,i-width+1):i+1]) for i in range(len(logs))]


def plots(arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={ARMS[0]:"#386cb0",ARMS[1]:"#e07b28"}
    for arm,info in arms.items():
        logs=info["logs"];xs=[r["stage_step"] for r in logs];color=colors[arm]
        axes[0,0].plot(xs,[r["objective"] for r in logs],color=color,alpha=.13,lw=.45)
        axes[0,0].plot(xs,moving(logs,50,lambda r:r["objective"]),color=color,label=arm+" mean50")
        axes[0,0].plot(xs,moving(logs,100,lambda r:r["objective"]),color=color,ls="--",label=arm+" mean100")
        for ax,g in zip((axes[0,1],axes[1,0],axes[1,1]),GROUPS):
            values=[]
            for i in range(len(logs)):
                rr=logs[max(0,i-99):i+1]
                count=sum(r["group_allocation"]["actual_group_scalar_counts"][g] for r in rr)
                values.append(sum(r["group_velocity_sse"][g] for r in rr)/count if count else float("nan"))
            ax.plot(xs,values,color=color,label=arm)
    axes[0,0].set_title("Registered objective: original steps + trailing means")
    for ax,g in zip((axes[0,1],axes[1,0],axes[1,1]),GROUPS):ax.set_title(g+": trailing100 sum(SSE) / sum(scalars)")
    for ax in axes.flat:ax.set_xlabel("New optimizer updates");ax.set_ylabel("Training loss");ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle("Fixed-C multi-LOD training curves — different targets; loss is not generated-mesh success",fontsize=11)
    for ext in ("png","pdf"):
        path=DEST/("training_loss."+ext)
        if not path.exists():fig.savefig(path,dpi=180,metadata={"Creator":"CPU training audit"})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    reg=read(OUT/"training_registration.json")
    require(reg["updates"]==1500 and reg["arms"]==list(ARMS),"Registration budget differs")
    if not args.check:
        states={a:read(OUT/"training"/a/"status.json") if (OUT/"training"/a/"status.json").exists() else {"status":"NOT_STARTED"} for a in ARMS}
        if any(s["status"]!="COMPLETE" for s in states.values()):
            print(json.dumps(dict(status="PENDING_BOTH_COMPLETE",arms={a:s["status"] for a,s in states.items()},final_artifacts_written=False)));return
    arms={a:arm_audit(a,reg,not args.check) for a in ARMS};pair=paired_audit(arms)
    if args.check:
        print(json.dumps(dict(status="PASS_COMMITTED_PREFIX_ONLY",steps={a:r["n"] for a,r in arms.items()},pairing=pair,model_calls=0,final_artifacts_written=False)));return
    done=DEST/"complete.json"
    if done.exists():
        prior=read(done)
        for path,digest in prior["input_hashes"].items():require(sha(path)==digest,"Summary input changed")
        for path,digest in prior["files"].items():require(sha(path)==digest,"Summary artifact changed")
        print(json.dumps(dict(status="COMPLETE_REUSED",files=len(prior["files"]),model_calls=0)));return
    parent,groups,nd=exposure_tables(arms)
    rows=[];windows={50:[],100:[]};resources=[];timings=[]
    pids={pid for info in arms.values() for pid in info["calls"]["pids"]}
    for arm,info in arms.items():
        for log in info["logs"]:
            row={k:log[k] for k in ("arm","stage_step","global_step","objective","FM","all_free_velocity_sse","all_free_scalar_count","input_seconds","update_seconds","generator_gradient_norm","peak_allocated_bytes","peak_reserved_bytes")}
            for g in GROUPS:
                count=log["group_allocation"]["actual_group_scalar_counts"][g];sse=log["group_velocity_sse"][g]
                row[g+"_sse"],row[g+"_count"],row[g+"_mse"],row[g+"_a_g"]=sse,count,sse/count if count else None,log["group_allocation"]["reference_group_mse_weights"][g]
            rows.append(row)
        for width in windows:windows[width].extend(window_rows(arm,info["logs"],width))
        rr,tt=resource_tables(arm,info,pids);resources.extend(rr);timings.append(tt)
    DEST.mkdir(parents=True,exist_ok=True)
    tables=dict(loss_per_step=rows,loss_windows50=windows[50],loss_windows100=windows[100],
                paired_parent_exposure=parent,group_exposure=groups,N_distribution=nd,resource_observations=resources,timing_and_peaks=timings)
    for name,table in tables.items():save(DEST/(name+".csv"),csv_bytes(table))
    plots(arms)
    a,b=(arms[x]["calls"] for x in ARMS)
    overlap=max(0.,min(a["last_call_unix"],b["last_call_unix"])-max(a["first_call_unix"],b["first_call_unix"]))
    inputs=[OUT/"training_registration.json"]
    for arm in ARMS:
        folder=OUT/"training"/arm
        inputs.extend(folder/name for name in ("status.json","training_log.jsonl","exposure.jsonl","calls.jsonl"))
        inputs.extend(resolve(reg[k][arm]["path"]) for k in ("schedules","coefficient_schedules"))
        inputs.extend((folder/"attempts").glob("*/resources.json"));inputs.extend((folder/"attempts").glob("*/final_status.json"))
    summary=dict(status="COMPLETE",scope="TRAINING_ACCOUNTING_ONLY_NOT_GENERATION_RESULT",pairing=pair,
        actual_calls={arm:info["calls"] for arm,info in arms.items()},
        LOW_instances={arm:sum(r["variant"]=="LOW" for r in info["exposures"]) for arm,info in arms.items()},
        actual_training_timing=timings,overlapping_call_wall_interval_seconds=overlap,
        loss_interpretation=WARNING,weight_artifacts="Existence and recorded receipts checked; full tensor/weight bytes not loaded or rehashed by this CPU log summary.",
        model_calls=0,scoring_calls=0,OT_calls=0,new_training_updates=0,
        input_hashes={str(p):sha(p) for p in inputs},script_sha256=sha(__file__),
        files={str(p):sha(p) for p in sorted(DEST.iterdir()) if p.is_file()})
    save(done,(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode())
    print(json.dumps(dict(status="COMPLETE",actual_LOW=summary["LOW_instances"],paired_instances=pair["paired_completed_instances"],files=len(summary["files"]),model_calls=0)))


if __name__=="__main__":main()
