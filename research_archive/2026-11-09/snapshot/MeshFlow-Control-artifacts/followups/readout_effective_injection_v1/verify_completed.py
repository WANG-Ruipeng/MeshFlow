"""Pure CPU completion verifier; never imports torch, worker, or model code.
Missing, duplicate, altered, or incomplete evidence is an error. This command
cannot resume computation or issue a forward; a complete run is only verified.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
from pathlib import Path
import collections, hashlib, json, time
import numpy as np

ROOT = Path(__file__).resolve().parent
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()
def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def verify():
    begin=time.time()
    checks=[]; errors=[]
    def check(name,value,detail=None):
        item=dict(check=name,passed=bool(value))
        if detail is not None: item["detail"]=detail
        checks.append(item)
        if not value: errors.append(item)
    result=dict(schema="readout_effective_injection_cpu_completion_verification_v1",
                verifier_model_calls=0,verifier_weight_loads=0,
                can_resume_model_computation=False,
                missing_or_duplicate_policy="FAIL; never regenerate or retry")
    try:
        plan=read(ROOT/"plan.json");status=read(ROOT/"status.json")
        check("status_complete",status["status"]=="COMPLETE")
        check("frozen_dimensions",plan["labels"]==["B1000","C1000"] and plan["steps"]==[0,25,45] and plan["modes"]==["on","off"] and plan["layers"]==[3,6,9,12] and len(plan["requests"])==4)
        check("registered_budget",plan["forwards"]==48 and all(plan[k]==0 for k in ("backwards","optimizer_updates","new_gaussians","new_rollouts","quality_rescoring")))
        check("runner_matches_plan",sha(ROOT/"run_check.py")==plan["runner_sha256"])
        target={}
        for label in plan["labels"]:
            for request in plan["requests"]:
                for step in plan["steps"]:
                    for mode in plan["modes"]:
                        ident=f"{label}__{request['request_id']}__s{step:02d}__{mode}"
                        target[ident]=(label,request,step,mode)
        check("exact_48_unique_target_cells",len(target)==48)
        ledger=rows(ROOT/"budget/calls.jsonl")
        events=collections.Counter(r["event"] for r in ledger)
        allowed={"RESOURCE","RESERVE","RETURN","COMMIT"}
        check("no_errors_or_unexpected_events",set(events)<=allowed,dict(events))
        groups={}
        for event in ("RESERVE","RETURN","COMMIT"):
            rr=[(i,r) for i,r in enumerate(ledger) if r["event"]==event]
            cc=collections.Counter(r["cell_id"] for _,r in rr)
            check(event+"_exact_48_unique",len(rr)==48 and set(cc)==set(target) and all(n==1 for n in cc.values()),dict(count=len(rr),unique=len(cc)))
            groups[event]={r["cell_id"]:(i,r) for i,r in rr}
        check("reserve_ordinals_exact",sorted(r["ordinal"] for _,r in groups["RESERVE"].values())==list(range(1,49)))
        cell_files={p.stem:p for p in (ROOT/"cells").glob("*.json")}
        npz_files={p.stem:p for p in (ROOT/"cells").glob("*.npz")}
        check("receipts_exact_target_set",set(cell_files)==set(target))
        check("arrays_exact_target_set",set(npz_files)==set(target))
        check("no_partial_cell_artifacts",not list((ROOT/"cells").glob("*.partial")))
        streams={(r["label"],r["request_id"]):r for r in plan["streams"]}
        layer_count=0;on_replays=0;dtype_counts=collections.Counter()
        receipt_hashes={}
        for ident,(label,request,step,mode) in target.items():
            rv=groups["RESERVE"][ident];rt=groups["RETURN"][ident];cm=groups["COMMIT"][ident]
            check(ident+":event_order",rv[0]<rt[0]<cm[0])
            check(ident+":event_context",rv[1]["kind"]==rt[1]["kind"]=="diagnostic_forward" and (rv[1]["label"],rv[1]["step"],rv[1]["mode"])==(label,step,mode))
            c=read(cell_files[ident]);n=request["N"];k=request["K"]
            check(ident+":receipt_identity",(c["cell_id"],c["status"],c["label"],c["request_id"],c["N"],c["K"],c["step"],c["mode"])==(ident,"COMPLETE",label,request["request_id"],n,k,step,mode))
            check(ident+":checkpoint_identity",c["checkpoint_sha256"]==plan["checkpoints"][label]["sha256"])
            check(ident+":array_path",Path(c["array_path"]).resolve()==npz_files[ident].resolve())
            hh=sha(npz_files[ident])
            check(ident+":array_hash",hh==c["array_sha256"]==cm[1]["array_sha256"])
            check(ident+":sites",c["readout_sites_visited"]==[3,6,9,12])
            check(ident+":prediction_dtype",c["prediction_dtype"]=="torch.bfloat16 stored as float32 values")
            sr=streams[label,request["request_id"]]
            with np.load(npz_files[ident],allow_pickle=False) as z:
                expected={"velocity","state","next_state"}
                if mode=="on":
                    expected|={f"l{l}_{name}" for l in (3,6,9,12) for name in ("before","residual","cast","after")}
                check(ident+":array_keys",set(z.files)==expected)
                check(ident+":base_shapes_finite",all(z[name].shape==(1,n,9) and z[name].dtype==np.float32 and np.isfinite(z[name]).all() for name in ("velocity","state","next_state")))
                check(ident+":saved_prediction_values_are_bf16",bool(((z["velocity"].view(np.uint32)&65535)==0).all()))
                check(ident+":actual_input_state_hash",ah(z["state"])==sr["state_hashes"][str(step)]==c["input_state_sha256"])
                check(ident+":known_state_unchanged",z["state"][:,:k].tobytes()==z["next_state"][:,:k].tobytes() and c["known_coordinates_next_exact"] is True)
                if mode=="on":
                    exact=ah(z["next_state"])==sr["next_state_hashes"][str(step)]
                    check(ident+":actual_on_next_state_hash",exact and c["trajectory_replay_exact"] is True)
                    on_replays+=int(exact and c["trajectory_replay_exact"] is True)
                    lc=c["layer_checks"]
                    check(ident+":four_layer_receipts",len(lc)==4 and [r["layer"] for r in lc]==[3,6,9,12])
                    for rec in lc:
                        layer=rec["layer"];layer_count+=1
                        dt=(rec["hidden_dtype"],rec["residual_dtype"],rec["output_dtype"]);dtype_counts[str(dt)]+=1
                        check(ident+f":layer{layer}:dtype",dt==("torch.bfloat16","torch.float32","torch.bfloat16"))
                        check(ident+f":layer{layer}:checks",all(rec[q] is True for q in ("query_is_actual_free","known_padding_unchanged","actual_add_matches")) and rec["free_corners"]==3*(n-k))
                        for name in ("before","residual","cast","after"):
                            arr=z[f"l{layer}_{name}"]
                            check(ident+f":layer{layer}:{name}:stored_shape_dtype_finite",arr.shape==(3*(n-k),768) and arr.dtype==np.float32 and np.isfinite(arr).all())
                            if name!="residual":
                                check(ident+f":layer{layer}:{name}:bf16_values",bool(((arr.view(np.uint32)&65535)==0).all()))
                else:
                    check(ident+":off_no_probe_compute",c["layer_checks"]==[] and c["trajectory_replay_exact"] is None)
            receipt_hashes[ident]=dict(receipt_sha256=sha(cell_files[ident]),array_sha256=hh)
        check("24_actual_on_replays",on_replays==24)
        check("96_layer_records",layer_count==96)
        check("final_accounting_counts",status["actual_forwards"]==status["completed_cells"]==48 and status["on_forwards"]==status["off_forwards"]==24 and status["layer_records"]==96 and all(status[k]==0 for k in ("backwards","optimizer_updates","new_gaussians","new_rollouts","geometry_rescoring","scientific_source_files_modified")))
        integrity=[]
        for label in plan["labels"]:
            item=read(ROOT/(label+"_integrity.json"))
            check(label+":model_rng_integrity",item["label"]==label and item["unchanged"] is True and item["optimizer_created"] is False and item["model_state_sha256_before"]==item["model_state_sha256_after"] and item["RNG_sha256_before"]==item["RNG_sha256_after"])
            check(label+":integrity_matches_status",item in status["model_integrity"])
            integrity.append(item)
        protected_count=0
        for path,expected in plan["protected_files"].items():
            check("protected:"+path,sha(path)==expected)
            protected_count+=1
        occ=rows(ROOT/"budget/gpu_occupancy.jsonl")
        ss=[r for r in occ if r["event"]=="START"];ee=[r for r in occ if r["event"]=="END"]
        check("one_closed_gpu_owner",len(ss)==len(ee)==1 and ss[0]["id"]==ee[0]["id"] and occ.index(ss[0])<occ.index(ee[0]))
        owner_seconds=sum(r["seconds"] for r in ee)
        check("gpu_owner_within_600s",owner_seconds<=plan["gpu_owner_limit_seconds"],owner_seconds)
        monitor=rows(ROOT/"resource_monitor.jsonl")
        sampling_errors=[r for r in monitor if "error" in r]
        gpu_samples=[]
        for r in monitor:
            g=r.get("gpu",{})
            lines=g.get("stdout","").strip().splitlines()
            if len(lines)>=2:
                parts=[s.strip() for s in lines[1].split(",")]
                try:gpu_samples.append(dict(name=parts[0],total_mib=float(parts[1].split()[0]),used_mib=float(parts[2].split()[0]),free_mib=float(parts[3].split()[0])))
                except (ValueError,IndexError):pass
        resource=dict(gpu_owner_seconds=owner_seconds,worker_status_wall_seconds=status["wall_seconds"],pytorch_phase_peaks=integrity,
                      monitor_rows=len(monitor),monitor_error_rows=len(sampling_errors),
                      parsed_device_samples=len(gpu_samples),
                      observed_device_used_peak_mib=max((r["used_mib"] for r in gpu_samples),default=None),
                      observed_device_free_min_mib=min((r["free_mib"] for r in gpu_samples),default=None),
                      interpretation="PyTorch counters reset after each model load; include resident tensors and instrumented capture, not complete load-time or per-process VRAM. Device samples include other processes/desktop and are sampled, not a continuous process peak. Different B/C allocated peaks do not establish a retention cause or isolated per-model cost. Instrumented times are not inference benchmarks.")
        result.update(actual_forwards=48,actual_backwards=0,actual_updates=0,new_rollouts=0,on_replays=on_replays,
                      layer_records=layer_count,dtype_counts=dict(dtype_counts),ledger_counts=dict(events),
                      protected_file_count=protected_count,resources=resource,receipt_hashes=receipt_hashes,
                      source_hashes={str(p):sha(p) for p in (ROOT/"plan.json",ROOT/"run_check.py",Path(__file__),ROOT/"budget/calls.jsonl",ROOT/"budget/gpu_occupancy.jsonl")})
    except Exception as ex:
        errors.append(dict(check="verification_exception",passed=False,detail=repr(ex)))
    result.update(status="PASS" if not errors else "FAIL",checks_count=len(checks),errors=errors,checks=checks,
                  cpu_verification_wall_seconds=time.time()-begin)
    path=ROOT/"final_verification.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("checks","receipt_hashes","source_hashes")},ensure_ascii=True))
    return not errors
if __name__=="__main__":
    raise SystemExit(0 if verify() else 1)
