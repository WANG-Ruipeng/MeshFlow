"""Independent CPU-only audit of this fixed 32-new/64-cache midpoint experiment.

Reads actual arrays/exports/journals; imports no torch, scorer, sampler or model.
All former experiment files are read-only. Only this run's audit outputs change.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv, hashlib, json, math, os
from pathlib import Path
import time
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="2"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"native_t1/runs/chair_capped500_half_v1"
LABELS=("START","HIST_CAPPED500","HALF_CAPPED500")
NEW=LABELS[-1]
BASE_CACHE_SHA="180244bf8e5b127d6315b779ed272bfc6101687e1867d2641d18b20f7f0c9499"
HALF_STATE_SHA="f47e92e70a0ab4a4a84bfc04d1fe2985d4a5e66810b2bb9b5a477b9dd0b1bd41"

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temp,p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):h.update(b)
    return h.hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def old_array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def sampler_array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def ply_read(p):
    with Path(p).open("rb") as f:
        lines=[]
        while True:
            line=f.readline()
            if not line:raise ValueError("Incomplete PLY header")
            lines.append(line.decode("ascii").strip())
            if line==b"end_header\n":break
        if lines[:2]!=["ply","format binary_little_endian 1.0"]:raise ValueError("PLY format changed")
        nv=int(next(x.split()[-1] for x in lines if x.startswith("element vertex ")))
        nf=int(next(x.split()[-1] for x in lines if x.startswith("element face ")))
        vertices=np.frombuffer(f.read(12*nv),dtype="<f4").reshape(nv,3).copy()
        dtype=np.dtype([("count","u1"),("indices","<i4",(3,)),("known","u1"),("known_slot","<i4"),("source_face","<i4")])
        faces=np.frombuffer(f.read(nf*dtype.itemsize),dtype=dtype).copy()
        if f.read(1):raise ValueError("Unexpected trailing PLY bytes")
    if nv!=3*nf or not np.all(faces["count"]==3) or not np.array_equal(faces["indices"],np.arange(nv).reshape(nf,3)):
        raise ValueError("PLY welding, face order or corner indices changed")
    return vertices[faces["indices"]],faces

class Audit:
    def __init__(self,out):
        self.out=Path(out).resolve();self.checks=0;self.errors=[];self.missing=[];self.hashes={}
    def check(self,ok,message,**where):
        self.checks+=1
        if not bool(ok):self.errors.append(dict(message=message,**where))
        return bool(ok)
    def need(self,p):
        p=Path(p)
        if not p.exists():self.missing.append(str(p));return False
        return True
    def hash(self,p,expected=None):
        p=Path(p);key=str(p)
        if not self.need(p):return None
        if key not in self.hashes:self.hashes[key]=digest(p)
        if expected is not None:self.check(self.hashes[key]==expected,"SHA256 mismatch",path=key,expected=expected,actual=self.hashes[key])
        return self.hashes[key]
    def lines(self,p):
        if not self.need(p):return []
        return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    def guarded(self,name,fn,*args):
        try:return fn(*args)
        except Exception as exc:
            self.check(False,"Auditor section could not complete",section=name,error=repr(exc))
            return {"status":"BLOCKED","error":repr(exc)}

def audit_calls(a,requests):
    path=a.out/"ledger/calls.jsonl";rows=a.lines(path)
    begin={};end={};counts=Counter();returned=Counter();failed=Counter();samples={};active=None
    for row in rows:
        event,cid,kind=row.get("event"),row.get("call_id"),row.get("kind")
        ctx=row.get("context",{})
        if not a.check(event in ("CALL_BEGIN","CALL_RETURN","CALL_FAILED") and kind in ("sampling","sampling_forward") and isinstance(cid,str),
                       "Unexpected call event/kind; training and all extra calls forbidden",row=row):continue
        rid=ctx.get("request_id")
        a.check(rid in requests and ctx.get("label")==NEW and ctx.get("split")==requests.get(rid,{}).get("panel"),
                "Call uses unregistered label/request/panel",call_id=cid)
        if event=="CALL_BEGIN":
            if not a.check(cid not in begin,"Duplicate CALL_BEGIN",call_id=cid):continue
            begin[cid]=row;counts[kind]+=1
            if kind=="sampling":
                a.check(active is None,"Overlapping Euler attempts",call_id=cid)
                active=cid;samples[cid]={"request_id":rid,"split":ctx.get("split"),"forward_ids":[],"steps":[]}
            else:
                if a.check(active in samples,"Forward outside an Euler attempt",call_id=cid):
                    a.check(samples[active]["request_id"]==rid,"Forward request differs from parent",call_id=cid)
                    samples[active]["forward_ids"].append(cid);samples[active]["steps"].append(ctx.get("euler_step"))
        else:
            if not a.check(cid in begin and cid not in end,"Unmatched/duplicate call terminal",call_id=cid):continue
            b=begin[cid];end[cid]=row
            a.check(b["kind"]==kind and b["context"]==ctx and b["pid"]==row["pid"],"Call terminal identity changed",call_id=cid)
            a.check(row["time_unix"]>=b["time_unix"] and math.isfinite(row.get("wall_seconds",-1)) and row.get("wall_seconds",-1)>=0,
                    "Call duration malformed",call_id=cid)
            (returned if event=="CALL_RETURN" else failed)[kind]+=1
            if kind=="sampling":
                a.check(active==cid,"Euler outer terminal does not close active call",call_id=cid)
                active=None
                if cid in samples:samples[cid]["status"]=event
    pending=sorted(set(begin)-set(end))
    a.check(not pending and active is None,"Unclosed calls",pending=pending)
    a.check(dict(counts)=={"sampling":32,"sampling_forward":1600},"Actual call counts differ",counts=dict(counts))
    a.check(returned==counts and not failed,"Not every actual call succeeded",returned=dict(returned),failed=dict(failed))
    a.check(Counter(x["request_id"] for x in samples.values())==Counter({rid:1 for rid in requests}),
            "Each fixed request must have exactly one Euler attempt")
    for cid,s in samples.items():
        a.check(s["steps"]==list(range(50)),"Euler steps not exactly0..49",call_id=cid)
        a.check(s.get("status")=="CALL_RETURN" and all(end.get(i,{}).get("event")=="CALL_RETURN" for i in s["forward_ids"]),
                "Incomplete sampling attempt",call_id=cid)
    occupancy=a.lines(a.out/"ledger/gpu_occupancy.jsonl")
    a.check([r.get("event") for r in occupancy]==["GPU_ACQUIRE","GPU_RELEASE"],"Expected one closed exclusive GPU interval")
    if len(occupancy)==2:
        ac,release=occupancy
        a.check(ac["pid"]==release["pid"] and ac["stage"]==release["stage"] and release.get("status")=="PASS",
                "GPU owner/status changed")
        for cid,b in begin.items():
            a.check(b["pid"]==ac["pid"] and ac["time_unix"]<=b["time_unix"]<=end.get(cid,{}).get("time_unix",float("inf"))<=release["time_unix"],
                    "Call outside sole GPU lock interval",call_id=cid)
    accounting=read(a.out/"accounting.json") if a.need(a.out/"accounting.json") else {}
    a.check(accounting.get("attempts")==dict(counts) and accounting.get("returns")==dict(returned) and not accounting.get("failed") and accounting.get("pending")==[],
            "Accounting summary differs from actual journal")
    for k in ("training_updates","backward_calls","optimizer_updates","baseline_new_generation","additional_model_calls"):
        a.check(accounting.get(k)==0,"Zero-work contract summary differs",field=k)
    a.check(accounting.get("per_request_forward_counts")=={rid:50 for rid in requests},"Per-request NFE summary differs")
    extra=list((a.out/"ledger").glob("calls_*.jsonl"))
    a.check(not extra,"Unexpected additional call ledgers",paths=list(map(str,extra)))
    training_files=list((a.out/"training").rglob("*")) if (a.out/"training").exists() else []
    a.check(not [p for p in training_files if p.is_file()],"Unexpected training artifacts")
    return {"attempts":dict(counts),"returns":dict(returned),"failures":dict(failed),"pending":pending,
      "sampling_by_request":{s["request_id"]:dict(call_id=cid,steps=s["steps"],status=s.get("status")) for cid,s in samples.items()},
      "GPU_intervals":occupancy,"training_updates":0,"backward_calls":0,
      "journal_sha256":a.hash(path),"GPU_journal_sha256":a.hash(a.out/"ledger/gpu_occupancy.jsonl")}

def audit_sources(a,protocol):
    expected=dict(protocol["protected_files"])
    for panel,h in protocol["manifest_sha256"].items():expected[str(a.out/"manifests"/(panel+".json"))]=h
    expected[protocol["start_checkpoint"]]=protocol["start_checkpoint_sha256"]
    expected[protocol["capped_checkpoint"]]=protocol["capped_checkpoint_sha256"]
    expected[str(a.out/"registration/baseline_cache_manifest.json")]=BASE_CACHE_SHA
    for p,h in expected.items():a.hash(p,h)
    side_path=a.out/"checkpoints/half_capped500_alpha050.json"
    side=read(side_path)
    a.hash(side["path"],side["checkpoint_sha256"]);m=side["metadata"]
    a.check(side["schema"]=="chair_capped500_half_inference_v1" and side["alpha"]==.5 and
            m["alpha"]==.5 and m["model_state_sha256"]==HALF_STATE_SHA,"Midpoint artifact identity differs")
    a.check(m["start_checkpoint_sha256"]==protocol["start_checkpoint_sha256"] and m["capped_checkpoint_sha256"]==protocol["capped_checkpoint_sha256"],
            "Midpoint endpoints differ")
    a.check(m["inference_only"] and m["training_updates"]==0 and not m["optimizer_present"] and m["training_step"] is None,
            "Midpoint must have no training-recovery interpretation")
    a.check(m["parameter_tensors"]==194 and m["parameter_elements"]==130401155 and m["buffer_tensors"]==0,
            "Midpoint model topology differs")
    review=read(a.out/"audit/interpolation_review_fp64.json")
    a.check(review["status"]=="PASS" and review["independent_fp64_midpoint_model_state_sha256"]==m["model_state_sha256"],
            "Independent midpoint receipt differs")
    # No large checkpoint is deserialized here: file identity binds the already
    # independently audited all-parameter tensor state.
    return {"protected_files":len(expected),"half_checkpoint_sha256":side["checkpoint_sha256"],
      "baseline_checkpoint_sha256":{"START":protocol["start_checkpoint_sha256"],"HIST_CAPPED500":protocol["capped_checkpoint_sha256"]},
      "checkpoint_payloads_loaded":0,"before_after":[{"path":p,"expected":h,"actual":a.hashes.get(str(Path(p)))} for p,h in expected.items()]}

def audit_output(a,req,label,cache,checkpoints,calls):
    rid=req["request_id"];n,k=req["N"],req["K"];panel=req["panel"]
    context=dict(label=label,request_id=rid)
    start_errors=len(a.errors)
    a.hash(req["bank_npz"],req["bank_sha256"]);bank=arrays(req["bank_npz"])
    a.hash(req["condition_npz"],req["condition_sha256"]);condition=arrays(req["condition_npz"])
    a.check(set(bank)=={"z","C","known_mask","valid_mask","N","K"},"Inference bank contains extra/target arrays",**context)
    C=bank["C"];known=np.arange(n)<k;valid=np.ones(n,dtype=bool)
    a.check(bank["z"].shape==(1,n,9) and C.shape==(k,3,3) and bank["z"].dtype==C.dtype==np.float32,
            "Bank shape/dtype differs",**context)
    a.check(bits(bank["known_mask"],known) and bits(bank["valid_mask"],valid) and int(bank["N"])==n and int(bank["K"])==k,
            "Bank masks/N/K differ",**context)
    a.check(bits(condition["C"],C) and bits(condition["GT_full"][condition["source_face_ids"]],C),
            "Known source face order/winding does not reconstruct C",**context)
    a.check(len(set(condition["source_face_ids"].tolist()))==k,"Known source IDs not unique",**context)
    if cache:
        a.check(cache["status"]=="PASS" and cache["label"]==label and cache["panel"]==panel and cache["request_id"]==rid,
                "Baseline frozen identity differs",**context)
        for path,h in cache["source_files"].items():a.hash(path,h)
        result_paths=[Path(p) for p in cache["source_files"] if Path(p).name=="result.json" and Path(p).parent.name==rid]
        raw_paths=[Path(p) for p in cache["source_files"] if Path(p).name=="raw.npz" and Path(p).parent.name==rid]
        a.check(len(result_paths)==len(raw_paths)==1,"Ambiguous baseline raw/result source",**context)
        rp,rawp=result_paths[0],raw_paths[0]
        expected_checkpoint=checkpoints["baseline_checkpoint_sha256"][label]
        a.check(cache["baseline_checkpoint_sha256"]==expected_checkpoint and cache["metrics_recomputed"]==0,"Baseline source checkpoint/cache scope differs",**context)
    else:
        rp=a.out/"runtime"/label/rid/"result.json";rawp=rp.with_name("raw.npz")
        expected_checkpoint=checkpoints["half_checkpoint_sha256"]
    if not a.need(rp) or not a.need(rawp):return dict(**context,panel=panel,status="NOT_RUN",origin="CACHE" if cache else "NEW")
    result=read(rp);a.hash(rp);a.hash(rawp,result["raw_sha256"]);raw=arrays(rawp)
    a.check(result["status"]=="PASS" and result["request_id"]==rid and result["checkpoint_sha256"]==expected_checkpoint,
            "Saved result/checkpoint identity differs",**context)
    path=raw["path"];output=raw["output"];velocity=raw["velocities"]
    a.check(path.shape==(51,n,9) and output.shape==(n,9) and velocity.shape==(50,n,9),"Saved trajectory shapes differ",**context)
    a.check(all(x.dtype==np.float32 and np.isfinite(x).all() for x in (path,output,velocity,raw["C"],raw["z"])),
            "Raw dtype/finite contract differs",**context)
    a.check(bits(raw["C"],C) and bits(raw["z"],bank["z"]) and bits(raw["initial_gaussian"],bank["z"]),
            "Raw C/Gaussian bytes changed",**context)
    a.check(bits(raw["known_mask"],known) and bits(raw["valid_mask"],valid),"Raw masks changed",**context)
    a.check(bits(path[-1],output) and bits(path[0,k:],bank["z"][0,k:]),"First/free-noise or final state differs",**context)
    C_ok=True
    for step in range(51):
        passed=bits(path[step,known],C.reshape(k,9));C_ok=C_ok and passed
        a.check(passed,"Known C not bitwise preserved",state=step,**context)
    reconstructed=path[:-1,k:]+velocity[:,k:]*np.float32(1/50)
    euler_exact=bits(reconstructed,path[1:,k:])
    a.check(euler_exact,"Saved free states do not satisfy original FP32 Euler update",**context)
    meta=result["metadata"]
    a.check(meta["nfe"]==50 and meta["accepted_states"]==51 and meta["solver"]=="original_Euler50" and
            meta["integration_dtype"]=="float32" and meta["forward_dtype"]=="bfloat16" and
            meta["observed_raw_output_dtypes"]==["torch.bfloat16"] and meta["actual_backbone_calls"]==meta["actual_Geo_recomputations"]==50,
            "Actual runtime/forward metadata differs",**context)
    a.check(meta["known_all51_bitwise"] and meta["known_state_checks"]==51 and not meta["GT_model_inputs"] and
            not meta["initial_gaussian_scaled_by_alpha"] and meta["geometry_repairs"]==0,
            "Sampler invariants metadata differs",**context)
    a.check(meta["noise_sha256"]==sampler_array_hash(bank["z"]) and meta["condition_sha256"]==sampler_array_hash(C.reshape(k,9)),
            "Sampler input hashes differ",**context)
    for phase in ("runtime_before","runtime_after"):
        m=meta[phase]
        a.check(m["deterministic_algorithms"] and not m["deterministic_warn_only"] and m["cublas_workspace_config"]==":4096:8" and
                not m["tf32_matmul"] and not m["tf32_cudnn"] and m["sdpa_math_enabled"] and
                not m["sdpa_flash_enabled"] and not m["sdpa_mem_efficient_enabled"] and not m["sdpa_cudnn_enabled"] and
                m["cuda_autocast_enabled"] and m["cuda_autocast_dtype"]=="torch.bfloat16" and m["all_modules_eval"],
                "Deterministic BF16/math runtime differs",phase=phase,**context)
    ex=result["export"]
    a.hash(ex["npz"],ex["npz_sha256"]);a.hash(ex["ply"],ex["ply_sha256"])
    export=arrays(ex["npz"]);ply,faces=ply_read(ex["ply"]);full=output.reshape(n,3,3)
    ids=np.arange(k,dtype=np.int64);slots=np.full(n,-1,dtype=np.int64);slots[known]=ids
    source=np.full(n,-1,dtype=np.int64);source[known]=condition["source_face_ids"]
    a.check(bits(export["output"],full) and bits(ply,full) and bits(export["C"],C) and bits(ply[known],C),
            "NPZ/PLY coordinate bytes or winding changed",**context)
    a.check(bits(export["known_mask"],known) and bits(export["valid_mask"],valid) and
            np.array_equal(export["known_to_output_face"],ids) and np.array_equal(export["output_to_known_slot"],slots) and
            np.array_equal(export["known_source_face_id_by_output"],source),"NPZ known/source association differs",**context)
    a.check(np.array_equal(faces["known"],known) and np.array_equal(faces["known_slot"],slots) and
            np.array_equal(faces["source_face"],source),"PLY known/source association differs",**context)
    for key in ("C_world","known_source_vertex_ids","original_known_source_face_ids"):
        if key in condition:a.check(key in export and bits(export[key],condition[key]),"Export provenance array changed",field=key,**context)
    a.check(ex["N"]==n and ex["K"]==k and ex["vertex_count"]==3*n and not ex["geometry_modified"] and
            not ex["world_inverse_applied"] and not ex["generated_free_GT_correspondence_inferred"],
            "Export size/frame/repair differs",**context)
    a.check(ex["known_to_output_face"]==ids.tolist() and ex["known_source_face_ids"]==source[known].tolist() and
            ex["C_sha256"]==old_array_hash(C) and ex["output_sha256"]==old_array_hash(full),
            "Export hashes/receipt mapping differ",**context)
    if not cache:
        attempt=read(rp.with_name("attempt.json"))
        a.check(attempt["context"]=={"label":label,"request_id":rid,"split":panel} and
                attempt["checkpoint_sha256"]==expected_checkpoint and attempt["bank_sha256"]==req["bank_sha256"],
                "Frozen generation attempt differs",**context)
        a.check(rid in calls.get("sampling_by_request",{}) and calls["sampling_by_request"][rid]["status"]=="CALL_RETURN",
                "New raw lacks one completed actual generation",**context)
        a.check(result["label"]==NEW and result["split"]==panel,"New result label/panel differs",**context)
    return dict(**context,panel=panel,uid=req["uid"],candidate=req["candidate"],N=n,K=k,
       status="PASS" if len(a.errors)==start_errors else "FAIL",origin="VERIFIED_BASELINE_CACHE" if cache else "NEW_EULER50",
       raw_path=str(rawp),raw_sha256=result["raw_sha256"],result_path=str(rp),result_sha256=a.hash(rp),
       checkpoint_sha256=expected_checkpoint,bank_sha256=req["bank_sha256"],condition_sha256=req["condition_sha256"],
       path51_C_bitwise=C_ok,original_Gaussian_bitwise=True,Euler50_free_bitwise=euler_exact,
       Euler_max_abs_residual=float(np.max(np.abs(reconstructed-path[1:,k:]))),
       output_NPZ_PLY_all_coordinate_bytes=True,known_mapping_order_winding_preserved=True,
       npz_sha256=ex["npz_sha256"],ply_sha256=ex["ply_sha256"])

def run(out=OUT):
    tick=time.perf_counter();a=Audit(out)
    if a.out!=OUT.resolve():raise ValueError("Audit restricted to this run")
    protocol=read(a.out/"protocol.json")
    a.check(protocol["labels"]==list(LABELS) and protocol["alpha"]==.5 and protocol["status"]=="FROZEN","Protocol scope differs")
    a.check(protocol["budget"]==dict(training_updates=0,backward_calls=0,model_forward_calls=1600,new_Euler50_attempts=32,
        solver_steps_per_output=50,additional_seeds=0,additional_alphas=0,baseline_generated=0),"Frozen budget differs")
    requests={};panels={}
    for panel in ("OLD","FRESH"):
        p=a.out/"manifests"/(panel+".json");a.hash(p,protocol["manifest_sha256"][panel]);data=read(p);rows=data["rows"];panels[panel]=rows
        a.check(len(rows)==16 and len({r["uid"] for r in rows})==8 and data["labels"]==list(LABELS),"Panel cohort differs",panel=panel)
        a.check(Counter((r["uid"],r["candidate"]) for r in rows)==Counter({(u,c):1 for u in {r["uid"] for r in rows} for c in ((0,1) if panel=="OLD" else (2,3))}),
                "Panel parent/noise balance differs",panel=panel)
        for r in rows:
            a.check(r["request_id"]==r["id"] and r["panel"]==r["split"]==panel and r["status"]=="READY" and r["alpha"]==1.,
                    "Manifest identity differs",request_id=r["request_id"])
            a.check(r["request_id"] not in requests,"Duplicate request_id");requests[r["request_id"]]=r
    checkpoints=a.guarded("sources",audit_sources,a,protocol)
    calls=a.guarded("calls",audit_calls,a,requests)
    baseline=read(a.out/"registration/baseline_cache_manifest.json")
    a.hash(a.out/"registration/baseline_cache_manifest.json",BASE_CACHE_SHA)
    cache={(r["label"],r["request_id"]):r for r in baseline["records"]}
    a.check(len(cache)==len(baseline["records"])==baseline["cells"]==64 and baseline["status"]=="PASS" and
            baseline["model_calls"]==baseline["metrics_recomputed"]==0,"Frozen baseline cache manifest differs")
    a.check(set(cache)=={(label,rid) for label in LABELS[:2] for rid in requests},"Baseline fixed cell coverage differs")
    rows=[]
    for panel in ("OLD","FRESH"):
        for req in panels[panel]:
            for label in LABELS:
                result=a.guarded(label+"/"+req["request_id"],audit_output,a,req,label,cache.get((label,req["request_id"])),checkpoints,calls)
                if "request_id" not in result:result.update(request_id=req["request_id"],label=label,panel=panel,uid=req["uid"],candidate=req["candidate"])
                rows.append(result)
    a.check(len(rows)==96 and sum(r.get("origin")=="NEW_EULER50" for r in rows)==32 and
            sum(r.get("origin")=="VERIFIED_BASELINE_CACHE" for r in rows)==64,"Actual output count differs")
    a.check(all(r["status"]=="PASS" for r in rows),"One or more output audits not PASS")
    allowed={str(a.out/"runtime"/NEW/rid/"result.json") for rid in requests}
    actual={str(p) for p in (a.out/"runtime").rglob("result.json")}
    a.check(actual==allowed,"Unexpected or missing runtime output cells",extra=sorted(actual-allowed),missing=sorted(allowed-actual))
    a.check(not list((a.out/"runtime").rglob("failure.json")) and not list((a.out/"runtime").rglob("engineering_retry_used.json")),
            "Failed/retried sampling found")
    status=read(a.out/"status/sampling.json") if a.need(a.out/"status/sampling.json") else {}
    a.check(status.get("status")=="PASS","Sampling worker final status is not PASS")
    csvpath=a.out/"audit/96output.csv";csvpath.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with csvpath.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result={"status":"PASS" if not a.errors and not a.missing else "INCOMPLETE" if a.missing else "FAIL",
       "checks":a.checks,"errors":a.errors,"missing":a.missing,"outputs":len(rows),"new_outputs":32,"baseline_outputs":64,
       "actual_new_forward_attempts":calls.get("attempts",{}).get("sampling_forward"),"actual_training_updates":0,
       "calls":calls,"source_protection":checkpoints,"all_output_status_counts":dict(Counter(r["status"] for r in rows)),
       "files_actually_hashed":len(a.hashes),"hashes":a.hashes,"output_csv":str(csvpath),"output_csv_sha256":digest(csvpath),
       "audit_source_sha256":digest(__file__),"seconds":time.perf_counter()-tick,
       "audit_scientific_calls":dict(model_forward=0,backward=0,training=0,scoring=0,generation=0),
       "limits":["Output byte/winding checks do not establish watertightness, attachment quality or mechanical usability.",
                 "No checkpoint payload was loaded by this runtime audit; source and midpoint file hashes bind the separate CPU tensor audit.",
                 "CPU Euler reconstruction uses saved actual velocities; it does not rerun the network."]}
    write(a.out/"audit/runtime_final.json",result)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",default=str(OUT));parser.add_argument("--final",action="store_true")
    args=parser.parse_args();r=run(args.out)
    print(json.dumps({k:r[k] for k in ("status","checks","outputs","all_output_status_counts","actual_new_forward_attempts","files_actually_hashed","seconds")}))
    raise SystemExit(0 if r["status"]=="PASS" else 1)
