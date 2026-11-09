"""Independent CPU completion audit; never loads weights or imports model code.
Can repair only missing output-manifest rows from completely verified receipts.
No model invocation, sampler, new noise, scoring, or geometry alteration exists.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"
from pathlib import Path
import collections,csv,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");os.replace(tmp,p)
def read_raw_ply(p,n):
 with Path(p).open("rb") as f:
  lines=[]
  while True:
   line=f.readline()
   if not line:raise ValueError("Truncated PLY header")
   lines.append(line.decode("ascii").strip())
   if lines[-1]=="end_header":break
  if "format binary_little_endian 1.0" not in lines:raise ValueError("Wrong PLY format")
  nv=int(next(x for x in lines if x.startswith("element vertex ")).split()[-1])
  nf=int(next(x for x in lines if x.startswith("element face ")).split()[-1])
  if (nv,nf)!=(3*n,n):raise ValueError("PLY count mismatch")
  v=np.frombuffer(f.read(nv*12),dtype="<f4").copy().reshape(nv,3)
  ff=np.frombuffer(f.read(nf*13),dtype=np.dtype([("n","u1"),("v","<i4",(3,))],align=False))
  if len(ff)!=n or not np.all(ff["n"]==3) or not np.array_equal(ff["v"],np.arange(3*n).reshape(n,3)) or f.read(1):raise ValueError("PLY soup indices/winding/order mismatch")
  return v.reshape(n,9)
def verify():
 tick=time.perf_counter();errors=[];checks=[];output_rows=[];repair=None
 def require(name,ok,detail=None):
  checks.append(dict(check=name,passed=bool(ok),detail=detail))
  if not ok:raise ValueError(name+": "+str(detail))
 result=dict(schema="readout_precision_independent_cpu_verification_v1",model_calls_by_verifier=0,weights_loaded_by_verifier=0,geometry_modified=False)
 try:
  plan=read(ROOT/"plan.json");old=Path(plan["old_followup"]);ledger=rows(ROOT/"budget/calls.jsonl")
  expected={}
  for row in plan["source_rows"]:
   label,rid=row["label"],row["request_id"]
   for s in plan["steps"]:
    for mode in ("on","off"):
     expected[f"fixed__{label}__{rid}__s{s:02d}__{mode}"]=("fixed",row,"FP32_REFERENCE",mode,s)
   if rid in plan["fixture_requests"]:
    expected[f"preflight__replay__{label}__{rid}"]=("preflight",row,"LEGACY_BF16","on",25)
    for prec in ("LEGACY_BF16","FP32_REFERENCE"):
     expected[f"preflight__zero__{prec}__{label}__{rid}"]=("preflight",row,prec,"zero",25)
   for prec,mode in (("LEGACY_BF16","off"),("FP32_REFERENCE","on"),("FP32_REFERENCE","off")):
    name=f"{label}__{prec}__{mode.upper()}__{rid}"
    for s in range(50):expected[f"rollout__{name}__s{s:02d}"]=("sampling",row,prec,mode,s)
  require("registered1260",len(expected)==1260)
  grouped={}
  for event in ("RESERVE","RETURN","COMMIT"):
   rr=[(i,e) for i,e in enumerate(ledger) if e["event"]==event]
   require(event+"_unique_exact",len(rr)==1260 and len({e["id"] for _,e in rr})==1260 and {e["id"] for _,e in rr}==set(expected))
   grouped[event]={e["id"]:(i,e) for i,e in rr}
  require("no_ERROR_events",not any(e["event"]=="ERROR" for e in ledger))
  count=collections.Counter(e["kind"] for _,e in grouped["RESERVE"].values())
  require("forward_kind_counts",dict(count)==dict(preflight=12,fixed=48,sampling=1200),dict(count))
  commits={};fixed_off_before_gate=0;source_hashes={}
  for ident,(kind,row,prec,mode,s) in expected.items():
   rs,rt,cm=[grouped[event][ident] for event in ("RESERVE","RETURN","COMMIT")]
   require(ident+":ordering",rs[0]<rt[0]<cm[0])
   require(ident+":event_kinds",all(x[1]["kind"]==kind for x in (rs,rt,cm)))
   jp=Path(cm[1]["receipt_path"]);c=read(jp);ap=Path(c["array_path"])
   require(ident+":receipt_identity",c["status"]=="COMPLETE" and c["call_id"]==ident and c["label"]==row["label"] and c["request_id"]==row["request_id"] and c["step"]==s and c["t"]==s/50 and c["precision"]==prec and c["mode"]==mode and (c["N"],c["K"])==(row["N"],row["K"]))
   require(ident+":array_hash",sha(ap)==c["array_sha256"])
   require(ident+":prediction_dtype",c["velocity_dtype"]==("torch.bfloat16" if prec=="LEGACY_BF16" else "torch.float32"))
   commits[ident]=(jp,c)
   source_hashes[str(jp)]=sha(jp)
  preflight=read(ROOT/"preflight/status.json")
  require("preflight_pass",preflight["status"]=="PASS" and preflight["new_preflight_calls"]==12 and len(preflight["checks"])==12 and all(x["bitwise"] is True for x in preflight["checks"]))
  last_preflight_return=max(i for i,e in grouped["RETURN"].values() if e["kind"]=="preflight")
  before=[ident for ident,(i,e) in grouped["RESERVE"].items() if e["kind"]=="fixed" and i<last_preflight_return]
  require("four_planned_fp32_OFF_refs_before_gate",len(before)==4 and all(expected[i][3]=="off" and expected[i][4]==25 and expected[i][1]["request_id"] in plan["fixture_requests"] for i in before))
  for ident,(kind,row,prec,mode,s) in expected.items():
   if kind!="preflight":continue
   c=commits[ident][1];rid=row["request_id"];label=row["label"]
   if mode=="on":ref=read(old/"cells"/f"{label}__{rid}__s25__on.json")
   elif prec=="LEGACY_BF16":ref=read(old/"cells"/f"{label}__{rid}__s25__off.json")
   else:ref=commits[f"fixed__{label}__{rid}__s25__off"][1]
   with np.load(c["array_path"],allow_pickle=False) as z,np.load(ref["array_path"],allow_pickle=False) as zz:
    require(ident+":actual_reference_prediction",same(z["velocity"],zz["velocity"]) and same(z["state"],zz["state"]))
   if prec=="FP32_REFERENCE":
    require(ident+":FP32_dtype_observations",bool(c["dtype_records"]) and all(d=="torch.float32" for obs in c["dtype_records"] for field in ("input_float_dtypes","output_float_dtypes") for d in obs[field]))
  layer_count=0;full_layers=0
  for ident,(kind,row,prec,mode,s) in expected.items():
   if kind!="fixed":continue
   c=commits[ident][1]
   require(ident+":fixed_weight_identity",c["checkpoint_sha256"]==row["checkpoint_sha256"] and c["input_sha256"]==row["input_sha256"])
   with np.load(c["array_path"],allow_pickle=False) as z:
    with np.load(row["raw_path"],allow_pickle=False) as zz:
     require(ident+":old_ON_state_input",same(z["state"],zz["trajectory"][s:s+1]))
    n,k=row["N"],row["K"];calc=z["state"].copy();calc[:,k:]=calc[:,k:]+z["velocity"][:,k:]*(1/50)
    require(ident+":fixed_Euler",same(calc,z["next_state"]))
    if mode=="on":
     mm=c["layer_metrics"];require(ident+":four_layers",[x["layer"] for x in mm]==[3,6,9,12])
     for m in mm:
      layer_count+=1;require(ident+f":l{m['layer']}:FP32",all(m[key]=="torch.float32" for key in ("hidden_dtype","residual_dtype","cast_dtype","output_dtype")))
      saved=row["request_id"] in plan["fixture_requests"] and s==25
      require(ident+f":l{m['layer']}:storage",m["arrays_saved"]==saved)
      if saved:
       full_layers+=1
       require(ident+f":l{m['layer']}:captured_array_hashes",all(ah(z[f"l{m['layer']}_{key}"])==value for key,value in m["array_hashes"].items()))
    else:require(ident+":OFF_no_readout_metrics",c["layer_metrics"]==[])
  require("96newFP32layers16arrays",layer_count==96 and full_layers==16)
  summary=read(ROOT/"analysis/fixed_state_summary.json")
  require("fixed_analysis_COMPLETE",summary["status"]=="COMPLETE" and (summary["complete_cells"],summary["complete_layers"],summary["complete_four_grid_units"])==(96,192,24))
  for name,value in summary["output_hashes"].items():require("fixed_analysis_hash:"+name,sha(ROOT/"analysis"/name)==value)
  manifest=read(ROOT/"outputs/manifest.json");present={}
  for item in manifest["rows"]:
   key=(item["label"],item["request_id"]);require("manifest_unique:"+str(key),key not in present);present[key]=item
  expected_outputs={};verified_receipts={}
  for row in plan["source_rows"]:
   n,k=row["N"],row["K"];label,rid=row["label"],row["request_id"]
   require(rid+":input_hash",sha(row["input_path"])==row["input_sha256"])
   with np.load(row["input_path"],allow_pickle=False) as zz:d={key:zz[key].copy() for key in zz.files}
   require(rid+":valid_known_prefix",d["known_mask"].dtype==d["valid_mask"].dtype==bool and np.array_equal(d["known_mask"],np.arange(n)<k) and bool(d["valid_mask"].all()) and (int(d["N"]),int(d["K"]))==(n,k))
   with np.load(row["raw_path"],allow_pickle=False) as zz:oldtrajectory=zz["trajectory"].copy()
   for prec,mode in (("LEGACY_BF16","ON"),("LEGACY_BF16","OFF"),("FP32_REFERENCE","ON"),("FP32_REFERENCE","OFF")):
    name=label+"__"+prec+"__"+mode;key=(name,rid);cached=prec=="LEGACY_BF16" and mode=="ON"
    if cached:
     item=dict(row,model_label=label,label=name,precision=prec,readout=mode,generation_status="VERIFIED_CACHE",ply_path=str(Path(row["raw_path"]).with_suffix(".ply")))
    else:
     rp=ROOT/"outputs"/name/rid/"receipt.json";item=read(rp)
     require(name+"__"+rid+":rollout_identity",item["model_label"]==label and item["label"]==name and item["request_id"]==rid and item["precision"]==prec and item["readout"]==mode and item["checkpoint_sha256"]==row["checkpoint_sha256"] and item["input_sha256"]==row["input_sha256"])
     verified_receipts[key]=item;source_hashes[str(rp)]=sha(rp)
    expected_outputs[key]=item
    require(name+"__"+rid+":raw_hash",sha(item["raw_path"])==item["raw_sha256"])
    ply=Path(item["ply_path"]);ply_hash=sha(ply)
    if not cached:require(name+"__"+rid+":ply_hash",ply_hash==item["ply_sha256"])
    audit=read(item["result_path"])
    with np.load(item["raw_path"],allow_pickle=False) as z:
     raw=z["raw"];tr=z["trajectory"];start=d["z"][0].copy();start[:k]=d["C"].reshape(k,9)
     require(name+"__"+rid+":raw_shapes_finite",raw.shape==(n,9) and tr.shape==(51,n,9) and raw.dtype==tr.dtype==np.float32 and np.isfinite(tr).all() and np.isfinite(raw).all())
     require(name+"__"+rid+":raw_final",same(raw,tr[-1]))
     require(name+"__"+rid+":actual_gaussian_C",same(z["z"],d["z"]) and same(z["C"],d["C"]) and same(tr[0],start) and all(t[:k].tobytes()==d["C"].tobytes() for t in tr))
     require(name+"__"+rid+":free_and_mapping",same(z["free"],raw[k:].reshape(-1,3,3)) and np.array_equal(z["known_face_indices"],np.arange(k,dtype=np.int64)) and np.array_equal(z["source_corner_to_output_vertex"],np.arange(3*k,dtype=np.int64)))
     require(name+"__"+rid+":PLY_byte_order_winding",same(read_raw_ply(ply,n),raw))
     if not cached:
      vv=z["velocities"];require(name+"__"+rid+":50actual_velocities",vv.shape==(50,n,9) and vv.dtype==np.float32 and np.isfinite(vv).all())
      require(name+"__"+rid+":stored_masks",same(z["known_mask"],d["known_mask"]) and same(z["valid_mask"],d["valid_mask"]))
      require(name+"__"+rid+":sampling_contract",audit["nfe"]==50 and audit["known_state_checks"]==51 and audit["precision"]==prec and audit["readout"]==mode and audit["checkpoint_sha256"]==row["checkpoint_sha256"] and audit["input_sha256"]==row["input_sha256"] and len(audit["step_checks"])==50)
      for s in range(50):
       cid=f"rollout__{name}__{rid}__s{s:02d}";c=commits[cid][1]
       calc=tr[s].copy();calc[k:]=tr[s,k:]+vv[s,k:]*(1/50);calc[:k]=d["C"].reshape(k,9)
       require(cid+":own_recurrence_exact",same(calc,tr[s+1]))
       with np.load(c["array_path"],allow_pickle=False) as stepdata:
        require(cid+":actual_step_capture",same(stepdata["state"][0],tr[s]) and same(stepdata["next_state"][0],tr[s+1]) and same(stepdata["velocity"][0],vv[s]))
       if prec=="LEGACY_BF16":require(cid+":BF16_stored_values",bool(((vv[s].view(np.uint32)&65535)==0).all()))
      recurrence=True
     else:
      require(name+"__"+rid+":cached_trajectory_source",same(tr,oldtrajectory))
      recurrence=None
     same_old=sum(same(tr[s],oldtrajectory[s]) for s in range(51))
    output_rows.append(dict(label=name,model_label=label,request_id=rid,panel=row["panel"],N=n,K=k,cached=cached,states=51,actual_velocities=0 if cached else 50,
     own_recurrence_verified=recurrence,initial_Gaussian_exact=True,C_all51_exact=True,mapping_exact=True,PLY_bitwise=True,
     states_equal_to_old_ON=same_old,equality_to_old_ON_is_not_a_gate=True,raw_sha256=item["raw_sha256"],ply_sha256=ply_hash,status="PASS",
     seconds=audit.get("seconds"),peak_allocated_bytes=None if cached else audit["cuda_peak_allocated_bytes"],peak_reserved_bytes=None if cached else audit["cuda_peak_reserved_bytes"]))
    if key in present:
     require(name+"__"+rid+":manifest_binding",all(present[key].get(field)==item.get(field) for field in ("raw_path","raw_sha256","checkpoint_sha256","input_sha256","precision","readout","model_label")))
  require("no_unexpected_manifest_output",set(present)<=set(expected_outputs))
  for event in ("ROLLOUT_START","ROLLOUT_COMPLETE"):
   ee=[e for e in ledger if e["event"]==event];ids=[e["id"] for e in ee]
   expected_ids={name+"__"+rid for name,rid in verified_receipts}
   require(event+"_24unique",len(ids)==len(set(ids))==24 and set(ids)==expected_ids)
  for p,h in plan["protected_files"].items():require("protected:"+p,sha(p)==h)
  execution=read(ROOT/"execution_source_hashes.json")
  for p,h in execution.items():require("executed_source:"+p,sha(p)==h)
  integrity=[]
  for stage in ("preflight","fixed-state","rollout"):
   for label in plan["labels"]:
    v=read(ROOT/"integrity"/(stage+"__"+label+".json"));oldi=read(old/(label+"_integrity.json"))
    require(stage+"__"+label+":state_rng_frozen",v["unchanged"] is True and v["optimizer"] is False and v["before"]==v["after"]==oldi["model_state_sha256_before"] and v["RNG_before"]==v["RNG_after"])
    integrity.append(dict(stage=stage,label=label,**v))
  occ=rows(ROOT/"budget/gpu_occupancy.jsonl");starts=[v for v in occ if v["event"]=="START"];ends=[v for v in occ if v["event"]=="END"]
  require("GPU_owner_closed",len({v["id"] for v in starts})==len(starts)==len(ends) and {v["id"] for v in starts}=={v["id"] for v in ends})
  elapsed=sum(v["seconds"] for v in ends);require("GPU_under45min",elapsed<=2700,elapsed)
  b=read(ROOT/"budget.json")
  require("budget_matches_events",b["actual_forwards"]==1260 and b["returned_forwards"]==1260 and b["committed_forwards"]==1260 and b["by_kind"]==dict(preflight=12,fixed=48,sampling=1200) and b["new_rollout_attempts"]==b["new_rollouts_complete"]==24 and b["cached_rollouts"]==8 and b["cached_fixed_cells"]==48 and all(b[k]==0 for k in ("backward","optimizer_updates","new_gaussians","new_conditions")) and not b["errors"])
  missing=set(expected_outputs)-set(present)
  if missing:
   require("manifest_missing_only_new_verified",missing<=set(verified_receipts))
   before_sha=sha(ROOT/"outputs/manifest.json")
   save(ROOT/"manifest_before_cpu_repair.json",manifest)
   manifest["rows"] += [expected_outputs[k] for k in sorted(missing)]
   save(ROOT/"outputs/manifest.json",manifest)
   repair=dict(kind="CPU_MANIFEST_ASSEMBLE_ONLY",missing_rows_added=[list(k) for k in sorted(missing)],before_sha256=before_sha,after_sha256=sha(ROOT/"outputs/manifest.json"),model_calls=0)
   save(ROOT/"manifest_cpu_repair.json",repair)
  require("final32unique_outputs",len(manifest["rows"])==32 and len({(v["label"],v["request_id"]) for v in manifest["rows"]})==32)
  result.update(actual_new_forwards=1260,preflight_forwards=12,fixed_forwards=48,sampling_forwards=1200,
   new_rollouts=24,cached_rollouts=8,verified_outputs=32,verified_new_euler_steps=1200,new_fixed_layers=layer_count,representative_saved_layers=full_layers,
   no_training_or_backward=True,GPU_owner_seconds=elapsed,source_file_count=len(execution),protected_file_count=len(plan["protected_files"]),integrity=integrity,
   resource_scope="Per-rollout PyTorch peaks include loaded weights and journaling; loading peak excluded. Rollout integrity counters were reset again per rollout and describe only the last reset window, not the entire stage. Device/process memory are separate.",
   maximum_rollout_allocated_bytes=max(v["peak_allocated_bytes"] for v in output_rows if not v["cached"]),
   maximum_rollout_reserved_bytes=max(v["peak_reserved_bytes"] for v in output_rows if not v["cached"]),
   manifest_repair=repair,manifest_sha256=sha(ROOT/"outputs/manifest.json"),verifier_sha256=sha(__file__),
   source_receipt_hashes=source_hashes,ledger_sha256=sha(ROOT/"budget/calls.jsonl"),plan_sha256=sha(ROOT/"plan.json"))
 except Exception as ex:
  errors.append(dict(type=type(ex).__name__,message=str(ex)))
 result.update(status="PASS" if not errors else "FAIL",errors=errors,checks_count=len(checks),checks=checks,cpu_seconds=time.perf_counter()-tick)
 if output_rows:
  with (ROOT/"independent_output_verification.csv").open("w",encoding="utf-8-sig",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(output_rows[0]));w.writeheader();w.writerows(output_rows)
 save(ROOT/"independent_verification.json",result)
 print(json.dumps({k:v for k,v in result.items() if k not in ("checks","integrity","source_receipt_hashes")},ensure_ascii=True))
 return not errors
if __name__=="__main__":raise SystemExit(0 if verify() else 2)
