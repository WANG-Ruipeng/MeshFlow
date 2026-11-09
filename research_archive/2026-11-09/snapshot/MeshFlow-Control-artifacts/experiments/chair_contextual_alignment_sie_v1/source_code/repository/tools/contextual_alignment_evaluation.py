#!/usr/bin/env python3
"""Frozen contextual-alignment panels and incremental CPU scoring; no Torch/model."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_k]="2"
import argparse,copy,hashlib,importlib,json,shutil,subprocess,sys,time,traceback,types
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from meshflow_control.evaluation.io import resolve,digest,read_json,write_json,write_csv,arrays,array_sha,save_npz,load_manifest
from meshflow_control.evaluation.reference import copy_exact,new_bank
ev=importlib.import_module("meshflow_control.evaluation.evaluate")
VERSION="chair_contextual_alignment_sie_evaluation_v1"
LABELS=("START","A_FM_1000","B_FREE_ALIGN_1000","C_KNOWN_ALIGN_1000")
NAMESPACE="CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1"
DEFAULT_OUT="E:/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1"
START_SHA="d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80"
CONTRACTS=("C_output_bitwise","N_preserved","finite","C_all51_bitwise","actual_Gaussian_bitwise","NPZ_C_bitwise","PLY_C_bitwise")
META=("request_id","label","panel","cohort","condition_id","parent_id","candidate","N","K")
REPO=Path(__file__).resolve().parents[1]

def root(out):
    p=resolve(out).resolve()
    if p.name!="chair_contextual_alignment_sie_v1" or p.parent.name!="experiments": raise ValueError("Not the authorized experiment folder")
    p.mkdir(parents=True,exist_ok=True);return p
def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def seed(uid,index):return int.from_bytes(hashlib.sha256(f"{NAMESPACE}/eval/{uid}/{index}".encode()).digest()[:4],"little")
def request_rows(out):return load_manifest(root(out)/"data/evaluation/requests.json")
def canonical_raw(d,N,K):
    return dict(raw=np.asarray(d.get("raw",d.get("output"))).reshape(N,9),trajectory=np.asarray(d.get("trajectory",d.get("path"))).reshape(51,N,9),C=np.asarray(d["C"]).reshape(K,3,3),z=np.asarray(d.get("z",d.get("initial_gaussian"))).reshape(1,N,9))
def read_only_I_reference(c,cid,folder=None):
    d=resolve(folder);m=read_json(d/"I_reference.json");p=d/"I_reference.npz";aa=arrays(p)
    if digest(p)!=m["npz_sha256"] or {k:array_sha(v) for k,v in aa.items()}!=m["arrays_sha256"]:raise ValueError("P2 frozen I reference changed")
    z=m["identity"]
    if z["condition_id"]!=cid or z["C_sha256"]!=array_sha(c["C"]) or float(z["L0"])!=float(c["L0"]):raise ValueError("P2 I reference identity")
    return dict(arrays=aa,identity=z)
def original_scorer():
    g=dict(ev.score_request.__globals__);g["open_reference"]=read_only_I_reference
    return types.FunctionType(ev.score_request.__code__,g,"score_request_frozen_reference",ev.score_request.__defaults__,ev.score_request.__closure__)

def validate_baseline(r,art):
    p=art/"evaluation/scoring/START"/r["request_id"]/"score.json";s=read_json(p)
    if r.get("checkpoint_sha256")!=START_SHA or s.get("status")!="PASS":raise ValueError("START/cache identity not accepted")
    if s["identity"]["sources"]!=ev.sources():raise ValueError("Cached scorer source mismatch; explicit rescoring decision required")
    for k in ("request_id","label","raw_sha256","input_sha256","reference_sha256"):
        if s["identity"][k]!=r[k]:raise ValueError("START score identity differs: "+k)
    hashes={str(p):digest(p)}
    for k in ("raw","input","reference"):
        if digest(r[k+"_path"])!=r[k+"_sha256"]:raise ValueError("START "+k+" changed")
        hashes[r[k+"_path"]]=r[k+"_sha256"]
    result=read_json(r["result_path"]);hashes[r["result_path"]]=digest(r["result_path"])
    if result.get("status") not in (None,"PASS","COMPLETE"):raise ValueError("Cached START run receipt explicitly unsuccessful")
    b,m,c=ev._data(r);raw=arrays(r["raw_path"]);_,facts=ev._contracts(r,raw,b,result)
    if any(facts.get(k) is not True for k in CONTRACTS):raise ValueError("START hard/51-state/export contract unproven")
    canon=canonical_raw(raw,r["N"],r["K"])
    if array_sha(canon["raw"])!=array_sha(canon["trajectory"][-1]):raise ValueError("START final trajectory not RAW")
    for path,h in s["files"].items():
        if digest(path)!=h:raise ValueError("START vectors changed")
        hashes[path]=h
    if r["panel"]=="P1":
        ev._control_ref(m)
        for v in m["assets"].values():hashes[v["path"]]=v["sha256"]
    else:
        d=resolve(r["reference_path"]).parent;read_only_I_reference(c,r["condition_id"],d)
        for name in ("I_reference.json","I_reference.npz"):hashes[str(d/name)]=digest(d/name)
    return dict(request_id=r["request_id"],checkpoint_sha256=START_SHA,status="VERIFIED_CACHE",
        raw_path=r["raw_path"],raw_sha256=r["raw_sha256"],result_path=r["result_path"],result_sha256=digest(r["result_path"]),
        score_path=str(p),score_sha256=digest(p),quality_vectors_path=s["quality_vectors_path"],control_vectors_path=s["control_vectors_path"],
        scalar_metrics_sha256=canonical_hash(s["metrics"]),hashes=hashes,contracts=facts,
        array_hashes={k:array_sha(v) for k,v in canon.items()},scientific_scores_recomputed=0)

def _prior_banks(art,legacy,conditions):
    # Only registered DEV/OLD/FRESH or quick requests; never CONFIRM manifests/results.
    paths=[art/"data/evaluation/requests.json"]
    for run in ("chair_capped500_half_v1","chair_capped500_gradbudget_v1","chair_control_feedback_auto_v1"):
        for name in ("OLD","FRESH","DEV"):
            p=legacy/"native_t1/runs"/run/"manifests"/(name+".json")
            if p.is_file():paths.append(p)
    quick=legacy/"native_t1/runs/chair_control_usability_quick_v1/requests.json"
    if quick.is_file():paths.append(quick)
    for p in sorted((art/"experiments").glob("*/data/evaluation/requests.json")):
        if "chair_contextual_alignment_sie_v1" not in str(p):paths.append(p)
    records=[];seen=set()
    for p in paths:
        for r in load_manifest(p):
            cid=r.get("condition_id");pid=r.get("parent_id",r.get("uid"))
            if cid not in conditions and pid not in {v["parent_id"] for v in conditions.values()}:continue
            ip=r.get("input_path",r.get("bank_npz",r.get("input_npz")))
            if not ip or str(resolve(ip)) in seen:continue
            ip=resolve(ip)
            if not ip.is_file():raise FileNotFoundError("Registered prior bank missing: "+str(ip))
            seen.add(str(ip));b=arrays(ip)
            if "z" not in b or "C" not in b:continue
            N=int(b["N"]);K=int(b["K"]);z=np.asarray(b["z"]).reshape(1,N,9)
            records.append(dict(manifest=str(p),manifest_sha256=digest(p),path=str(ip),file_sha256=digest(ip),condition_id=cid,parent_id=pid,
                candidate=r.get("candidate"),N=N,K=K,C_sha256=array_sha(np.asarray(b["C"]).reshape(K,3,3)),z_sha256=array_sha(z),free_z_sha256=array_sha(z[:,K:])))
    return records

def prepare(out,artifacts=None,legacy_root="E:/MeshFlow"):
    out=root(out);art=resolve(artifacts) if artifacts else out.parents[1];legacy=resolve(legacy_root);d=out/"data/evaluation"
    original=load_manifest(art/"data/evaluation/requests.json")
    completed={(r["label"],r["request_id"]):r for r in load_manifest(art/"outputs/all_completed.json")}
    old=[r for r in original if r["panel"]=="P2" or (r["panel"]=="P1" and r["candidate"] in (0,1))]
    if len(old)!=32 or len({r["parent_id"] for r in old if r["panel"]=="P1"})!=8:raise ValueError("Expected OLD16/P2 16")
    rows=[];conditions={};cache=[];cached_outputs=[]
    for src in sorted(old,key=lambda r:(r["panel"],r["condition_id"],r["candidate"])):
        r=copy.deepcopy(src);r.pop("historical_start",None);cid=r["condition_id"]
        if cid not in conditions:
            m=read_json(r["reference_path"]);rd=d/"evaluation_reference"/cid;nm=copy.deepcopy(m)
            cp,ch=copy_exact(m["condition_path"],rd/"condition.npz",m["condition_sha256"]);nm.update(condition_path=cp,condition_sha256=ch)
            if r["panel"]=="P1":
                nm["assets"]={name:dict(zip(("path","sha256"),copy_exact(v["path"],rd/name,v["sha256"]))) for name,v in m["assets"].items()}
            else:
                for name in ("I_reference.json","I_reference.npz"):copy_exact(resolve(r["reference_path"]).parent/name,rd/name)
            nm.update(source_reference_path=r["reference_path"],source_reference_sha256=r["reference_sha256"])
            write_json(rd/"reference.json",nm,freeze=True);conditions[cid]=nm
        rd=d/"evaluation_reference"/cid
        ip,ih=copy_exact(r["input_path"],d/"inputs"/(r["request_id"]+".npz"),r["input_sha256"])
        r.update(input_path=ip,input_sha256=ih,reference_path=str(rd/"reference.json"),reference_sha256=digest(rd/"reference.json"))
        base=copy.deepcopy(completed["START",r["request_id"]]);entry=validate_baseline(base,art)
        r["historical_start"]=entry;rows.append(r);cache.append(entry)
        cached_outputs.append(dict(r,label="START",raw_path=entry["raw_path"],raw_sha256=entry["raw_sha256"],result_path=entry["result_path"],result_sha256=entry["result_sha256"],checkpoint_sha256=START_SHA,generation_status="VERIFIED_CACHE"))
    prior=_prior_banks(art,legacy,{k:v for k,v in conditions.items() if v["panel"]=="P1"})
    collisions=[];newhashes=set()
    for cid,m in sorted(conditions.items()):
        if m["panel"]!="P1":continue
        c=arrays(m["condition_path"]);C=c["C"];N,K=m["N"],m["K"];ch=array_sha(C)
        relevant=[p for p in prior if p["N"]==N and p["K"]==K and p["C_sha256"]==ch]
        old_candidates={p["candidate"] for p in relevant}
        if not set(range(6))<=old_candidates:raise ValueError("Prior n0..n5 bank coverage incomplete: "+cid+str(old_candidates))
        for i,n in enumerate((6,7)):
            s=seed(m["parent_id"],i);b=new_bank(C,N,s);h=array_sha(b["z"]);fh=array_sha(b["z"][:,K:])
            duplicate=[p["path"] for p in prior if p["z_sha256"]==h or (p["N"]==N and p["K"]==K and p["free_z_sha256"]==fh)]
            if duplicate or h in newhashes:raise ValueError("New Gaussian collision")
            newhashes.add(h);rid=f"P1__{cid}__n{n}";ip=d/"inputs"/(rid+".npz");save_npz(ip,b)
            rows.append(dict(request_id=rid,panel="P1",cohort="NEW",condition_id=cid,parent_id=m["parent_id"],candidate=n,new_noise_index=i,N=N,K=K,input_path=str(ip),input_sha256=digest(ip),reference_path=str(d/"evaluation_reference"/cid/"reference.json"),reference_sha256=digest(d/"evaluation_reference"/cid/"reference.json"),noise_seed=s,noise_namespace=f"{NAMESPACE}/eval/{m['parent_id']}/{i}",noise_array_sha256=h,C_array_sha256=ch))
            collisions.append(dict(request_id=rid,seed=s,prior_same_condition_banks=len(relevant),prior_candidates=sorted(old_candidates),global_prior_banks=len(prior),duplicate_paths=duplicate,array_sha256=h,status="PASS"))
    rows.sort(key=lambda r:(r["panel"],r["condition_id"],r["candidate"]))
    if len(rows)!=48 or len(newhashes)!=16:raise ValueError("Fixed panel size")
    write_json(d/"requests.json",dict(schema=VERSION,rows=rows,labels=LABELS,expected_output_cells=192),freeze=True)
    for panel in ("P1","P2"):write_json(d/(panel+".json"),dict(rows=[r for r in rows if r["panel"]==panel]),freeze=True)
    write_json(d/"conditions.json",list(conditions.values()),freeze=True)
    write_json(d/"baseline_cache_manifest.json",dict(status="PASS",entries=cache,cache_count=32,scientific_scores_recomputed=0),freeze=True)
    write_json(d/"start_cached_outputs.json",dict(rows=cached_outputs),freeze=True)
    write_json(d/"noise_identity_audit.json",dict(status="PASS",seed_method="SHA256 exact UTF8 namespace, first4 bytes little-endian uint32; MT19937 RandomState standard_normal->FP32; known z=0",new=collisions,prior_banks=prior,CONFIRM_read=False),freeze=True)
    srcdir=out/"source_code/original_scoring"
    for p in sorted(Path(ev.__file__).parent.glob("*.py")):copy_exact(p,srcdir/p.name)
    receipt=dict(status="PASS",requests=48,expected_outputs=192,START_cached=32,new_rollouts=160,model_calls=0,scoring_calls=0,seed_namespace=NAMESPACE,NEW_candidates=[6,7],NEW_is_seen_DEV_new_noise_not_new_parent=True,scorer_sources=ev.sources(),source_requests_sha256=digest(art/"data/evaluation/requests.json"),source_outputs_sha256=digest(art/"outputs/all_completed.json"))
    write_json(d/"preparation_receipt.json",receipt,freeze=True)
    if not (d/"git_status_before.txt").exists():
        x=subprocess.run(["git","-C",str(REPO),"status","--short"],capture_output=True,text=True,check=True);(d/"git_status_before.txt").write_text(x.stdout,encoding="utf8")
    return receipt

def _cache_row(r,out,entry):
    s=read_json(entry["score_path"])
    for p,h in entry["hashes"].items():
        if digest(p)!=h:raise ValueError("Frozen baseline asset changed: "+p)
    b,m,c=ev._data(r);raw=arrays(r["raw_path"]);res=read_json(r["result_path"]);_,facts=ev._contracts(r,raw,b,res)
    if any(facts.get(k) is not True for k in CONTRACTS):raise ValueError("Cached output contract failed")
    if {k:array_sha(v) for k,v in canonical_raw(raw,r["N"],r["K"]).items()}!=entry["array_hashes"]:raise ValueError("START output arrays differ")
    row=copy.deepcopy(s);row.update({k:r[k] for k in META})
    row.update(raw_path=r["raw_path"],contracts=facts,original_scoring_identity=s["identity"],cache="VERIFIED_START_SCALARS",scientific_scores_recomputed=0,scoring_seconds=0,original_scoring_seconds=s.get("scoring_seconds"),source_score_path=entry["score_path"],source_score_sha256=entry["score_sha256"],source_result_path=r["result_path"])
    row["identity"]=dict(s["identity"],raw_sha256=r["raw_sha256"],input_sha256=r["input_sha256"],reference_sha256=r["reference_sha256"])
    if canonical_hash(row["metrics"])!=entry["scalar_metrics_sha256"]:raise ValueError("Cached scalars changed")
    return row

def _worker(args):
    r,out,entry=args;out=root(out);jp=out/"evaluation/rows"/r["label"]/(r["request_id"]+".json")
    identity=dict(version=VERSION,label=r["label"],request_id=r["request_id"],raw_sha256=r.get("raw_sha256"),input_sha256=r["input_sha256"],reference_sha256=r["reference_sha256"],source_hashes=ev.sources(),adapter_sha256=digest(__file__))
    if jp.exists():
        row=read_json(jp)
        if row["adapter_identity"]!=identity:raise ValueError("Completed scoring identity changed")
        for field in ("raw","input","reference"):
            if digest(r[field+"_path"])!=r[field+"_sha256"]:raise ValueError("Completed asset changed: "+field)
        for p,h in row.get("files",{}).items():
            if digest(p)!=h:raise ValueError("Completed vector changed")
        return row
    t=time.perf_counter()
    try:
        if r.get("status") in ("FAILED","BLOCKED","NONFINITE"):raise ValueError("Generation unsuccessful; fixed denominator retained")
        if digest(r["raw_path"])!=r["raw_sha256"]:raise ValueError("Completed RAW hash differs")
        if entry:row=_cache_row(r,out,entry)
        else:
            row=original_scorer()(r,out/"evaluation")
            row["cache"]="NEW_SCIENTIFIC_SCORE"
        row.update(adapter_identity=identity,cpu_adapter_seconds=time.perf_counter()-t)
    except Exception as e:
        row={k:r.get(k) for k in META};row.update(status="SCORING_BLOCKED",metrics={},metric_status={},contracts={},files={},adapter_identity=identity,error=repr(e),traceback=traceback.format_exc(),cpu_adapter_seconds=time.perf_counter()-t)
    write_json(jp,row);return row

def completed_rows(out,manifest=None):
    out=root(out);path=resolve(manifest) if manifest else out/"outputs/manifest.json"
    actual=load_manifest(path) if path.is_file() else []
    # Caches are immutable known completions even before the GPU controller writes its manifest.
    actual += load_manifest(out/"data/evaluation/start_cached_outputs.json")
    by={}
    qs={r["request_id"]:r for r in request_rows(out)}
    for r in actual:
        key=(r["label"],r["request_id"])
        if r["label"] not in LABELS or r["request_id"] not in qs:raise ValueError("Unregistered output")
        q=qs[r["request_id"]]
        for k in ("panel","cohort","condition_id","parent_id","candidate","N","K","input_sha256","reference_sha256"):
            if r[k]!=q[k]:raise ValueError("Output request identity differs: "+k)
        if key in by and by[key].get("raw_sha256")!=r.get("raw_sha256"):raise ValueError("Two outputs in same registered cell")
        by[key]=dict(q,**{k:v for k,v in r.items() if k not in q or k in ("historical_start",)})
    return [by[k] for k in sorted(by)]

def score(out,manifest=None,workers=2,label=None):
    out=root(out);workers=int(workers)
    if not 1<=workers<=2:raise ValueError("At most two scoring processes")
    rows=completed_rows(out,manifest)
    if label:rows=[r for r in rows if r["label"]==label]
    entries={x["request_id"]:x for x in read_json(out/"data/evaluation/baseline_cache_manifest.json")["entries"]}
    todo=[r for r in rows if r.get("status") not in ("RUNNING","PENDING","NOT_RUN")]
    args=[(r,str(out),entries.get(r["request_id"]) if r["label"]=="START" else None) for r in todo]
    preexisting={(r["label"],r["request_id"]) for r in todo if (out/"evaluation/rows"/r["label"]/(r["request_id"]+".json")).exists()}
    t=time.perf_counter()
    if workers==1:rr=[_worker(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:rr=list(pool.map(_worker,args))
    receipt=dict(status="COMPLETE" if all(r["status"]=="PASS" for r in rr) else "PARTIAL",requested=len(rows),processed=len(rr),workers=workers,threads_per_worker=2,wall_seconds=time.perf_counter()-t,model_calls=0,
        prior_score_returns=len(preexisting),new_baseline_copies=sum(r["status"]=="PASS" and r.get("cache")=="VERIFIED_START_SCALARS" and (r["label"],r["request_id"]) not in preexisting for r in rr),
        new_scientific_scores=sum(r["status"]=="PASS" and r.get("scientific_scores_recomputed")==1 and (r["label"],r["request_id"]) not in preexisting for r in rr))
    log=out/"evaluation/executions.jsonl";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf8") as f:f.write(json.dumps(receipt)+"\n")
    from contextual_alignment_analysis import aggregate
    receipt["aggregate"]=aggregate(out)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=("prepare","score","aggregate","render"));p.add_argument("--out",default=DEFAULT_OUT);p.add_argument("--manifest");p.add_argument("--workers",type=int,default=2);p.add_argument("--label",choices=LABELS);a=p.parse_args()
    if a.stage=="prepare":v=prepare(a.out)
    elif a.stage=="score":v=score(a.out,a.manifest,a.workers,a.label)
    elif a.stage=="aggregate":
        from contextual_alignment_analysis import aggregate
        v=aggregate(a.out,compute_diversity=True)
    else:
        from contextual_alignment_gallery import render
        v=render(a.out,a.manifest)
    print(json.dumps(v,ensure_ascii=False))
if __name__=="__main__":main()
