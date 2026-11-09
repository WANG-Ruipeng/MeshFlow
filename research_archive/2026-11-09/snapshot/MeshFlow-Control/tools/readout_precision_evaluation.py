#!/usr/bin/env python3
"""CPU-only fixed precision/readout rollout scoring and gallery; old evidence read-only."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[_k]="2"
import argparse,copy,csv,hashlib,html,importlib,itertools,json,math,statistics,subprocess,sys,time,traceback,types
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
ev=importlib.import_module("meshflow_control.evaluation.evaluate")
ren=importlib.import_module("meshflow_control.evaluation.render")
from meshflow_control.evaluation import registry
from meshflow_control.evaluation.io import resolve,digest,read_json,write_json,write_csv,arrays,array_sha
VERSION="readout_precision_rollout_evaluation_v1"
MODELS=("B1000","C1000")
PRECISIONS=("LEGACY_BF16","FP32_REFERENCE")
SWITCHES=("ON","OFF")
REQUEST_IDS=("P1__33ec1e64a02ff718faaadcdc0221eb72_C1__n4","P1__f59627a077dcc2566c63b8b48495261a_C1__n4","P2_seat2__n0","P2_seat_back4__n0")
DEFAULT_OUT="E:/MeshFlow-Control-artifacts/followups/readout_precision_path_v1"
REPO=Path(__file__).resolve().parents[1]
REQUIRED_REQUEST=("request_id","panel","cohort","condition_id","parent_id","candidate","N","K","input_path","input_sha256","reference_path","reference_sha256")
QUAD=(("LEGACY_BF16","ON"),("LEGACY_BF16","OFF"),("FP32_REFERENCE","ON"),("FP32_REFERENCE","OFF"))
CONTRAST_COEFFICIENTS={"branch_BF":(1,-1,0,0),"branch_FP":(0,0,1,-1),"precision_ON":(-1,0,1,0),"precision_OFF":(0,-1,0,1),"interaction":(-1,1,1,-1)}
CONTRACT_NAMES=("C_output_bitwise","N_preserved","finite","C_all51_bitwise","actual_Gaussian_bitwise","NPZ_C_bitwise","PLY_C_bitwise")
def canonical_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def full_label(m,p,s):return f"{m}__{p}__{s}"
def expected_keys():return set(itertools.product(MODELS,PRECISIONS,SWITCHES,REQUEST_IDS))
def folder(out):
    out=resolve(out).resolve()
    if out.name!="readout_precision_path_v1" or out.parent.name!="followups":raise ValueError("Output outside authorized followup")
    out.mkdir(parents=True,exist_ok=True);return out
def artifacts_for(out,artifacts=None):return resolve(artifacts).resolve() if artifacts else folder(out).parents[1]
def source_map():
    return dict(adapter=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),scoring_package=ev.sources(),
        renderer_path=str(Path(ren.__file__).resolve()),renderer_sha256=digest(ren.__file__),definition_change=False,
        P2_reference_adapter="Function-local globals copy: open_reference loads existing frozen I_reference; original module globals unchanged.",
        omitted_wrappers="No original aggregate/diversity/fixed-seven-label render wrapper; score_request mathematics retained.")
def selected_originals(art):
    rows=read_json(art/"outputs/all_completed.json")["rows"];by={(r["label"],r["request_id"]):r for r in rows}
    return {(m,r):copy.deepcopy(by[m,r]) for m,r in itertools.product(MODELS,REQUEST_IDS)}
def canonical_raw(d,N,K):
    return dict(raw=np.asarray(d.get("raw",d.get("output"))).reshape(N,9),
        trajectory=np.asarray(d.get("trajectory",d.get("path"))).reshape(51,N,9),
        C=np.asarray(d["C"]).reshape(K,3,3),z=np.asarray(d["z"]).reshape(1,N,9))
def read_only_I_reference(c,cid,folder=None):
    d=resolve(folder);meta=read_json(d/"I_reference.json");p=d/"I_reference.npz"
    if digest(p)!=meta["npz_sha256"]:raise ValueError("Frozen P2 I NPZ changed")
    aa=arrays(p)
    if {k:array_sha(v) for k,v in aa.items()}!=meta["arrays_sha256"]:raise ValueError("Frozen P2 I arrays changed")
    ident=meta["identity"]
    if ident["condition_id"]!=cid or ident["C_sha256"]!=array_sha(c["C"]) or float(ident["L0"])!=float(c["L0"]):raise ValueError("P2 reference identity differs")
    return dict(arrays=aa,identity=ident)
def original_score_function():
    gg=dict(ev.score_request.__globals__);gg["open_reference"]=read_only_I_reference
    return types.FunctionType(ev.score_request.__code__,gg,"score_request_frozen_queries",ev.score_request.__defaults__,ev.score_request.__closure__)
def validate_original(original,art):
    model,rid=original["label"],original["request_id"];sp=art/"evaluation/scoring"/model/rid/"score.json";saved=read_json(sp)
    if saved.get("status")!="PASS":raise ValueError("Historical score not PASS")
    for k in ("request_id","label","raw_sha256","input_sha256","reference_sha256"):
        if saved["identity"][k]!=original[k]:raise ValueError("Historical identity differs: "+k)
    if saved["identity"]["sources"]!=ev.sources():raise ValueError("Original scoring source hashes changed")
    hashes={str(sp):digest(sp)}
    for k,hk in (("raw_path","raw_sha256"),("input_path","input_sha256"),("reference_path","reference_sha256")):
        if digest(original[k])!=original[hk]:raise ValueError("Historical asset changed: "+k)
        hashes[original[k]]=original[hk]
    hashes[original["result_path"]]=digest(original["result_path"])
    bank,meta,c=ev._data(original);raw=arrays(original["raw_path"]);result=read_json(original["result_path"])
    _,facts=ev._contracts(original,raw,bank,result)
    if any(facts.get(k) is not True for k in CONTRACT_NAMES):raise ValueError("Historical hard/export contract not proven")
    canon=canonical_raw(raw,original["N"],original["K"])
    if array_sha(canon["raw"])!=array_sha(canon["trajectory"][-1]):raise ValueError("Historical final/trajectory differs")
    for p,h in saved.get("files",{}).items():
        if digest(p)!=h:raise ValueError("Historical score witness changed")
        hashes[p]=h
    if original["panel"]=="P2":
        d=resolve(original["reference_path"]).parent;read_only_I_reference(c,original["condition_id"],d)
        for name in ("I_reference.json","I_reference.npz"):hashes[str(d/name)]=digest(d/name)
    else:
        for spec in meta["assets"].values():
            if digest(spec["path"])!=spec["sha256"]:raise ValueError("P1 frozen query changed")
            hashes[spec["path"]]=spec["sha256"]
    return dict(model_label=model,request_id=rid,score_path=str(sp),score_sha256=digest(sp),
        raw_path=original["raw_path"],raw_sha256=original["raw_sha256"],
        result_path=original["result_path"],result_sha256=digest(original["result_path"]),
        scalar_metrics_sha256=canonical_sha(saved["metrics"]),array_hashes={k:array_sha(v) for k,v in canon.items()},
        hashes=hashes,contracts=facts,scientific_scores_recomputed=0)
def prepare(out,artifacts=None):
    out=folder(out);art=artifacts_for(out,artifacts);dest=out/"evaluation";originals=selected_originals(art)
    receipt=dict(schema=VERSION,status="PASS",artifacts=str(art),source_manifest_sha256=digest(art/"outputs/all_completed.json"),
        original_requests_sha256=digest(art/"data/evaluation/requests.json"),
        entries=[validate_original(r,art) for r in originals.values()],model_calls=0,geometry_rescoring=0,cache_count=8)
    write_json(dest/"baseline_cache_manifest.json",receipt,freeze=True)
    write_json(dest/"evaluation_source_map.json",source_map(),freeze=True)
    if not (dest/"git_status_before.txt").exists():
        v=subprocess.run(["git","-C",str(REPO),"status","--short"],capture_output=True,text=True,check=True)
        (dest/"git_status_before.txt").write_text(v.stdout,encoding="utf8")
    print(json.dumps(dict(stage="prepare",status="PASS",verified_baselines=8,scientific_scores_recomputed=0)));return receipt
def read_manifest(out):
    out=folder(out);data=read_json(out/"outputs/manifest.json");rows=data["rows"] if isinstance(data,dict) else data
    if len(rows)!=32:raise ValueError("All32 registered terminal outputs required before scoring/rendering")
    got={}
    for original in rows:
        r=copy.deepcopy(original)
        for k in REQUIRED_REQUEST:
            if k not in r:raise ValueError("Output manifest missing "+k)
        m,p,s=r["model_label"],r["precision"],str(r["readout"]).upper();r["readout"]=s
        if r["label"]!=full_label(m,p,s):raise ValueError("Composite label differs")
        key=(m,p,s,r["request_id"])
        if key in got:raise ValueError("Duplicate output cell")
        got[key]=r
        if r.get("status") in ("PENDING","RUNNING","NOT_RUN"):raise ValueError("Output still pending")
        if r.get("status") not in ("FAILED","BLOCKED","NONFINITE","INVALID_OUTPUT"):
            for k in ("raw_path","result_path"):
                if not r.get(k) or not resolve(r[k]).is_file():raise ValueError("Completed asset absent: "+k)
            if not r.get("raw_sha256") or digest(r["raw_path"])!=r["raw_sha256"]:raise ValueError("RAW hash absent/changed")
    if set(got)!=expected_keys():raise ValueError("Output cells differ from fixed2x4x4")
    return [got[k] for k in sorted(got)]
def cache_new_row(r,out,entry):
    old=read_json(entry["score_path"])
    if digest(entry["score_path"])!=entry["score_sha256"]:raise ValueError("Frozen old score changed")
    for p,h in entry["hashes"].items():
        if digest(p)!=h:raise ValueError("Frozen baseline witness changed")
    bank,meta,c=ev._data(r);raw=arrays(r["raw_path"]);result=read_json(r["result_path"]);_,facts=ev._contracts(r,raw,bank,result)
    if any(facts.get(k) is not True for k in CONTRACT_NAMES):raise ValueError("Cached output contracts failed")
    canon=canonical_raw(raw,r["N"],r["K"])
    if {k:array_sha(v) for k,v in canon.items()}!=entry["array_hashes"]:raise ValueError("BF16_ON arrays differ")
    row=copy.deepcopy(old)
    for k in REQUIRED_REQUEST+("label","model_label","precision","readout","raw_path"):row[k]=r[k]
    row["original_scoring_identity"]=old["identity"];row["identity"]=dict(old["identity"],label=r["label"],raw_sha256=r["raw_sha256"])
    row.update(cache="VERIFIED_PREVIOUS_BF16_ON",scientific_scores_recomputed=0,original_scoring_seconds=old.get("scoring_seconds"),
        scoring_seconds=0,source_score_path=entry["score_path"],source_score_sha256=entry["score_sha256"],source_result_path=r["result_path"],contracts=facts)
    if canonical_sha(row["metrics"])!=entry["scalar_metrics_sha256"]:raise ValueError("Cached metric values changed")
    return row
def _worker(args):
    r,out,entry=args;out=resolve(out);dest=out/"evaluation/rows"/r["label"]/(r["request_id"]+".json")
    ident=dict(version=VERSION,label=r["label"],request_id=r["request_id"],raw_sha256=r.get("raw_sha256"),
        input_sha256=r["input_sha256"],reference_sha256=r["reference_sha256"],sources=ev.sources(),adapter_sha256=digest(__file__))
    if dest.exists():
        saved=read_json(dest)
        if saved.get("adapter_identity")!=ident:raise ValueError("CPU cached result identity changed")
        for p,h in saved.get("files",{}).items():
            if digest(p)!=h:raise ValueError("Completed score witness changed")
        return saved
    tick=time.perf_counter()
    try:
        if r.get("status") in ("FAILED","BLOCKED","NONFINITE","INVALID_OUTPUT"):raise ValueError("Generation failed; denominator retained")
        if entry is not None:row=cache_new_row(r,out,entry)
        else:
            row=original_score_function()(r,out/"evaluation")
            for k in ("model_label","precision","readout"):row[k]=r[k]
            row["cache"]="NEW_SCIENTIFIC_SCORE"
        row.update(adapter_identity=ident,cpu_adapter_seconds=time.perf_counter()-tick,execution_precision=r["precision"],rollout_readout=r["readout"])
    except Exception as exc:
        row={k:r.get(k) for k in REQUIRED_REQUEST+("label","model_label","precision","readout")}
        row.update(status="SCORING_BLOCKED",metrics={},metric_status={},contracts={},files={},adapter_identity=ident,
            error=repr(exc),traceback=traceback.format_exc(),scientific_scores_recomputed=None,cpu_adapter_seconds=time.perf_counter()-tick)
    write_json(dest,row);return row
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def metric_interval(metric,values):
    v=values.get(metric)
    if not finite(v):return None,None
    if metric=="ExposedRetain":return values.get("ExposedRetain_global_lower",v),values.get("ExposedRetain_global_upper",v)
    up={"SeamCov":"SeamCov_upper","ConflictCov_C":"ConflictCov_C_upper"}.get(metric)
    if metric.startswith("SeamCov_") and not metric.startswith("SeamCov_upper"):up="SeamCov_upper"+metric[len("SeamCov"):]
    return float(v),values.get(up,v) if up else float(v)
def contrasts(values,intervals):
    out={}
    for name,coef in CONTRAST_COEFFICIENTS.items():
        selected=[i for i,c in enumerate(coef) if c]
        out[name]=math.fsum(coef[i]*values[i] for i in selected) if all(finite(values[i]) for i in selected) else None
        if all(all(finite(v) for v in intervals[i]) for i in selected):
            out[name+"_lower"]=sum(coef[i]*(intervals[i][0] if coef[i]>0 else intervals[i][1]) for i in selected)
            out[name+"_upper"]=sum(coef[i]*(intervals[i][1] if coef[i]>0 else intervals[i][0]) for i in selected)
        else:out[name+"_lower"]=out[name+"_upper"]=None
    return out
def summarize(out):
    out=folder(out);rows=read_json(out/"evaluation/per_output.json")["rows"]
    by={(r["model_label"],r["precision"],r["readout"],r["request_id"]):r for r in rows}
    if set(by)!=expected_keys():raise ValueError("Summary requires32 fixed records")
    metrics=sorted({k for r in rows for k,v in r["metrics"].items() if finite(v) or v is None});flat=[]
    for r in rows:
        record={k:r.get(k) for k in REQUIRED_REQUEST+("label","model_label","precision","readout","status")}
        record.update(cache=r.get("cache"),scientific_scores_recomputed=r.get("scientific_scores_recomputed"),
            scoring_seconds=r.get("scoring_seconds"),cpu_adapter_seconds=r.get("cpu_adapter_seconds"),
            reference_completion_scope="REFERENCE_COMPLETION_DIAGNOSTICS" if r["panel"]=="P1" else "NO_SOURCE_GT",
            human_usability="NOT_RUN",**r["metrics"])
        record.update({k:r.get("contracts",{}).get(k) for k in CONTRACT_NAMES})
        for k,v in r.get("metric_status",{}).items():record[k+"_status"]=v
        flat.append(record)
    write_csv(out/"rollout_metrics.csv",flat);paired=[]
    for model,rid in itertools.product(MODELS,REQUEST_IDS):
        qq=[by[model,p,s,rid] for p,s in QUAD]
        for metric in metrics:
            vals=[x["metrics"].get(metric) for x in qq];ints=[metric_interval(metric,x["metrics"]) for x in qq];r=qq[0]
            spec=registry.describe(metric,r["panel"],r["N"],r["K"],r.get("L0") or 1)
            row=dict(model_label=model,request_id=rid,panel=r["panel"],condition_id=r["condition_id"],parent_id=r["parent_id"],
                candidate=r["candidate"],metric=metric,BF16_ON=vals[0],BF16_OFF=vals[1],FP32_ON=vals[2],FP32_OFF=vals[3],
                finite_cells=sum(finite(v) for v in vals),expected_cells=4,
                status="COMPLETE" if all(finite(v) for v in vals) else "NA_NOT_APPLICABLE" if r["panel"]=="P2" and spec.get("requires_GT") else "INCOMPLETE",
                interval_basis="global reference-eligibility bounds" if metric=="ExposedRetain" else "original metric bounds or point values",
                interpretation="Descriptive final-rollout four-cell contrast; independent trajectories, no same-state derivative.",**contrasts(vals,ints))
            for tag,x in zip(("BF16_ON","BF16_OFF","FP32_ON","FP32_OFF"),qq):row[tag+"_status"]=x.get("metric_status",{}).get(metric,x["status"])
            paired.append(row)
    write_csv(out/"paired_rollout_comparisons.csv",paired)
    grouped=defaultdict(list)
    for row in paired:grouped[row["model_label"],row["panel"],row["metric"]].append(row)
    agg=[]
    for (model,panel,metric),rr in grouped.items():
        assert len(rr)==2
        for name in CONTRAST_COEFFICIENTS:
            vv=[x[name] for x in rr if finite(x[name])];lo=[x[name+"_lower"] for x in rr if finite(x[name+"_lower"])];hi=[x[name+"_upper"] for x in rr if finite(x[name+"_upper"])]
            agg.append(dict(model_label=model,panel=panel,metric=metric,contrast=name,expected_conditions=2,
                finite_conditions=len(vv),NA_conditions=2-len(vv),mean=statistics.mean(vv) if len(vv)==2 else None,
                observed_mean=statistics.mean(vv) if vv else None,minimum=min(vv) if vv else None,maximum=max(vv) if vv else None,
                positive_conditions=sum(v>0 for v in vv),negative_conditions=sum(v<0 for v in vv),zero_conditions=sum(v==0 for v in vv),
                mean_lower=statistics.mean(lo) if len(lo)==2 else None,mean_upper=statistics.mean(hi) if len(hi)==2 else None,
                meaning="Condition-equal; panel scales not pooled; sign not automatic quality judgment."))
    write_csv(out/"evaluation/paired_panel_summary.csv",agg)
    write_csv(out/"evaluation/metric_registry_results.csv",[v for r in rows if r.get("L0") for v in registry.records(r)])
    summary=dict(status="COMPLETE" if all(r["status"]=="PASS" for r in rows) else "INCOMPLETE",
        expected_outputs=32,scored=sum(r["status"]=="PASS" for r in rows),
        verified_cached_outputs=sum(r.get("cache")=="VERIFIED_PREVIOUS_BF16_ON" for r in rows),
        new_scored_outputs=sum(r.get("scientific_scores_recomputed")==1 for r in rows),paired_metric_rows=len(paired),
        actual_conditions=4,P1_parents=2,P2_handcrafted_conditions=2,candidates_per_model_condition=1,
        diversity="NOT_APPLICABLE_ONE_NOISE",human_usability="NOT_RUN",
        references="P1 S/E/source distances are reference completion diagnostics, not unique-target gates. P2 has no source GT.",
        source_hashes={n:digest(out/n) for n in ["rollout_metrics.csv","paired_rollout_comparisons.csv","evaluation/paired_panel_summary.csv"]})
    write_json(out/"evaluation/summary.json",summary);return summary
def score(out,workers=2,artifacts=None):
    out=folder(out);art=artifacts_for(out,artifacts)
    if not (out/"evaluation/baseline_cache_manifest.json").exists():prepare(out,art)
    if read_json(out/"evaluation/evaluation_source_map.json")!=source_map():raise ValueError("Frozen scoring sources differ")
    manifest=read_manifest(out);receipt=read_json(out/"evaluation/baseline_cache_manifest.json")
    entries={(e["model_label"],e["request_id"]):e for e in receipt["entries"]}
    try:
        import psutil
        available=psutil.virtual_memory().available;cores=psutil.cpu_count()
    except ImportError:available=None;cores=os.cpu_count()
    workers=max(1,min(int(workers),2))
    if available is not None and available<4*1024**3:workers=1
    args=[(r,str(out),entries[r["model_label"],r["request_id"]] if r["precision"]=="LEGACY_BF16" and r["readout"]=="ON" else None) for r in manifest]
    tick=time.perf_counter()
    if workers==1:results=[_worker(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(_worker,args))
    write_json(out/"evaluation/per_output.json",dict(schema=VERSION,rows=results));summary=summarize(out)
    run=dict(status=summary["status"],workers=workers,threads_per_worker=2,available_RAM_before=available,cpu_cores=cores,
        wall_seconds=time.perf_counter()-tick,model_calls=0,expected_outputs=32,baseline_scores_recomputed=0,
        diversity_computed=False,source_manifest_sha256=digest(out/"outputs/manifest.json"),summary=summary)
    p=out/"evaluation/execution.json";write_json(out/"evaluation/last_resume.json" if p.exists() else p,run)
    print(json.dumps(run));return run
def page_four_cells(p,columns,bounds,title):
    """Original rendering geometry/material/views, transposed to four-cell columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    n=len(columns);fig=plt.figure(figsize=(3*n,14.5),dpi=110,facecolor="white")
    lo,hi=np.asarray(bounds[0]),np.asarray(bounds[1]);center=(lo+hi)/2;span=max(float(np.max(hi-lo)),.01)*1.08
    for i,(view,elev,azim) in enumerate(ren.VIEWS):
        for j,(label,C,F) in enumerate(columns):
            ax=fig.add_subplot(5,n,i*n+j+1,projection="3d");tri=np.concatenate([F,C])
            colors=["#e7994c"]*len(F)+["#3a82c5"]*len(C);displayed=np.asarray(tri)[:,:,(0,2,1)]
            if view=="wire":
                edges=displayed[:,((0,1),(1,2),(2,0)),:].reshape(-1,2,3)
                artist=Line3DCollection(edges,colors=np.repeat(colors,3),linewidths=.38)
            else:artist=Poly3DCollection(displayed,facecolors=colors,edgecolors="#444444",linewidths=.10,alpha=1)
            ax.add_collection3d(artist);cc=center[(0,2,1),]
            ax.set_xlim(cc[0]-span/2,cc[0]+span/2);ax.set_ylim(cc[1]-span/2,cc[1]+span/2);ax.set_zlim(cc[2]-span/2,cc[2]+span/2)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_axis_off();ax.set_title(label+" / "+view,fontsize=8,pad=0)
    fig.suptitle(title,fontsize=12,y=.994);fig.subplots_adjust(left=0,right=1,top=.95,bottom=0,wspace=0,hspace=.02)
    p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,facecolor="white");plt.close(fig)
def render(out,artifacts=None):
    out=folder(out);art=artifacts_for(out,artifacts);rows=read_manifest(out)
    if any(r.get("status") in ("FAILED","BLOCKED","NONFINITE","INVALID_OUTPUT") for r in rows):raise ValueError("Failed outputs: cannot silently reduce gallery denominator")
    gallery=out/"gallery";gallery.mkdir(exist_ok=True);old=read_json(art/"gallery/camera_manifest.json")
    identity=dict(source_manifest_sha256=digest(out/"outputs/manifest.json"),original_camera_sha256=digest(art/"gallery/camera_manifest.json"),
        adapter_sha256=digest(__file__),renderer_sha256=digest(ren.__file__))
    mp=gallery/"gallery_manifest.json"
    if mp.exists():
        prior=read_json(mp)
        if prior.get("identity")!=identity:raise ValueError("Completed gallery identity changed")
        for x in prior["pages"]:
            for kind in ("blind","reference"):
                if kind+"_page" in x and digest(x[kind+"_page"])!=x[kind+"_sha256"]:raise ValueError("Completed image changed")
        return prior
    tick=time.perf_counter();camera={};pages=[]
    for rid in REQUEST_IDS:
        rr=[r for r in rows if r["request_id"]==rid];assert len(rr)==8
        first=rr[0];meta=read_json(first["reference_path"]);condition=arrays(meta["condition_path"]);C=condition["C"];K=first["K"];loaded={}
        for r in rr:
            t=ren._triangles(r)
            if t[:K].tobytes()!=C.tobytes() or not np.isfinite(t).all():raise ValueError("Render raw/C contract")
            loaded[r["label"]]=t
        geometry=[C]+list(loaded.values())+([condition["GT_full"]] if first["panel"]=="P1" else [])
        pts=np.concatenate([x.reshape(-1,3) for x in geometry]);newlo,newhi=pts.min(0),pts.max(0)
        prev=np.asarray(old["conditions"][first["condition_id"]]["bounds"]);lo=np.minimum(prev[0],newlo);hi=np.maximum(prev[1],newhi);bounds=(lo,hi)
        camera[rid]=dict(bounds=[lo.tolist(),hi.tolist()],old_bounds=prev.tolist(),expanded=bool(np.any(newlo<prev[0]) or np.any(newhi>prev[1])),
            scope="Both checkpoints and all4 cells; P1 source included; no per-output scaling/centering",views=ren.VIEWS,
            raw_hashes={r["label"]:r["raw_sha256"] for r in rr})
        for model in MODELS:
            cols=[("Input C",C,np.empty((0,3,3),np.float32))]
            for precision,switch in QUAD:
                t=loaded[full_label(model,precision,switch)];cols.append((precision+"\n"+switch,C,t[K:]))
            bp=gallery/"blind"/(model+"__"+rid+".png")
            page_four_cells(bp,cols,bounds,model+" | "+rid+" | C blue/free orange | source hidden")
            pg=dict(model_label=model,request_id=rid,panel=first["panel"],condition_id=first["condition_id"],
                labels=[full_label(model,p,s) for p,s in QUAD],candidate=first["candidate"],blind_page=str(bp),blind_sha256=digest(bp),
                view_scope=[x[0] for x in ren.VIEWS],AI_VISUAL_REVIEW="PENDING",human_usability="NOT_RUN")
            if first["panel"]=="P1":
                rp=gallery/"reference"/(model+"__"+rid+".png")
                page_four_cells(rp,cols+[("One source\ncompletion",C,condition["GT_free"])],bounds,
                    model+" | "+rid+" | one source completion, not the unique valid target")
                pg.update(reference_page=str(rp),reference_sha256=digest(rp))
            pages.append(pg)
    write_json(gallery/"camera_manifest.json",dict(status="FINAL_COMPLETE",identity=identity,requests=camera))
    result=dict(status="FINAL_COMPLETE",identity=identity,pages=pages,output_count=32,request_count=4,blind_pages=8,reference_pages=4,
        wall_seconds=time.perf_counter()-tick,workers=1,model_calls=0,mesh_modified=False,AI_VISUAL_REVIEW="PENDING",human_usability="NOT_RUN")
    write_json(mp,result)
    lines=["<!doctype html><html><meta charset='utf-8'><title>Precision/readout four-cell gallery</title>",
        "<h1>Precision/readout:32 raw outputs</h1><p>Four conditions,2 checkpoints,1 Gaussian per condition. C blue/free orange. Four views+wire; no repair. Rendering is not review.</p>",
        "<h2>1. View without source completion</h2>"]
    for pg in pages:
        link=resolve(pg["blind_page"]).relative_to(out).as_posix()
        lines.append("<p><a href='"+html.escape(link)+"'>"+html.escape(pg["model_label"]+" "+pg["request_id"])+"</a></p>")
    lines.append("<h2>2. Source reference (P1 only; not a unique target)</h2>")
    for pg in pages:
        if "reference_page" in pg:
            link=resolve(pg["reference_page"]).relative_to(out).as_posix()
            lines.append("<p><a href='"+html.escape(link)+"'>"+html.escape(pg["model_label"]+" "+pg["request_id"])+"</a></p>")
    lines.append("<p>AI_VISUAL_REVIEW=PENDING until separate actual review receipts. Human usability=NOT_RUN.</p></html>")
    (out/"OPEN_GALLERY.html").write_text("\n".join(lines),encoding="utf8");return result
def verify(out):
    out=folder(out);read_manifest(out);sm=read_json(out/"evaluation/summary.json")
    if sm["status"]!="COMPLETE":raise ValueError("Incomplete scoring")
    po=read_json(out/"evaluation/per_output.json")["rows"];assert len(po)==32 and sum(x.get("cache")=="VERIFIED_PREVIOUS_BF16_ON" for x in po)==8
    for row in po:
        assert row["status"]=="PASS"
        for p,h in row.get("files",{}).items():
            if digest(p)!=h:raise ValueError("Score witness changed")
    g=read_json(out/"gallery/gallery_manifest.json");assert g["status"]=="FINAL_COMPLETE" and len(g["pages"])==8
    for pg in g["pages"]:
        for kind in ("blind","reference"):
            if kind+"_page" in pg and digest(pg[kind+"_page"])!=pg[kind+"_sha256"]:raise ValueError("Image changed")
    result=dict(status="PASS",outputs=32,cached_scores=8,new_scores=24,gallery_outputs=32,
        actual_review_status="See separate visual_review.csv; rendering is not review.",source_map=source_map(),
        files={n:digest(out/n) for n in ["rollout_metrics.csv","paired_rollout_comparisons.csv","OPEN_GALLERY.html"]})
    write_json(out/"evaluation/verify.json",result);return result
def self_test(out):
    out=folder(out);v=[1.,2.,4.,8.];d=contrasts(v,[(x,x) for x in v])
    assert d["branch_BF"]==-1 and d["branch_FP"]==-4 and d["interaction"]==-3
    assert d["precision_ON"]==3 and d["precision_OFF"]==6
    d=contrasts([None,2.,4.,8.],[(None,None),(2.,2.),(4.,4.),(8.,8.)])
    assert d["branch_BF"] is None and d["branch_FP"]==-4 and d["precision_OFF"]==6
    q=contrasts([1.,2.,4.,8.],[(0.,1.),(1.,3.),(3.,5.),(7.,9.)])
    assert q["interaction_lower"]==-6 and q["interaction_upper"]==1
    assert len(expected_keys())==32 and "torch" not in sys.modules
    result=dict(status="PASS",tests=["four_cell_algebra","NA_not_zero","conservative_intervals","fixed32","no_torch_import"],
        adapter_sha256=digest(__file__),model_calls=0,geometry_rescoring=0)
    write_json(out/"evaluation/cpu_interface_test.json",result);print(json.dumps(result));return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=["prepare","score","render","summarize","verify","self-test"],required=True)
    p.add_argument("--out",default=DEFAULT_OUT);p.add_argument("--artifacts");p.add_argument("--workers",type=int,default=2);a=p.parse_args()
    if a.stage=="prepare":r=prepare(a.out,a.artifacts)
    elif a.stage=="score":r=score(a.out,a.workers,a.artifacts)
    elif a.stage=="render":r=render(a.out,a.artifacts)
    elif a.stage=="summarize":r=summarize(a.out)
    elif a.stage=="verify":r=verify(a.out)
    else:r=self_test(a.out)
    if a.stage not in ("prepare","score","self-test"):print(json.dumps(r))
    return 0 if r.get("status") in ("PASS","COMPLETE","FINAL_COMPLETE") else 2
if __name__=="__main__":sys.exit(main())
