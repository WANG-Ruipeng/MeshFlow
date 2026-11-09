"""CPU-only fixed-state precision analysis. Never imports torch or loads weights.
The model runner may import layer_numbers and pass CPU NumPy captures. All new
artifacts remain external; existing BF16 captures and results are read-only.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
from pathlib import Path
import argparse, collections, csv, hashlib, itertools, json, math, time
import numpy as np

VERSION="readout_precision_fixed_cpu_v1"
DEFAULT_ROOT="/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1"
DEFAULT_OLD="/mnt/e/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1"
LAYERS=(3,6,9,12)
PRECISIONS=("LEGACY_BF16","FP32_REFERENCE")
QUANTILES=(0,5,25,50,75,95,100)
DEFINITIONS={
 "scope":"Fixed existing ON trajectory states; four observed conditions, not 24 independent conditions. No quality inference.",
 "effective":"e=FP64(after)-FP64(before), from captured values.",
 "effective_error_relative_l2":"||e-r||/||r||; FP64 reduction.",
 "effective_residual_projection":"<e,r>/||r||^2; unbounded.",
 "effective_over_residual_l2":"||e||/||r||; unbounded above, not retained-information percentage.",
 "residual_nonzero_swallowed_fraction":"count(r!=0 and e==0)/count(r!=0); not information loss.",
 "zero_denominators":"None with explicit UNDEFINED status in JSON; UNDEFINED in CSV. No epsilon denominator or zero/one imputation.",
 "near_zero":"No cutoff changes a nonzero denominator; report absolute norms and the denominator beside ratios.",
 "local_fp32":"Same saved BF16-path h and original FP32 r, NumPy FP32 add h+r. Local arithmetic only, no downstream model.",
 "D_BF":"v_BF_ON-v_BF_OFF on valid free coordinates.",
 "D_FP":"v_FP_ON-v_FP_OFF on valid free coordinates.",
 "P":"v_FP_OFF-v_BF_OFF: precision change with new readout bypassed.",
 "J":"D_FP-D_BF: precision/readout interaction; not unique residual-addition causality.",
 "algebra":"v_FP_ON-v_BF_ON=P+J; FP64 check with explicit floating-roundoff bound.",
 "relative_difference":"||D_FP-D_BF||/||D_BF|| plus symmetric 2||D_FP-D_BF||/(||D_FP||+||D_BF||).",
 "Euler":"Original FP32 state + velocity.float()*(1/50), C copied; isolated step is not an OFF rollout.",
 "aggregation":"Equal cells within label/data_panel/time; layers additionally separated. Missing cells never silently reduce denominators.",
 "precision_comparison":"Whole FP32 path can change h and r; its layer statistics are not a same-h/r local intervention.",
 "B_C_comparison":"At t>0 B/C use their own ON trajectory states; no same-state architecture causal comparison.",
}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");os.replace(tmp,p)
def bitwise(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def norm(x):return float(np.sqrt(np.sum(np.asarray(x,dtype=np.float64)**2,dtype=np.float64)))
def ratio(row,key,numerator,denominator):
 row[key]=float(numerator/denominator) if denominator>0 else None
 row[key+"_status"]="DEFINED" if denominator>0 else "UNDEFINED"
def distribution(prefix,x):
 x=np.asarray(x,dtype=np.float64).reshape(-1);x=x[np.isfinite(x)]
 out={prefix+"_valid_count":len(x),prefix+"_mean":float(np.mean(x)) if len(x) else None}
 for q in QUANTILES:out[f"{prefix}_q{q:02d}"]=float(np.quantile(x,q/100)) if len(x) else None
 return out

def layer_numbers(before,residual,cast,after):
 """Return JSON-safe scalar metrics; inputs are finite CPU arrays [corners,D].
 Actual arithmetic producing the inputs is never rerun. FP64 reductions retain
 the previous diagnostic definitions and add error/projection/explicit statuses.
 """
 original=[np.asarray(x) for x in (before,residual,cast,after)]
 if not original or original[0].ndim!=2 or 0 in original[0].shape:
  raise ValueError("Expected nonempty [free_corners,hidden_dim] CPU arrays")
 if any(x.shape!=original[0].shape or x.dtype.kind!="f" or not np.isfinite(x).all() for x in original):
  raise ValueError("All captures must be finite, floating and identically shaped")
 b,r,c,a=[x.astype(np.float64,copy=False) for x in original];e=a-b
 values=dict(before=b,residual=r,cast=c,after=a,effective=e,cast_error=c-r,addition_error=e-c,effective_error=e-r)
 out=dict(element_count=int(r.size),corner_count=int(r.shape[0]),hidden_width=int(r.shape[1]),reduction_dtype="float64")
 for name,v in values.items():
  out[name+"_l2"]=norm(v);out[name+"_rms"]=float(np.sqrt(np.mean(v*v)));out[name+"_max_abs"]=float(np.max(np.abs(v)))
 rn,cn,en=out["residual_l2"],out["cast_l2"],out["effective_l2"]
 for key,num,den in [
  ("cast_error_relative_l2",out["cast_error_l2"],rn),
  ("addition_error_relative_l2_cast",out["addition_error_l2"],cn),
  ("effective_error_relative_l2",out["effective_error_l2"],rn),
  ("effective_over_residual_l2",en,rn),("cast_over_residual_l2",cn,rn),("effective_over_cast_l2",en,cn),
  ("effective_residual_cosine",float(np.sum(e*r,dtype=np.float64)),en*rn),
  ("cast_residual_cosine",float(np.sum(c*r,dtype=np.float64)),cn*rn),
  ("effective_residual_projection",float(np.sum(e*r,dtype=np.float64)),rn*rn),
  ("residual_to_hidden_l2",rn,out["before_l2"]),("effective_to_hidden_l2",en,out["before_l2"])]:
  ratio(out,key,num,den)
 nz=int(np.count_nonzero(r!=0));sw=int(np.count_nonzero((r!=0)&(e==0)));cz=int(np.count_nonzero((r!=0)&(c==0)))
 out.update(residual_nonzero_elements=nz,residual_nonzero_swallowed_elements=sw,
  cast_zero_elements_nonzero_residual=cz,changed_element_count=int(np.count_nonzero(a!=b)),
  changed_element_fraction=float(np.mean(a!=b)),changed_corner_count=int(np.count_nonzero(np.any(e!=0,axis=1))),
  changed_corner_fraction=float(np.mean(np.any(e!=0,axis=1))))
 ratio(out,"residual_nonzero_swallowed_fraction",sw,nz);ratio(out,"cast_zero_fraction_nonzero_residual",cz,nz)
 cnorms={}
 for name in ("residual","cast","effective","cast_error","addition_error","effective_error"):
  cnorms[name]=np.sqrt(np.sum(values[name]**2,axis=1,dtype=np.float64))
  out.update(distribution("corner_"+name+"_l2",cnorms[name]))
 good=cnorms["residual"]>0
 out.update(distribution("corner_effective_over_residual",cnorms["effective"][good]/cnorms["residual"][good]))
 out["corner_ratio_zero_denominator_count"]=int(np.count_nonzero(~good))
 ratio(out,"corner_nonzero_residual_swallowed_fraction",int(np.count_nonzero(good&(cnorms["effective"]==0))),int(np.count_nonzero(good)))
 return out

def vector_numbers(name,x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=2 or x.shape[1]!=9 or not len(x) or not np.isfinite(x).all():raise ValueError("Expected free [F,9]")
 return {name+"_l2":norm(x),name+"_rms":float(np.sqrt(np.mean(x*x))),name+"_max_abs":float(np.max(np.abs(x))),
  name+"_changed_scalar_fraction":float(np.mean(x!=0)),
  name+"_changed_corner_fraction":float(np.mean(np.any(x.reshape(-1,3)!=0,axis=1))),
  name+"_changed_face_fraction":float(np.mean(np.any(x!=0,axis=1)))}
def four_grid(bon,boff,fon,foff):
 """Four free-coordinate velocity arrays; outputs metrics and exact FP64 differences."""
 a,b,c,d=[np.asarray(x,dtype=np.float64) for x in (bon,boff,fon,foff)]
 if any(x.shape!=a.shape for x in (b,c,d)):raise ValueError("Four-grid shape mismatch")
 vv=dict(D_BF=a-b,D_FP=c-d,P=d-b,J=(c-d)-(a-b),precision_ON=c-a)
 out={}
 for key,x in dict(v_BF_ON=a,v_BF_OFF=b,v_FP_ON=c,v_FP_OFF=d,**vv).items():out.update(vector_numbers(key,x))
 nr,nf,nj=out["D_BF_l2"],out["D_FP_l2"],out["J_l2"]
 for key,num,den in [
  ("D_FP_D_BF_cosine",float(np.sum(vv["D_FP"]*vv["D_BF"])),nf*nr),
  ("D_relative_difference_to_BF",nj,nr),("D_symmetric_relative_difference",2*nj,nr+nf),
  ("D_FP_over_D_BF_l2",nf,nr),("D_BF_relative_to_own_ON",nr,out["v_BF_ON_l2"]),
  ("D_FP_relative_to_own_ON",nf,out["v_FP_ON_l2"]),
  ("P_relative_to_BF_OFF",out["P_l2"],out["v_BF_OFF_l2"]),
  ("J_projection_onto_D_BF",float(np.sum(vv["J"]*vv["D_BF"])),nr*nr)]:
  ratio(out,key,num,den)
 error=vv["precision_ON"]-(vv["P"]+vv["J"])
 bound=16*np.finfo(np.float64).eps*(np.abs(c)+np.abs(a)+np.abs(vv["P"])+np.abs(vv["J"]))+np.nextafter(0.,1.)
 out.update(algebra_exact=bool(np.all(error==0)),algebra_roundoff_pass=bool(np.all(np.abs(error)<=bound)),
  algebra_error_l2=norm(error),algebra_error_max_abs=float(np.max(np.abs(error))),
  algebra_max_pointwise_bound=float(np.max(bound)))
 return out,vv

def cell_key(r):return (r["label"],r["request_id"],int(r["step"]),r["mode"])
def index_cells(folder):
 found={};duplicates=[]
 for p in sorted(Path(folder).glob("*.json")):
  r=read(p)
  if not all(k in r for k in ("label","request_id","step","mode")):raise ValueError("Non-cell JSON in cells: "+str(p))
  key=cell_key(r)
  if key in found:duplicates.append(key)
  found[key]=(p,r)
 if duplicates:raise ValueError("Duplicate fixed cells: "+repr(duplicates))
 return found
def resolve(p,jp):
 p=Path(p)
 return p if p.is_absolute() else jp.parent/p
def required_array(a,shape,name):
 if a.dtype!=np.float32 or a.shape!=shape or not np.isfinite(a).all():raise ValueError(name+": expected finite FP32 "+str(shape))
def write_csv(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rr:w.writerow({k:("UNDEFINED" if v is None and r.get(k+"_status")=="UNDEFINED" else v) for k,v in r.items()})
def grouped(rr,fields,metrics):
 groups=collections.defaultdict(list)
 for r in rr:groups[tuple(r.get(k) for k in fields)].append(r)
 out=[]
 for values,gg in sorted(groups.items(),key=lambda q:str(q[0])):
  for metric in metrics:
   nums=[r[metric] for r in gg if r["status"]=="COMPLETE" and isinstance(r.get(metric),(int,float)) and not isinstance(r.get(metric),bool) and math.isfinite(r[metric])]
   out.append(dict(zip(fields,values),metric=metric,expected_cells=len(gg),finite_cells=len(nums),
    mean=float(math.fsum(nums)/len(nums)) if len(nums)==len(gg) else None,
    observed_mean=float(math.fsum(nums)/len(nums)) if nums else None,
    minimum=min(nums) if nums else None,maximum=max(nums) if nums else None,
    status="COMPLETE" if len(nums)==len(gg) else "UNDEFINED_OR_INCOMPLETE"))
 return out

def run(root=DEFAULT_ROOT,legacy_root=DEFAULT_OLD):
 root=Path(root).resolve();old=Path(legacy_root).resolve();out=root/"analysis";out.mkdir(parents=True,exist_ok=True)
 tick=time.perf_counter();plan=read(old/"plan.json")
 if plan["labels"]!=["B1000","C1000"] or plan["steps"]!=[0,25,45] or len(plan["requests"])!=4:raise ValueError("Legacy fixed plan changed")
 oldsummary=read(old/"analysis/summary.json")
 if oldsummary["status"]!="COMPLETE" or digest(old/"plan.json")!=oldsummary["plan_sha256"]:raise ValueError("Unverified old summary/plan")
 expected={(l,q["request_id"],s,m) for l,q,s,m in itertools.product(plan["labels"],plan["requests"],plan["steps"],("on","off"))}
 indexes={"LEGACY_BF16":index_cells(old/"cells"),"FP32_REFERENCE":index_cells(root/"fixed/cells")}
 if any(set(v)-expected for v in indexes.values()):raise ValueError("Unexpected fixed-state cell")
 requests={q["request_id"]:q for q in plan["requests"]};streams={(r["label"],r["request_id"]):r for r in plan["streams"]}
 representative={next(q["request_id"] for q in plan["requests"] if q["panel"]==panel) for panel in ("P1","P2")}
 records=[];layers=[];local=[];pred={};source_hashes={str(old/"plan.json"):digest(old/"plan.json"),str(old/"analysis/summary.json"):digest(old/"analysis/summary.json"),str(Path(__file__)):digest(__file__)}
 if (root/"plan.json").exists():source_hashes[str(root/"plan.json")]=digest(root/"plan.json")
 retained_layers=0
 for precision in PRECISIONS:
  for key in sorted(expected):
   label,rid,step,mode=key;q=requests[rid];n,k=q["N"],q["K"]
   base=dict(label=label,request_id=rid,panel=q["panel"],data_panel=q["panel"],N=n,K=k,step=step,t=step/50.,mode=mode,precision=precision)
   audit=dict(base,status="MISSING",error=None)
   lr=[dict(base,layer=l,status="MISSING",error=None) for l in LAYERS] if mode=="on" else []
   try:
    if key not in indexes[precision]:raise FileNotFoundError("Fixed cell not present")
    jp,receipt=indexes[precision][key];ap=resolve(receipt["array_path"],jp)
    if receipt["status"]!="COMPLETE" or receipt["N"]!=n or receipt["K"]!=k or receipt["t"]!=step/50:raise ValueError("Receipt identity/status mismatch")
    if receipt["checkpoint_sha256"]!=plan["checkpoints"][label]["sha256"]:raise ValueError("Checkpoint identity mismatch")
    if precision=="FP32_REFERENCE" and receipt.get("precision")!="FP32_REFERENCE":raise ValueError("New cell not declared FP32_REFERENCE")
    if precision=="LEGACY_BF16" and mode=="on" and receipt.get("trajectory_replay_exact") is not True:raise ValueError("Old ON replay unverified")
    h=digest(ap)
    if h!=receipt["array_sha256"]:raise ValueError("NPZ hash mismatch")
    source_hashes[str(jp)]=digest(jp);source_hashes[str(ap)]=h
    audit.update(receipt_path=str(jp),receipt_sha256=source_hashes[str(jp)],array_path=str(ap),array_sha256=h)
    with np.load(ap,allow_pickle=False) as z:
     v,x,nxt=[z[name].copy() for name in ("velocity","state","next_state")]
     for name,arr in (("velocity",v),("state",x),("next_state",nxt)):required_array(arr,(1,n,9),name)
     if ah(x)!=streams[label,rid]["state_hashes"][str(step)]:raise ValueError("Not the original common input state")
     calc=x.copy();calc[:,k:]=x[:,k:]+v[:,k:]*(1/50)
     if not bitwise(calc,nxt):raise ValueError("Isolated FP32 Euler expression mismatch")
     if precision=="LEGACY_BF16" and mode=="on" and ah(nxt)!=streams[label,rid]["next_state_hashes"][str(step)]:raise ValueError("Actual old ON replay hash mismatch")
     pred[precision,key]=dict(velocity=v,state=x,next_state=nxt,audit=audit)
     for row in lr:
      layer=row["layer"];names=[f"l{layer}_{name}" for name in ("before","residual","cast","after")]
      present=all(name in z.files for name in names)
      row.update(receipt_path=str(jp),receipt_sha256=source_hashes[str(jp)],array_path=str(ap),array_sha256=h,capture_arrays_saved=present)
      if precision=="LEGACY_BF16":
       if not present:raise ValueError("Old full layer capture missing")
       arrays=[z[name] for name in names]
       for arr in arrays:required_array(arr,(3*(n-k),768),"layer capture")
       metrics=layer_numbers(*arrays)
       dtype=next(c for c in receipt["layer_checks"] if c["layer"]==layer)
       if tuple(dtype[x] for x in ("hidden_dtype","residual_dtype","output_dtype"))!=("torch.bfloat16","torch.float32","torch.bfloat16"):raise ValueError("Legacy layer dtype differs")
       b,r=arrays[:2];local_after=np.add(b.astype(np.float32),r.astype(np.float32),dtype=np.float32)
       local.append(dict(base,layer=layer,status="COMPLETE",comparison="SAME_BF16_PATH_H_R_LOCAL_FP32_ADD_ONLY",source_array_path=str(ap),source_array_sha256=h,**layer_numbers(b,r,r,local_after)))
      else:
       metrics_list=receipt.get("layer_metrics",[])
       if len(metrics_list)!=4 or {c["layer"] for c in metrics_list}!=set(LAYERS):raise ValueError("Need four immediate FP32 layer metric records")
       dtype=next(c for c in metrics_list if c["layer"]==layer)
       metrics=dtype.get("metrics",dtype).copy()
       for name in ("element_count","corner_count","effective_error_relative_l2","effective_residual_projection","effective_over_residual_l2","changed_corner_fraction"):
        if name not in metrics:raise ValueError("Missing immediate layer metric: "+name)
       if metrics["corner_count"]!=3*(n-k) or metrics["element_count"]!=3*(n-k)*768:raise ValueError("FP32 metric domain differs")
       if tuple(dtype.get(x) for x in ("hidden_dtype","residual_dtype","output_dtype"))!=("torch.float32",)*3:raise ValueError("FP32 layer dtype not verified")
       should_save=rid in representative and step==25
       if present!=should_save:raise ValueError("Full capture storage does not match representative registration")
       if present:
        arrays=[z[name] for name in names]
        for arr in arrays:required_array(arr,(3*(n-k),768),"representative capture")
        re=layer_numbers(*arrays)
        if any(metrics.get(name)!=value for name,value in re.items()):raise ValueError("Immediate/recomputed FP32 metrics differ")
        retained_layers+=1
      row.update({kk:vv for kk,vv in metrics.items() if kk not in ("layer","status","error")})
      row.update(hidden_dtype=dtype["hidden_dtype"],residual_dtype=dtype["residual_dtype"],output_dtype=dtype["output_dtype"],status="COMPLETE",error=None)
    audit.update(status="COMPLETE",error=None,original_state_hash=ah(x),euler_expression_exact=True,known_next_exact=bitwise(x[:,:k],nxt[:,:k]))
   except Exception as ex:
    audit.update(status="MISSING" if isinstance(ex,FileNotFoundError) else "FAILED",error=type(ex).__name__+": "+str(ex))
    pred.pop((precision,key),None)
    for row in lr:row.update(status=audit["status"],error=audit["error"])
   records.append(audit);layers.extend(lr)
 pairs=[];interactions=[];(out/"fixed_vectors").mkdir(exist_ok=True)
 for label,q,step in itertools.product(plan["labels"],plan["requests"],plan["steps"]):
  rid=q["request_id"];k=q["K"];base=dict(label=label,request_id=rid,panel=q["panel"],data_panel=q["panel"],N=q["N"],K=k,step=step,t=step/50.)
  pair=dict(base,status="INCOMPLETE",error=None);inter=dict(pair)
  try:
   aa=[pred[p,(label,rid,step,m)] for p,m in (("LEGACY_BF16","on"),("LEGACY_BF16","off"),("FP32_REFERENCE","on"),("FP32_REFERENCE","off"))]
   if not all(bitwise(aa[0]["state"],a["state"]) for a in aa[1:]):raise ValueError("Four cells do not share exact input state")
   for name,a in zip(("BF_ON","BF_OFF","FP_ON","FP_OFF"),aa):
    pair[name+"_array_path"]=a["audit"]["array_path"];pair[name+"_array_sha256"]=a["audit"]["array_sha256"]
    pair[name+"_receipt_sha256"]=a["audit"]["receipt_sha256"]
   metrics,vectors=four_grid(*[a["velocity"][0,k:] for a in aa])
   if not metrics["algebra_roundoff_pass"]:raise ValueError("Four-grid algebra failed")
   nextmetrics,_=four_grid(*[a["next_state"][0,k:] for a in aa])
   pair.update(metrics,**{"isolated_next_"+name:value for name,value in nextmetrics.items()},state_four_grid_bitwise=True,input_state_sha256=ah(aa[0]["state"]),status="COMPLETE",error=None)
   dest=out/"fixed_vectors"/f"{label}__{rid}__s{step:02d}.npz";np.savez_compressed(dest,**vectors)
   inter.update(metrics,vector_path=str(dest),vector_sha256=digest(dest),status="COMPLETE",error=None)
  except Exception as ex:
   pair["error"]=inter["error"]=type(ex).__name__+": "+str(ex)
  pairs.append(pair);interactions.append(inter)
 assert len(records)==96 and len(layers)==192 and len(pairs)==len(interactions)==24
 files={
  "fixed_state_pairs.csv":pairs,"layer_precision_metrics.csv":layers,"prediction_precision_interactions.csv":interactions,
  "same_hr_local_fp32.csv":local,"fixed_cell_audit.csv":records,
  "layer_precision_group_summary.csv":grouped(layers,("precision","label","data_panel","step","layer"),("residual_nonzero_swallowed_fraction","effective_over_residual_l2","effective_residual_cosine","effective_error_relative_l2","effective_residual_projection","changed_corner_fraction")),
  "prediction_precision_group_summary.csv":grouped(interactions,("label","data_panel","step"),("D_BF_l2","D_FP_l2","P_l2","J_l2","D_FP_D_BF_cosine","D_relative_difference_to_BF","D_BF_relative_to_own_ON","D_FP_relative_to_own_ON"))}
 for name,rr in files.items():write_csv(out/name,rr)
 complete=[sum(r["status"]=="COMPLETE" for r in rr) for rr in (records,layers,pairs)]
 status="COMPLETE" if complete==[96,192,24] and retained_layers==16 and len(local)==96 else "INCOMPLETE"
 errors=[{k:r.get(k) for k in ("label","request_id","step","mode","precision","status","error")} for r in records+pairs if r["status"]!="COMPLETE"]
 summary=dict(schema=VERSION,status=status,complete_cells=complete[0],expected_cells=96,complete_layers=complete[1],expected_layers=192,
  complete_four_grid_units=complete[2],expected_four_grid_units=24,old_cached_on_layers=96,new_fp32_on_layers=96,
  retained_new_layer_arrays=retained_layers,local_same_hr_layers=len(local),preflight_records_included=0,
  model_calls=0,checkpoint_loads=0,new_rollouts=0,errors=errors,source_hashes=source_hashes,
  output_hashes={name:digest(out/name) for name in files},definitions=DEFINITIONS,wall_seconds=time.perf_counter()-tick,
  interpretation="Numerical diagnostics only. Whole-front precision affects hidden/residual/prediction; no mesh-quality or causal training conclusion.")
 write_json(out/"fixed_state_summary.json",summary);write_json(out/"fixed_metric_definitions.json",DEFINITIONS)
 print(json.dumps({k:summary[k] for k in ("status","complete_cells","complete_layers","complete_four_grid_units","retained_new_layer_arrays","local_same_hr_layers","wall_seconds")}))
 return summary

def self_test(root):
 z=np.zeros((3,8),np.float32);one=np.ones_like(z);small=np.full_like(z,.001)
 a=layer_numbers(one,small,small,one)
 assert a["residual_nonzero_swallowed_fraction"]==1 and a["effective_over_residual_l2"]==0 and a["effective_residual_cosine"] is None and a["effective_error_relative_l2"]==1
 zz=layer_numbers(z,z,z,z)
 assert zz["effective_over_residual_l2"] is None and zz["effective_over_residual_l2_status"]=="UNDEFINED" and zz["residual_nonzero_swallowed_fraction"] is None
 a=layer_numbers(one,small,small,one+np.float32(.002))
 assert a["effective_over_residual_l2"]>1 and a["effective_residual_projection"]>1
 a=layer_numbers(z,one,one,-one)
 assert np.allclose([a["effective_residual_cosine"],a["effective_residual_projection"],a["effective_error_relative_l2"]],[-1.,-1.,2.],rtol=1e-14,atol=0.)
 x=np.arange(18,dtype=np.float32).reshape(2,9)
 row,v=four_grid(x+2,x,x+5,x+1)
 assert row["algebra_roundoff_pass"] and np.array_equal(v["P"]+v["J"],v["precision_ON"]) and row["D_FP_over_D_BF_l2"]==2
 row,_=four_grid(np.zeros_like(x),np.zeros_like(x),np.zeros_like(x),np.zeros_like(x))
 assert row["D_FP_D_BF_cosine"] is None and row["D_relative_difference_to_BF_status"]=="UNDEFINED"
 local=np.add(one,small,dtype=np.float32);a=layer_numbers(one,small,small,local)
 assert a["effective_error_relative_l2"]<1e-3 and a["changed_corner_fraction"]==1
 mixed=z.copy();mixed[0,0]=1
 a=layer_numbers(z,mixed,mixed,mixed)
 assert a["changed_corner_fraction"]==1/3 and a["changed_element_fraction"]==1/24
 failed=False
 try:layer_numbers(z,one[:2],one,z)
 except ValueError:failed=True
 assert failed
 r=dict(status="PASS",test_groups=9,checks=["full_swallow","zero_UNDEFINED","unbounded_ratio","opposite_direction","four_grid_algebra","zero_prediction","same_hr_fp32","corner_denominator","shape_rejection"],model_calls=0,checkpoint_loads=0,script_sha256=digest(__file__))
 write_json(Path(root)/"analysis/fixed_cpu_tests.json",r);print(json.dumps(r));return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",default=DEFAULT_ROOT);p.add_argument("--legacy-root",default=DEFAULT_OLD);p.add_argument("--self-test",action="store_true")
 a=p.parse_args()
 if a.self_test:self_test(a.root);return 0
 return 0 if run(a.root,a.legacy_root)["status"]=="COMPLETE" else 2
if __name__=="__main__":raise SystemExit(main())
