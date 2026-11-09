"""Independent arithmetic from saved scalar records; no model/score/mesh operations."""
import csv,json,hashlib,math,statistics,collections,time,sys
from pathlib import Path
RUN=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
OUT=RUN/"audit/course_independent_review";OUT.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
sources={}
def readj(p):
 p=Path(p);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding="utf-8-sig"))
def readc(p):
 p=Path(p);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def writej(n,o):(OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
def writec(n,rows):
 with (OUT/n).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def avg(v):return statistics.fmean(v) if v and all(num(x) for x in v) else None
def sub(a,b):return a-b if num(a) and num(b) else None
def val(x):return None if x in ("",None) else float(x)
exceptions=[];counts=collections.Counter()
def check(kind,actual,expected,key):
 counts[kind]+=1
 if num(expected) and num(actual): ok=math.isclose(actual,expected,abs_tol=1e-12,rel_tol=1e-10)
 else:ok=actual==expected
 if not ok:exceptions.append(dict(kind=kind,key=str(key),actual=actual,expected=expected))
freeze=readj(RUN/"audit/visual_review/course/schedule40_train/initial_freeze_receipt.json")
assert freeze["status"]=="COMPLETE" and not freeze["course_mapping_read"] and not freeze["course_quality_metrics_read"]
ANA=Path(readj(RUN/"analysis/course/latest.json")["analysis_dir"])
main=readj(RUN/"phases/main/evaluation/per_output.json")["rows"]
course=readj(RUN/"phases/course/evaluation/per_output.json")["rows"]
labels=["R2_MIX_H_ALL","C20_40_MIX","C40_20_MIX"]
pairs=[(labels[1],labels[0]),(labels[2],labels[0]),(labels[1],labels[2])]
raw=[r for r in main+course if r["label"] in labels]
assert len(raw)==192
lookup={(r["label"],r["request_id"]):r for r in main+course};assert len(lookup)==512
cohort={r["parent_id"]:r["condition_cohort"] for r in raw if r["ratio"]==.4}
assert collections.Counter(cohort.values())=={"ORIGINAL6":6,"EXTENDED2":2}
requests={r["request_id"]:r for r in raw if r["label"]==labels[0]}
for r in raw:
 m=r["metrics"];check("I_union",m["ConflictCov_C"],m["ConflictCov_C_cross"]+m["ConflictCov_C_dup"]-m["ConflictCov_C_cross_and_dup"],(r["label"],r["request_id"]))
 check("score_status",r["status"],"PASS",(r["label"],r["request_id"]))
 for f in ("query_sha256","registered_C_queries","query_weight_denominator","registered_free_faces","N","K"):
  check("matched_metadata",r[f],requests[r["request_id"]][f],(r["label"],r["request_id"],f))
cat=readc(ANA/"metric_catalog.csv");directions={r["metric"]:r["direction"] for r in cat}
metrics=[r["metric"] for r in cat if r["numeric_summary"]=="True"]
parent_values=collections.defaultdict(list)
for r in raw:
 for m in metrics:parent_values[(r["label"],r["parent_id"],r["ratio"],m)].append(r["metrics"].get(m))
pmeans={k:avg(v) for k,v in parent_values.items()}
for r in readc(ANA/"paired_noise_metrics.csv"):
 a,b=lookup[(r["first"],r["request_id"])],lookup[(r["second"],r["request_id"])]
 av,bv=a["metrics"].get(r["metric"]),b["metrics"].get(r["metric"])
 for f,v in [("first_value",av),("second_value",bv),("delta",sub(av,bv))]:check("paired_noise_"+f,val(r[f]),v,(r["first"],r["second"],r["request_id"],r["metric"]))
for r in readc(ANA/"per_parent_metrics.csv"):
 v=pmeans[(r["label"],r["parent_id"],float(r["ratio"]),r["metric"])]
 check("parent_mean",val(r["mean"]),v,(r["label"],r["parent_id"],r["ratio"],r["metric"]))
for r in readc(ANA/"paired_parent_metrics.csv"):
 av=pmeans[(r["first"],r["parent_id"],float(r["ratio"]),r["metric"])]
 bv=pmeans[(r["second"],r["parent_id"],float(r["ratio"]),r["metric"])]
 for f,v in [("first_mean",av),("second_mean",bv),("delta",sub(av,bv))]:check("paired_parent_"+f,val(r[f]),v,(r["first"],r["second"],r["parent_id"],r["ratio"],r["metric"]))
for r in readc(ANA/"model_summary.csv"):
 pp=[p for p in cohort if r["parent_stratum"]=="ALL" or cohort[p]==r["parent_stratum"]]
 v=avg([pmeans[(r["label"],p,float(r["ratio"]),r["metric"])] for p in pp])
 check("model_summary_mean",val(r["mean"]),v,(r["label"],r["ratio"],r["metric"],r["parent_stratum"]))
for r in readc(ANA/"paired_parent_summary.csv"):
 pp=[p for p in cohort if r["parent_stratum"]=="ALL" or cohort[p]==r["parent_stratum"]]
 dd=[sub(pmeans[(r["first"],p,float(r["ratio"]),r["metric"])],pmeans[(r["second"],p,float(r["ratio"]),r["metric"])]) for p in pp]
 check("paired_summary_mean",val(r["mean"]),avg(dd),(r["first"],r["second"],r["ratio"],r["metric"],r["parent_stratum"]))
 if all(num(x) for x in dd):
  for f,v in [("negative_count",sum(x<0 for x in dd)),("positive_count",sum(x>0 for x in dd)),("exact_zero_count",sum(x==0 for x in dd))]:
   check("paired_summary_sign",int(r[f]),v,(r["first"],r["second"],r["ratio"],r["metric"],r["parent_stratum"],f))
  loo=[avg(dd[:i]+dd[i+1:]) for i in range(len(dd))]
  for f,v in [("leave_one_parent_out_min",min(loo)),("leave_one_parent_out_max",max(loo))]:
   check("paired_summary_LOO",val(r[f]),v,(r["first"],r["second"],r["ratio"],r["metric"],r["parent_stratum"],f))
# Independent diversity means from original input tables, then compare analysis copies/aggregates.
divpairs=readc(RUN/"phases/main/evaluation/diversity_pairs.csv")+readc(RUN/"phases/course/evaluation/diversity_pairs.csv")
divcond=readc(RUN/"phases/main/evaluation/diversity_per_condition.csv")+readc(RUN/"phases/course/evaluation/diversity_per_condition.csv")
dg=collections.defaultdict(list)
for r in divpairs:
 if r["label"] in labels:
  check("diversity_pair_status",r["status"],"PASS",(r["label"],r["condition_id"]))
  dg[(r["label"],r["condition_id"])].append(float(r["D_free_pair"]))
dv={}
for r in divcond:
 if r["label"] in labels:
  vals=dg[(r["label"],r["condition_id"])];check("diversity_pair_count",len(vals),6,(r["label"],r["condition_id"]))
  v=avg(vals);check("diversity_input_mean",float(r["D_free_mean"]),v,(r["label"],r["condition_id"]))
  dv[(r["label"],r["parent_id"],float(r["ratio"]))]=v
for r in readc(ANA/"diversity_per_condition.csv"):check("diversity_analyzer_mean",float(r["D_free_mean"]),dv[(r["label"],r["parent_id"],float(r["ratio"]))],(r["label"],r["condition_id"]))
ds=[]
for a,b in pairs:
 for ratio in [.2,.4]:
  for scope in ["ALL","ORIGINAL6","EXTENDED2"]:
   pp=[p for p in cohort if scope=="ALL" or cohort[p]==scope]
   av,bv=avg([dv[(a,p,ratio)] for p in pp]),avg([dv[(b,p,ratio)] for p in pp])
   ds.append(dict(first=a,second=b,ratio=ratio,scope=scope,first_D=av,second_D=bv,delta=av-bv,relative_change=av/bv-1,parents_decreased=sum(dv[(a,p,ratio)]<dv[(b,p,ratio)] for p in pp),parents_increased=sum(dv[(a,p,ratio)]>dv[(b,p,ratio)] for p in pp)))
for r in readc(ANA/"diversity_paired_summary.csv"):
 expected=next(x for x in ds if (x["first"],x["second"],x["ratio"],x["scope"])==(r["first"],r["second"],float(r["ratio"]),r["parent_stratum"]))
 check("diversity_summary",float(r["mean"]),expected["delta"],(r["first"],r["second"],r["ratio"],r["parent_stratum"]))
# Selected full component tables: no new geometric measurements.
selected=["CF_proper_affected_free_fraction","CF_proper_pair_count","CF_coplanar_pair_count","CF_unresolved_or_degenerate_pair_count","FF_illegal_affected_free_fraction","FF_illegal_pair_count","FF_proper_affected_free_fraction","FF_proper_pair_count","FF_coplanar_pair_count","FF_unresolved_or_degenerate_pair_count","ConflictCov_C","ConflictCov_C_upper","ConflictCov_C_cross","ConflictCov_C_dup","ConflictCov_C_cross_and_dup","ConflictCov_C_uncertain_fraction","ConflictCov_C_coplanar_illegal_pairs","ConflictCov_C_segment_extraction_failures","ConflictCov_C_unresolved_pairs","free_minimum_angle_lt5_fraction","free_angle_p05_degrees","free_q_gt10_fraction","free_exact_duplicate_excess_fraction","free_exact_zero_area_fraction","free_near_degenerate_fraction","whole_minimum_angle_lt5_fraction"]
allpairs=pairs+[(c,b) for c in labels[1:] for b in ["START5000","R1_40_H_ALL","HIST_FM6000"]]
noise=[]
for a,b in allpairs:
 for rid,q in requests.items():
  av,bv=lookup[(a,rid)],lookup[(b,rid)]
  for m in selected:
   va,vb=av["metrics"][m],bv["metrics"][m]
   assert num(va) and num(vb)
   noise.append(dict(first=a,second=b,request_id=rid,parent_id=q["parent_id"],ratio=q["ratio"],cohort=cohort[q["parent_id"]],noise=q["candidate"],metric=m,first_value=va,second_value=vb,delta=va-vb))
ng=collections.defaultdict(list)
for r in noise:ng[(r["first"],r["second"],r["parent_id"],r["ratio"],r["metric"])].append(r)
pr=[]
for (a,b,p,ratio,m),rr in ng.items():
 assert sorted(x["noise"] for x in rr)==[0,1,6,7]
 pr.append(dict(first=a,second=b,parent_id=p,ratio=ratio,cohort=cohort[p],metric=m,first_mean=avg([x["first_value"] for x in rr]),second_mean=avg([x["second_value"] for x in rr]),delta=avg([x["delta"] for x in rr])))
ss=[]
for a,b in allpairs:
 for ratio in [.2,.4]:
  for scope in ["ALL","ORIGINAL6","EXTENDED2"]:
   for m in selected:
    pp=[r for r in pr if (r["first"],r["second"],r["ratio"],r["metric"])==(a,b,ratio,m) and (scope=="ALL" or r["cohort"]==scope)]
    nn=[r for r in noise if (r["first"],r["second"],r["ratio"],r["metric"])==(a,b,ratio,m) and (scope=="ALL" or r["cohort"]==scope)]
    dd=[r["delta"] for r in pp];d=directions[m];lower=d.startswith("lower");defined=lower or d=="higher"
    gains=sorted([max(0,(-1 if lower else 1)*x) for x in dd],reverse=True) if defined else []
    total=sum(gains);loo=[avg(dd[:i]+dd[i+1:]) for i in range(len(dd))]
    ss.append(dict(first=a,second=b,ratio=ratio,scope=scope,metric=m,direction=d,parent_count=len(pp),first_mean=avg([r["first_mean"] for r in pp]),second_mean=avg([r["second_mean"] for r in pp]),delta=avg(dd),parent_better=sum(x<0 if lower else x>0 for x in dd) if defined else None,parent_worse=sum(x>0 if lower else x<0 for x in dd) if defined else None,parent_tie=sum(x==0 for x in dd),top2_gross_gain_share=sum(gains[:2])/total if total else None,loo_min=min(loo),loo_max=max(loo),worst_parent=max(pp,key=lambda r:r["delta"] if lower else -r["delta"])["parent_id"],worst_parent_delta=max(dd) if lower else min(dd),worst_noise=max(nn,key=lambda r:r["delta"] if lower else -r["delta"])["request_id"],worst_noise_delta=max(r["delta"] for r in nn) if lower else min(r["delta"] for r in nn)))
writec("per_noise.csv",noise);writec("per_parent.csv",pr);writec("summary.csv",ss);writec("diversity_summary.csv",ds)
sources[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt=dict(status="PASS" if not exceptions else "FAIL",counts=dict(counts),primary_rows=192,full_numeric_metrics=len(metrics),course_pairs=pairs,exceptions=exceptions,sources=sources,wall_seconds=time.perf_counter()-start,torch_imported="torch" in sys.modules,scientific_calls=0,limits=["Scalar arithmetic only; detector correctness not revalidated.","One training realization; no universal superiority or causal ratio-only claim.","Diversity is variation without preferred direction."])
writej("audit.json",receipt)
core=["CF_proper_affected_free_fraction","FF_illegal_affected_free_fraction","ConflictCov_C","free_minimum_angle_lt5_fraction"]
writej("core_summaries.json",[x for x in ss if x["metric"] in core])
print(json.dumps(receipt,indent=2))
print("CORE_COURSES",json.dumps([dict(first=x["first"],second=x["second"],ratio=x["ratio"],scope=x["scope"],metric=x["metric"],delta_pp=100*x["delta"],better=x["parent_better"],worse=x["parent_worse"],top2=x["top2_gross_gain_share"],loo_pp=[100*x["loo_min"],100*x["loo_max"]]) for x in ss if x["second"]=="R2_MIX_H_ALL" and x["metric"] in core],indent=2))
print("DIVERSITY",json.dumps(ds,indent=2))
