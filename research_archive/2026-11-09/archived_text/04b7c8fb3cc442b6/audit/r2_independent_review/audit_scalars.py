"""Read only saved scalar tables; no scoring, models or geometry calls."""
import csv, json, hashlib, math, statistics, collections, time
from pathlib import Path
RUN=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
OUT=RUN/"audit/r2_independent_review"
ANA=RUN/"analysis/main/recipe_factorial_table_analysis_v1_05c9596fa2afcb18"
EVAL=RUN/"phases/main/evaluation"
R2="R2_MIX_H_ALL"
BASES=["R1_40_H_ALL","START5000"]
METRICS=["CF_proper_affected_free_fraction","CF_proper_pair_count","CF_coplanar_pair_count","CF_unresolved_or_degenerate_pair_count","FF_illegal_affected_free_fraction","FF_illegal_pair_count","FF_proper_affected_free_fraction","FF_proper_pair_count","FF_coplanar_pair_count","FF_unresolved_or_degenerate_pair_count","ConflictCov_C","ConflictCov_C_upper","ConflictCov_C_cross","ConflictCov_C_dup","ConflictCov_C_cross_and_dup","ConflictCov_C_uncertain_fraction","ConflictCov_C_coplanar_illegal_pairs","ConflictCov_C_segment_extraction_failures","ConflictCov_C_unresolved_pairs","free_minimum_angle_lt5_fraction","free_angle_p05_degrees","free_q_gt10_fraction","free_exact_duplicate_excess_fraction","free_exact_zero_area_fraction","free_near_degenerate_fraction","whole_minimum_angle_lt5_fraction"]
CORE=["CF_proper_affected_free_fraction","FF_illegal_affected_free_fraction","ConflictCov_C","free_minimum_angle_lt5_fraction"]
PARTS=["ConflictCov_C_cross","ConflictCov_C_dup","ConflictCov_C_cross_and_dup"]
start=time.perf_counter()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def readc(p):
    with p.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def writej(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def writec(name,rows):
    with (OUT/name).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["EMPTY"]);w.writeheader();w.writerows(rows)
def mean(x): return statistics.mean(x)
def close(a,b): return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
def value_scaled(metric,v): return 100*v if ("fraction" in metric or metric.startswith("ConflictCov_C") and metric in ["ConflictCov_C","ConflictCov_C_upper",*PARTS,"ConflictCov_C_uncertain_fraction"]) else v
OUT.mkdir(parents=True,exist_ok=True)
raw=readj(EVAL/"per_output.json")["rows"]
assert len(raw)==384
lookup={(r["label"],r["request_id"]):r for r in raw}
assert len(lookup)==384
cohort={r["parent_id"]:r["condition_cohort"] for r in raw if r["ratio"]==.4}
assert collections.Counter(cohort.values())=={"ORIGINAL6":6,"EXTENDED2":2}
requests={r["request_id"]:r for r in raw if r["label"]==R2}
assert len(requests)==64
exceptions=[]
for r in raw:
    m=r["metrics"]
    if not close(m["ConflictCov_C"],m["ConflictCov_C_cross"]+m["ConflictCov_C_dup"]-m["ConflictCov_C_cross_and_dup"]): exceptions.append(["I_union",r["label"],r["request_id"]])
    if r["status"]!="PASS": exceptions.append(["score_status",r["label"],r["request_id"],r["status"]])
noise=[]
for b in BASES:
    for rid,r in requests.items():
        q=lookup[(b,rid)]
        for key in ["N","K","ratio","parent_id","query_sha256","registered_C_queries","query_weight_denominator","registered_free_faces"]:
            if r[key]!=q[key]: exceptions.append(["binding",b,rid,key])
        for metric in METRICS:
            v,qv=r["metrics"][metric],q["metrics"][metric]
            assert isinstance(v,(int,float)) and math.isfinite(v) and isinstance(qv,(int,float)) and math.isfinite(qv)
            noise.append(dict(candidate=R2,baseline=b,request_id=rid,parent_id=r["parent_id"],ratio=r["ratio"],cohort=cohort[r["parent_id"]],noise=r["candidate"],metric=metric,candidate_value=v,baseline_value=qv,delta=v-qv))
assert len(noise)==2*64*len(METRICS)
groups=collections.defaultdict(list)
for r in noise: groups[(r["baseline"],r["parent_id"],r["ratio"],r["metric"])].append(r)
parents=[]
for (b,p,ratio,metric),rr in groups.items():
    assert sorted(r["noise"] for r in rr)==[0,1,6,7]
    dd=[r["delta"] for r in rr]
    parents.append(dict(candidate=R2,baseline=b,parent_id=p,ratio=ratio,cohort=cohort[p],metric=metric,candidate_mean=mean(r["candidate_value"] for r in rr),baseline_mean=mean(r["baseline_value"] for r in rr),delta=mean(dd),noise_improved=sum(x<0 if metric!="free_angle_p05_degrees" else x>0 for x in dd),noise_worse=sum(x>0 if metric!="free_angle_p05_degrees" else x<0 for x in dd),noise_tied=sum(x==0 for x in dd),worst_noise_delta=max(dd),best_noise_delta=min(dd)))
# Verify independent arithmetic against the previously completed analysis tables, with their first-minus-second convention.
existing=readc(ANA/"paired_noise_metrics.csv")
existing_index={(r["first"],r["second"],r["request_id"],r["metric"]):float(r["delta"]) for r in existing}
checks=0
for r in noise:
    key=(R2,r["baseline"],r["request_id"],r["metric"])
    reverse=(r["baseline"],R2,r["request_id"],r["metric"])
    expected=existing_index[key] if key in existing_index else -existing_index[reverse]
    if not close(r["delta"],expected):exceptions.append(["prior_noise_arithmetic",key,r["delta"],expected])
    checks+=1
existingp=readc(ANA/"paired_parent_metrics.csv")
pindex={(r["first"],r["second"],r["parent_id"],float(r["ratio"]),r["metric"]):float(r["delta"]) for r in existingp}
for r in parents:
    key=(R2,r["baseline"],r["parent_id"],r["ratio"],r["metric"])
    rev=(r["baseline"],R2,r["parent_id"],r["ratio"],r["metric"])
    expected=pindex[key] if key in pindex else -pindex[rev]
    if not close(r["delta"],expected):exceptions.append(["prior_parent_arithmetic",key])
    checks+=1
summary=[]
for b in BASES:
 for ratio in [.2,.4]:
  for scope in ["ALL","ORIGINAL6","EXTENDED2"]:
   for metric in METRICS:
    pp=[r for r in parents if r["baseline"]==b and r["ratio"]==ratio and r["metric"]==metric and (scope=="ALL" or r["cohort"]==scope)]
    nn=[r for r in noise if r["baseline"]==b and r["ratio"]==ratio and r["metric"]==metric and (scope=="ALL" or r["cohort"]==scope)]
    dd=[r["delta"] for r in pp]
    lower=metric!="free_angle_p05_degrees"
    benefits=sorted([max(0,(-1 if lower else 1)*v) for v in dd],reverse=True)
    sumbenefit=sum(benefits)
    losses=sorted([max(0,(1 if lower else -1)*v) for v in dd],reverse=True)
    loo=[mean(dd[:i]+dd[i+1:]) for i in range(len(dd))]
    summary.append(dict(baseline=b,ratio=ratio,scope=scope,metric=metric,parent_count=len(pp),noise_count=len(nn),candidate_mean=mean(r["candidate_mean"] for r in pp),baseline_mean=mean(r["baseline_mean"] for r in pp),delta=mean(dd),parent_improved=sum(v<0 if lower else v>0 for v in dd),parent_worse=sum(v>0 if lower else v<0 for v in dd),parent_tied=sum(v==0 for v in dd),noise_improved=sum(r["delta"]<0 if lower else r["delta"]>0 for r in nn),noise_worse=sum(r["delta"]>0 if lower else r["delta"]<0 for r in nn),noise_tied=sum(r["delta"]==0 for r in nn),top1_share_of_gross_parent_improvement=benefits[0]/sumbenefit if sumbenefit else None,top2_share_of_gross_parent_improvement=sum(benefits[:2])/sumbenefit if sumbenefit else None,leave_one_parent_out_delta_min=min(loo),leave_one_parent_out_delta_max=max(loo),best_parent=min(pp,key=lambda r:r["delta"])["parent_id"],best_parent_delta=min(dd),worst_parent=max(pp,key=lambda r:r["delta"])["parent_id"],worst_parent_delta=max(dd),worst_noise_request=max(nn,key=lambda r:r["delta"])["request_id"],worst_noise_delta=max(r["delta"] for r in nn)))
# Diversity: recompute six pair means; variation has no improvement direction.
div=readc(EVAL/"diversity_per_condition.csv"); pairs=readc(EVAL/"diversity_pairs.csv")
pairgroups=collections.defaultdict(list)
for r in pairs:
    assert r["status"]=="PASS"
    pairgroups[(r["label"],r["condition_id"])].append(float(r["D_free_pair"]))
dindex={}
for r in div:
    vals=pairgroups[(r["label"],r["condition_id"])]
    assert len(vals)==6
    observed=mean(vals)
    if not close(observed,float(r["D_free_mean"])):exceptions.append(["diversity_mean",r["label"],r["condition_id"]])
    dindex[(r["label"],r["parent_id"],float(r["ratio"]))]=observed
dparents=[]
for b in BASES:
 for ratio in [.2,.4]:
  for p,c in cohort.items():
   cv,bv=dindex[(R2,p,ratio)],dindex[(b,p,ratio)]
   dparents.append(dict(baseline=b,ratio=ratio,parent_id=p,cohort=c,candidate_D_free=cv,baseline_D_free=bv,delta=cv-bv,relative_change=(cv/bv-1) if bv else None))
dsummary=[]
for b in BASES:
 for ratio in [.2,.4]:
  for scope in ["ALL","ORIGINAL6","EXTENDED2"]:
   rr=[r for r in dparents if r["baseline"]==b and r["ratio"]==ratio and (scope=="ALL" or r["cohort"]==scope)]
   cv,bv=mean(r["candidate_D_free"] for r in rr),mean(r["baseline_D_free"] for r in rr)
   dsummary.append(dict(baseline=b,ratio=ratio,scope=scope,parent_count=len(rr),candidate_D_free=cv,baseline_D_free=bv,delta=cv-bv,aggregate_relative_change=cv/bv-1,parent_reduced=sum(r["delta"]<0 for r in rr),parent_increased=sum(r["delta"]>0 for r in rr)))
writec("per_noise.csv",noise);writec("per_parent.csv",parents);writec("summary.csv",summary);writec("diversity_per_parent.csv",dparents);writec("diversity_summary.csv",dsummary)
sources=[EVAL/"per_output.json",EVAL/"diversity_per_condition.csv",EVAL/"diversity_pairs.csv",ANA/"paired_noise_metrics.csv",ANA/"paired_parent_metrics.csv",Path(__file__),RUN/"audit/visual_review/schedule40_train/observations.json",RUN/"audit/visual_review/schedule40_train/observations.csv"]
receipt=dict(status="PASS" if not exceptions else "FAIL",operation="Saved scalar table arithmetic only; no model, score, geometry, render or sampling calls",primary_rows=len(raw),compared_noise_metric_rows=len(noise),compared_parent_metric_rows=len(parents),prior_analysis_arithmetic_checks=checks,I_union_checks=len(raw),diversity_pair_means_checked=len(div),cohorts=cohort,exceptions=exceptions,sources={str(p):sha(p) for p in sources},wall_seconds=time.perf_counter()-start,notes=["Delta is R2 minus baseline; fractions may be shown as percentage points.","Original6/Extended2 for A20 are parent membership strata derived from the C40 registration, not separate A20 sampling cohorts.","Single training realization and observed DEV parents; no statistical significance or promotion decision.","MIX changes historical task-family/region distribution as well as ratios; not a pure ratio effect.","Diversity is variation, not larger-is-better."])
writej("audit.json",receipt)
lines=["# R2 independent scalar-table audit","","No rescoring or model calls. Anonymous notes were saved before reading these tables. Delta = R2 minus comparator. Percentages below are percentage-point differences. Four noises are averaged per parent, then parents receive equal weight.","","## Mean costs and distribution","","| Comparator | C | Parent stratum | Metric | Delta pp | Parent better/worse/tie | Noise better/worse/tie | Top-two share of gross gain | LOO mean delta range pp |","|---|---|---|---|---:|---|---|---:|---|"]
short={CORE[0]:"CF proper",CORE[1]:"FF illegal",CORE[2]:"I",CORE[3]:"Angle<5"}
for s in summary:
 if s["metric"] in CORE:
  share=s["top2_share_of_gross_parent_improvement"]
  share_text = f"{share:.1%}" if share is not None else "NA (no gain)"
  lines.append(f'| {s["baseline"]} | {s["ratio"]:.0%} | {s["scope"]} | {short[s["metric"]]} | {100*s["delta"]:+.4f} | {s["parent_improved"]}/{s["parent_worse"]}/{s["parent_tied"]} | {s["noise_improved"]}/{s["noise_worse"]}/{s["noise_tied"]} | {share_text} | {100*s["leave_one_parent_out_delta_min"]:+.4f} to {100*s["leave_one_parent_out_delta_max"]:+.4f} |')
lines+=["","## I components","","| Comparator | C | Stratum | Cross pp | Dup pp | Intersection pp |","|---|---|---|---:|---:|---:|"]
for b in BASES:
 for ratio in [.2,.4]:
  for scope in ["ALL","ORIGINAL6","EXTENDED2"]:
   ss={r["metric"]:r for r in summary if r["baseline"]==b and r["ratio"]==ratio and r["scope"]==scope}
   lines.append(f'| {b} | {ratio:.0%} | {scope} | {100*ss[PARTS[0]]["delta"]:+.4f} | {100*ss[PARTS[1]]["delta"]:+.4f} | {100*ss[PARTS[2]]["delta"]:+.4f} |')
lines+=["","## Diversity (variation, no good direction)","","| Comparator | C | Stratum | Mean D relative change | Parents lower/higher |","|---|---|---|---:|---|"]
for r in dsummary: lines.append(f'| {r["baseline"]} | {r["ratio"]:.0%} | {r["scope"]} | {r["aggregate_relative_change"]:+.2%} | {r["parent_reduced"]}/{r["parent_increased"]} |')
lines+=["","## Parent and noise tails","","Full per_noise.csv and per_parent.csv contain all selected components, including pair counts and unresolved/degenerate terms. The following lists the largest adverse paired-noise delta for each main cost and ratio, not selected attractive examples.",""]
for s in summary:
 if s["scope"]=="ALL" and s["metric"] in CORE:
  lines.append(f'- {s["baseline"]}, C={s["ratio"]:.0%}, {short[s["metric"]]}: worst noise {s["worst_noise_request"]}, delta {100*s["worst_noise_delta"]:+.4f} pp; worst parent {s["worst_parent"]}, parent delta {100*s["worst_parent_delta"]:+.4f} pp.')
lines+=["","## Audit and limitations","","Audit status: "+receipt["status"]+". "+str(checks)+" direct arithmetic checks against the prior noise/parent tables; "+str(len(raw))+" I-union checks; "+str(len(div))+" six-pair diversity means. Complete source hashes and exceptions are in audit.json. No training-repeat inference or automatic promotion is made. Exact lower cost does not certify coherent chairs or correct C use. The original-versus-extension A20 split is a parent-membership diagnostic, not a new sample design."]
(OUT/"AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
print(json.dumps(receipt,indent=2))
print("CORE_ALL",json.dumps([s for s in summary if s["scope"]=="ALL" and s["metric"] in CORE],indent=2))


