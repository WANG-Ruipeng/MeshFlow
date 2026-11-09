import csv,json,hashlib
from pathlib import Path
RUN=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
D=RUN/"analysis/confirmation/recipe_confirmation_table_analysis_v1_0a05824cbfbe240f"
def rows(name):
 with (D/name).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
core=["CF_proper_affected_free_fraction","FF_illegal_affected_free_fraction","ConflictCov_C","free_minimum_angle_lt5_fraction"]
components=["ConflictCov_C_cross","ConflictCov_C_dup","ConflictCov_C_cross_and_dup"]
def selected(name,metrics,fields,all_only=False):
 return [{k:r.get(k) for k in fields} for r in rows(name) if r.get("metric") in metrics and (not all_only or r.get("parent_stratum")=="ALL")]
digest=dict(
 models=selected("model_summary.csv",core,["label","ratio","metric","parent_stratum","mean"]),
 differences=selected("paired_parent_summary.csv",core,["ratio","metric","parent_stratum","mean","negative_count","exact_zero_count","positive_count","leave_one_parent_out_min","leave_one_parent_out_max","parent_deltas","top2_share_of_gross_improvement"]),
 components=selected("model_summary.csv",components,["label","ratio","metric","parent_stratum","mean"],True),
 parent_differences=selected("paired_parent_metrics.csv",core,["ratio","metric","parent_id","first","second","delta"]),
 diversity_summary=rows("diversity_summary.csv"),
 diversity_difference=rows("diversity_paired_summary.csv"),
 noise_header=list(rows("paired_noise_metrics.csv")[0]),
 cross_repeat=selected("cross_repeat_group_effects.csv",core, list(rows("cross_repeat_group_effects.csv")[0]),True)
)
noise=rows("paired_noise_metrics.csv")
digest["noise_tails"]={}
for ratio in ("0.2","0.4"):
 for metric in core:
  rs=[r for r in noise if r["ratio"]==ratio and r["metric"]==metric]
  key=next(k for k in ("delta","difference","first_minus_second") if k in rs[0])
  rs=sorted(rs,key=lambda r:float(r[key]))
  digest["noise_tails"][ratio+"/"+metric]={"lowest":rs[:2],"highest":rs[-3:]}
digest["sources"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob("*.csv")}
dest=RUN/"analysis/CONFIRMATION_DECISION_DIGEST.json"
dest.write_text(json.dumps(digest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print(json.dumps({"path":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"status":"COMPLETE","new_scores":0,"model_calls":0}))
