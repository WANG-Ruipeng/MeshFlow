import json,csv
from pathlib import Path
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
c=Path(json.loads((r/"analysis/course/latest.json").read_text())["analysis_dir"])
a=Path(json.loads((r/"analysis/all_baselines/latest.json").read_text())["analysis_dir"])
keys=["CF_proper_affected_free_fraction","FF_illegal_affected_free_fraction","ConflictCov_C","free_minimum_angle_lt5_fraction"]
def rows(p):return list(csv.DictReader(p.open(encoding="utf-8-sig")))
def short(x):
 return {k:x[k] for k in ("first","second","ratio","metric","parent_stratum","mean","negative_count","exact_zero_count","positive_count","leave_one_parent_out_min","leave_one_parent_out_max","top2_share_of_gross_improvement") if k in x}
out={"course_pairs":[short(x) for x in rows(c/"paired_parent_summary.csv") if x["metric"] in keys and x["parent_stratum"]=="ALL"],
"historical_pairs":[short(x) for x in rows(a/"paired_parent_summary.csv") if x["metric"] in keys and x["parent_stratum"]=="ALL"],
"tails":[],"diversity":rows(c/"diversity_summary.csv")}
n=rows(c/"paired_noise_metrics.csv")
for first in ("C20_40_MIX","C40_20_MIX"):
 for ratio in ("0.2","0.4"):
  for metric in keys:
   rr=[x for x in n if x["first"]==first and x["second"]=="R2_MIX_H_ALL" and x["ratio"]==ratio and x["metric"]==metric]
   out["tails"]+=sorted(rr,key=lambda x:float(x["delta"]),reverse=True)[:2]
p=r/"analysis/COURSE_DECISION_DIGEST.json"
p.write_text(json.dumps(out,indent=2),encoding="utf8")
print(json.dumps(out))
