import json,hashlib,csv
from pathlib import Path
from datetime import datetime,timezone
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
d=r/"audit/visual_review/course/root"
files=sorted(p for p in d.glob("*.json") if p.name!="unblinding_receipt.json")
v={"status":"INITIAL_NOTES_SAVED_BEFORE_MAPPING_AND_COURSE_QUALITY_READ","unblinding_recorded_utc":datetime.now(timezone.utc).isoformat(),"method_mapping":{"Method 1":"C40_20_MIX","Method 2":"C20_40_MIX","Method 3":"R2_MIX_H_ALL"},"initial_notes_unchanged":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],"limitation":"AI reviewer saw earlier main evidence; alias concealment is not an independent blinded human study."}
(d/"unblinding_receipt.json").write_text(json.dumps(v,indent=2),encoding="utf8")
a=Path(json.loads((r/"analysis/course/latest.json").read_text())["analysis_dir"])
rr=list(csv.DictReader((a/"model_summary.csv").open(encoding="utf-8-sig")))
metrics=["ConflictCov_C_cross","ConflictCov_C_dup","ConflictCov_C_cross_and_dup","free_near_degenerate_fraction","free_exact_duplicate_excess_fraction","CF_coplanar_pair_count","FF_coplanar_pair_count","ConflictCov_C_uncertain_fraction"]
print(json.dumps([{k:x[k] for k in ["label","ratio","metric","mean"]} for x in rr if x["parent_stratum"]=="ALL" and x["metric"] in metrics]))
