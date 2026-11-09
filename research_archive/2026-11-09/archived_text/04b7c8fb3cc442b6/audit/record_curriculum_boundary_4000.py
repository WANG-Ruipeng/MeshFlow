import csv,json,datetime,hashlib
from pathlib import Path
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
def rows(a):
 with (r/"training"/a/"attempts/initial/training_log.csv").open() as f:
  return {int(x["step"]):x for x in csv.DictReader(f) if x.get("step","").isdigit() and x.get("label_batch_sha256")}
base=rows("R2_MIX_H_ALL");result={}
for a in ("C20_40_MIX","C40_20_MIX"):
 allrows=rows(a)
 boundary=[allrows[k] for k in (3999,4000,4001,4002)]
 for x in boundary:
  k=int(x["step"])
  assert int(x["generator_adamw_step"])==k and float(x["lambda_value"])==.25
  if k>4000:assert int(x["ratio20_samples"])==int(x["ratio40_samples"])==4
  else:assert (int(x["ratio20_samples"]),int(x["ratio40_samples"]))==((0,8) if a=="C20_40_MIX" else (8,0))
 keys=("shared_batch_sha256","input_batch_sha256","label_batch_sha256","free_coordinate_denominator")
 done=[k for k in sorted(allrows) if k>4000]
 assert all(all(allrows[k][v]==base[k][v] for v in keys) for k in done)
 result[a]=dict(boundary=boundary,completed_tail_updates_checked=len(done),tail_through_step=max(done),all_tail_batch_identities_equal_R2=True)
out=dict(status="PASS",checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=result,scope="Completed scalar logs only; no checkpoint or model load.")
p=r/"audit/curriculum_boundary_4000_actual.json"
if p.exists():raise RuntimeError("Already recorded")
p.write_text(json.dumps(out,indent=2),encoding="utf8")
print(json.dumps(dict(status="PASS",sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows={a:{k:v for k,v in x.items() if k!="boundary"} for a,x in result.items()})))
